"""Frozen, descriptive relation-predictor screen for the final-slot contrast.

This runner never mutates Demian.  It saves the complete channel state and
implemented route tensors needed to reproduce every selected descriptive
metric.  Fitting uses only base-order training histories.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from development.demian_uncertainty_probe import encode_evidence
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, reference_belief
from development.demian_v1_gate_state import DemianV1GateState, V1_CHANNELS
from development.run_demian_uncertainty_decoder_tracking import apply_tracker, fit_tracker
from development.run_demian_uncertainty_localization import calibration_metrics, fit_probe, predict_probe, split_histories


CHECKPOINTS = ("cue6", "delay1", "delay2", "delay3", "delay4", "delay5", "delay6", "delay7", "delay8")
ROUTES = ("carrier_to_slow", "gate_modulates_carrier_to_slow", "message_to_carrier", "gate_modulates_message_to_carrier")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history_index(symbols):
    return sum(int(bit) << (5-position) for position, bit in enumerate(symbols))


def validate_protocol(protocol):
    if sorted(protocol["arms"]) != ["final05_base", "final05_upstream_swap12", "identity"]:
        raise ValueError("relation screen requires the three frozen arms")
    for name, arm in protocol["arms"].items():
        if len(arm["reliabilities"]) != 6 or sorted(arm["symbol_permutation"]) != list(range(6)):
            raise ValueError(f"invalid frozen arm: {name}")
    base, upstream = protocol["arms"]["final05_base"], protocol["arms"]["final05_upstream_swap12"]
    if base["reliabilities"][-1] != upstream["reliabilities"][-1] or base["symbol_permutation"][-1] != upstream["symbol_permutation"][-1]:
        raise ValueError("primary arms must match final evidence pair")
    if len(protocol["model_seeds"]) != 6:
        raise ValueError("screen requires six seeds")


def collect(protocol):
    """Collect all fixed public histories; no labels enter the model interface."""
    validate_protocol(protocol)
    hidden = int(protocol["hidden_size"])
    rows, states, routes = [], [], []
    for seed in protocol["model_seeds"]:
        for arm_name, arm in protocol["arms"].items():
            schedule, permutation = arm["reliabilities"], arm["symbol_permutation"]
            canonicals = list(itertools.product((0, 1), repeat=6))
            histories = [tuple(canonical[source] for source in permutation) for canonical in canonicals]
            frames_by_history = [AmbiguousEvidenceWorld(hidden_target=0, evidence=tuple(zip(symbols, schedule, strict=True)), delay_steps=8).frames()[:14]
                                 for symbols in histories]
            torch.manual_seed(int(seed)); model = DemianV1GateState(hidden, trace_routes=True)
            # Existing uncertainty episodes reset the same seed for every
            # history.  Repeat that one initial state across batch lanes; do
            # not accidentally turn history identity into random initialization.
            initial = model.initial_state(1, torch.device("cpu"))
            state = tuple(value.repeat(len(histories), 1) for value in initial)
            state_ticks, route_ticks = [], []
            for tick in range(14):
                encoded = torch.cat([encode_evidence(frames[tick], hidden) for frames in frames_by_history], dim=0)
                state = (state[0] + encoded, *state[1:]); state = model.step(state)
                if tick >= 5:
                    state_ticks.append(torch.cat(state, dim=1).detach().cpu().numpy())
                    trace = model.route_trace()
                    if trace is None:
                        raise RuntimeError("route trace missing")
                    route_ticks.append(np.stack([trace.values[route].detach().cpu().numpy() for route in ROUTES], axis=1))
            for index, (canonical, symbols, frames) in enumerate(zip(canonicals, histories, frames_by_history, strict=True)):
                rows.append({"model_seed": int(seed), "arm": arm_name, "paired_history_index": _history_index(canonical),
                             "symbols": list(symbols), "canonical_symbols": list(canonical), "reliabilities": schedule,
                             "target_value": float(reference_belief(frames[:6])["log_odds"]),
                             "latest_semantic_log_odds": float(reference_belief([frames[5]])["log_odds"])})
                states.append(np.stack(state_ticks, axis=1)[index]); routes.append(np.stack(route_ticks, axis=1)[index])
        print(json.dumps({"collected_seed": int(seed)}), flush=True)
    return np.asarray(states, dtype=np.float32), np.asarray(routes, dtype=np.float32), rows


def _cell(states, routes, rows, seed, arm):
    indexes = sorted((i for i, row in enumerate(rows) if row["model_seed"] == seed and row["arm"] == arm), key=lambda i: rows[i]["paired_history_index"])
    if [rows[i]["paired_history_index"] for i in indexes] != list(range(64)):
        raise ValueError(f"incomplete paired histories: {seed}/{arm}")
    return states[indexes], routes[indexes], [rows[i] for i in indexes]


def _basis(width, seed):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(width, 16)))[0]


def _ridge(x, y, alpha=1.0):
    return fit_probe(x, y, alpha=alpha)


def _route_auc(route_values):
    return np.linalg.norm(route_values, axis=-1).mean(axis=1)


def _rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranked[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranked


def _spearman(left, right):
    left, right = _rank(left), _rank(right)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def selection_gate(metrics):
    """Apply the frozen v2 leave-one-seed-out directional rule."""
    outcomes = {}
    for row in metrics:
        if row["kind"] == "outcome":
            outcomes.setdefault(row["model_seed"], {}).setdefault(row["projection_seed"], {})[row["arm"]] = row["r2"]
    outcome = {seed: float(np.median([cell["final05_base"] - cell["final05_upstream_swap12"] for cell in projections.values()]))
               for seed, projections in outcomes.items()}
    candidates = {}
    for row in metrics:
        if row["kind"] == "candidate":
            candidates.setdefault(row["metric"], {}).setdefault(row["model_seed"], {})[row["arm"]] = row["value"]
    report = []
    for metric, values in sorted(candidates.items()):
        seeds = sorted(values)
        contrast = np.asarray([values[seed]["final05_base"] - values[seed]["final05_upstream_swap12"] for seed in seeds])
        leave_one_out = []
        for omitted in range(len(seeds)):
            keep = [index for index in range(len(seeds)) if index != omitted]
            leave_one_out.append(_spearman(contrast[keep], np.asarray([outcome[seeds[index]] for index in keep])))
        full = _spearman(contrast, np.asarray([outcome[seed] for seed in seeds]))
        direction_counts = {str(direction): int(np.sum(np.sign(contrast) == direction)) for direction in (-1, 1)}
        loo_counts = {str(direction): int(sum(value is not None and np.sign(value) == direction for value in leave_one_out)) for direction in (-1, 1)}
        eligible = [direction for direction in (-1, 1) if direction_counts[str(direction)] >= 5 and loo_counts[str(direction)] >= 5]
        report.append({"metric": metric, "seeds": seeds, "contrasts": contrast.tolist(), "outcomes": [outcome[seed] for seed in seeds],
                       "full_spearman_descriptive": full, "leave_one_out_spearman": leave_one_out,
                       "direction_counts": direction_counts, "loo_direction_counts": loo_counts, "eligible_directions": eligible})
    eligible = [row["metric"] for row in report if row["eligible_directions"]]
    return {"outcomes": outcome, "candidates": report, "eligible_metrics_pending_tiebreak": eligible,
            "nominations": [], "selection_status": "pending_preregistered_standardized_tiebreak" if len(eligible) > 1 else ("no_eligible_metric" if not eligible else "single_metric_pending_review")}


def _fit_lag_model(source, target, train_ids):
    """Frozen scalar lag model: target norm from its prior norm plus source norm."""
    sn, tn = np.linalg.norm(source, axis=-1), np.linalg.norm(target, axis=-1)
    x = np.column_stack((tn[:, :-1].reshape(-1), sn[:, :-1].reshape(-1)))
    y = tn[:, 1:].reshape(-1)
    horizon = source.shape[1] - 1
    train_rows = np.concatenate([np.arange(index*horizon, (index+1)*horizon) for index in train_ids])
    return _ridge(x[train_rows], y[train_rows])


def _lag_deficit(fit, source, target, eval_ids):
    sn, tn = np.linalg.norm(source, axis=-1), np.linalg.norm(target, axis=-1)
    x = np.column_stack((tn[:, :-1].reshape(-1), sn[:, :-1].reshape(-1)))
    y = tn[:, 1:].reshape(-1)
    horizon = source.shape[1] - 1
    eval_rows = np.concatenate([np.arange(index*horizon, (index+1)*horizon) for index in eval_ids])
    return float(np.mean((y[eval_rows] - predict_probe(fit, x[eval_rows])) ** 2))


def analyze(protocol, states, routes, rows):
    identity_seed = int(protocol["history_split_seed"])
    _, _, reference = _cell(states, routes, rows, int(protocol["model_seeds"][0]), "final05_base")
    split = split_histories([{**row, "symbols": row["canonical_symbols"]} for row in reference], identity_seed)
    train_ids, eval_ids = set(split["train_history_indices"]), set(split["evaluation_history_indices"])
    metrics, fits = [], {}
    width = states.shape[-1]
    channel_offsets = {name: slice(i*int(protocol["hidden_size"]), (i+1)*int(protocol["hidden_size"])) for i, name in enumerate(V1_CHANNELS)}
    for seed in protocol["model_seeds"]:
        base_states, base_routes, base_rows = _cell(states, routes, rows, int(seed), "final05_base")
        upstream_states, upstream_routes, upstream_rows = _cell(states, routes, rows, int(seed), "final05_upstream_swap12")
        base_train = np.asarray([i for i, row in enumerate(base_rows) if row["paired_history_index"] in train_ids])
        base_eval = np.asarray([i for i, row in enumerate(base_rows) if row["paired_history_index"] in eval_ids])
        up_eval = np.asarray([i for i, row in enumerate(upstream_rows) if row["paired_history_index"] in eval_ids])
        outcome = []
        for projection_seed in (17, 29, 43):
            basis = _basis(width, projection_seed)
            readout = _ridge(base_states[base_train, 0] @ basis, np.asarray([base_rows[i]["target_value"] for i in base_train]))
            tracker = fit_tracker(base_states[base_train, 0] @ basis, base_states[base_train, -1] @ basis, mode="orthogonal_track")
            for name, data, indexes, cell_rows in (("final05_base", base_states, base_eval, base_rows), ("final05_upstream_swap12", upstream_states, up_eval, upstream_rows)):
                prediction = predict_probe(readout, apply_tracker(tracker, data[indexes, -1] @ basis))
                truth = np.asarray([cell_rows[i]["target_value"] for i in indexes])
                metrics.append({"model_seed": int(seed), "kind": "outcome", "metric": "true_r2", "arm": name, "projection_seed": projection_seed, **calibration_metrics(truth, prediction)})
                if name == "final05_base": outcome.append(metrics[-1]["r2"])
                else: outcome[-1] -= metrics[-1]["r2"]
        outcome_value = float(np.median(outcome))
        for channel in V1_CHANNELS:
            s = channel_offsets[channel]
            source = base_states[:, 0, s]; target = base_states[:, -1, s]
            tracker = fit_tracker(source[base_train], target[base_train], mode="orthogonal_track")
            for name, data, indexes in (("final05_base", base_states, base_eval), ("final05_upstream_swap12", upstream_states, up_eval)):
                residual = np.mean((apply_tracker(tracker, data[indexes, -1, s]) - data[indexes, 0, s]) ** 2)
                metrics.append({"model_seed": int(seed), "kind": "candidate", "metric": f"{channel}_endpoint_residual_fraction", "arm": name, "value": float(residual), "outcome_base_minus_upstream_r2": outcome_value})
        for route_index, relation in ((0, "carrier_to_slow"), (2, "gate_to_message")):
            for name, data, route_data, indexes in (("final05_base", base_states, base_routes, base_eval), ("final05_upstream_swap12", upstream_states, upstream_routes, up_eval)):
                metrics.append({"model_seed": int(seed), "kind": "candidate", "metric": f"{relation}_route_contribution_rms_auc", "arm": name, "value": float(_route_auc(route_data[indexes, :, route_index]).mean()), "outcome_base_minus_upstream_r2": outcome_value})
            source_name, target_name = (("carrier", "slow") if relation == "carrier_to_slow" else ("gate", "message"))
            ss, ts = channel_offsets[source_name], channel_offsets[target_name]
            lag_fit = _fit_lag_model(base_states[:, :, ss], base_states[:, :, ts], base_train)
            for name, data, indexes in (("final05_base", base_states, base_eval), ("final05_upstream_swap12", upstream_states, up_eval)):
                metrics.append({"model_seed": int(seed), "kind": "candidate", "metric": f"{relation}_frozen_lagged_prediction_deficit_auc", "arm": name, "value": _lag_deficit(lag_fit, data[:, :, ss], data[:, :, ts], indexes), "outcome_base_minus_upstream_r2": outcome_value})
    return {"metrics": metrics, "history_split": split, "fits": fits, "selection": selection_gate(metrics), "integrity": {"architecture_changed": False, "weights_changed": False, "analysis_only": True, "fit_boundary": "base-order train32 only"}}


def tiebreak_rows(protocol, states, routes, rows):
    """Atomic base-train dispersions for the v2 hash-pinned tie-break."""
    _, _, reference = _cell(states, routes, rows, int(protocol["model_seeds"][0]), "final05_base")
    split = split_histories([{**row, "symbols": row["canonical_symbols"]} for row in reference], int(protocol["history_split_seed"]))
    train_ids, eval_ids = set(split["train_history_indices"]), set(split["evaluation_history_indices"])
    offsets = {name: slice(i*int(protocol["hidden_size"]), (i+1)*int(protocol["hidden_size"])) for i, name in enumerate(V1_CHANNELS)}
    output = []
    for seed in protocol["model_seeds"]:
        base, base_routes, base_rows = _cell(states, routes, rows, int(seed), "final05_base")
        upstream, upstream_routes, upstream_rows = _cell(states, routes, rows, int(seed), "final05_upstream_swap12")
        train = np.asarray([i for i, row in enumerate(base_rows) if row["paired_history_index"] in train_ids])
        evaluate = np.asarray([i for i, row in enumerate(base_rows) if row["paired_history_index"] in eval_ids])
        upstream_evaluate = np.asarray([i for i, row in enumerate(upstream_rows) if row["paired_history_index"] in eval_ids])
        for channel in V1_CHANNELS:
            s = offsets[channel]; tracker = fit_tracker(base[train, 0, s], base[train, -1, s], mode="orthogonal_track")
            atom = lambda data, indexes: np.mean((apply_tracker(tracker, data[indexes, -1, s]) - data[indexes, 0, s]) ** 2, axis=1)
            train_values, base_values, upstream_values = atom(base, train), atom(base, evaluate), atom(upstream, upstream_evaluate)
            output.append({"model_seed": int(seed), "metric": f"{channel}_endpoint_residual_fraction", "base_train_sd": float(np.std(train_values)), "base_value": float(np.mean(base_values)), "upstream_value": float(np.mean(upstream_values))})
        for route_index, relation in ((0, "carrier_to_slow"), (2, "gate_to_message")):
            atom = lambda route_data, indexes: _route_auc(route_data[indexes, :, route_index])
            train_values, base_values, upstream_values = atom(base_routes, train), atom(base_routes, evaluate), atom(upstream_routes, upstream_evaluate)
            output.append({"model_seed": int(seed), "metric": f"{relation}_route_contribution_rms_auc", "base_train_sd": float(np.std(train_values)), "base_value": float(np.mean(base_values)), "upstream_value": float(np.mean(upstream_values))})
            source_name, target_name = (("carrier", "slow") if relation == "carrier_to_slow" else ("gate", "message"))
            ss, ts = offsets[source_name], offsets[target_name]
            lag_fit = _fit_lag_model(base[:, :, ss], base[:, :, ts], train)
            def lag_atoms(data, indexes):
                sn, tn = np.linalg.norm(data[:, :, ss], axis=-1), np.linalg.norm(data[:, :, ts], axis=-1)
                x = np.stack((tn[:, :-1], sn[:, :-1]), axis=-1)
                prediction = predict_probe(lag_fit, x.reshape(-1, 2)).reshape(x.shape[:2])
                return np.mean((tn[:, 1:] - prediction) ** 2, axis=1)[indexes]
            train_values, base_values, upstream_values = lag_atoms(base, train), lag_atoms(base, evaluate), lag_atoms(upstream, upstream_evaluate)
            output.append({"model_seed": int(seed), "metric": f"{relation}_frozen_lagged_prediction_deficit_auc", "base_train_sd": float(np.std(train_values)), "base_value": float(np.mean(base_values)), "upstream_value": float(np.mean(upstream_values))})
    return output


def write_artifacts(out, protocol, states, routes, rows, payload):
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "states_and_routes.npz", states=states, routes=routes)
    (out / "metadata.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    (out / "metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in payload["metrics"]))
    (out / "summary.json").write_text(json.dumps({"experiment": protocol["protocol_id"], "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), **payload["integrity"], "history_split": payload["history_split"], "selection": payload["selection"]}, indent=2, sort_keys=True) + "\n")
    hashes = {path.name: _sha(path) for path in out.iterdir()}
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(), "file_sha256": hashes}, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text()); states, routes, rows = collect(protocol); payload = analyze(protocol, states, routes, rows); write_artifacts(args.out, protocol, states, routes, rows, payload)
    print(json.dumps({"out": str(args.out), "metric_count": len(payload["metrics"]), **payload["integrity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
