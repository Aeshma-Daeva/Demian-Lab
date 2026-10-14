"""Train-only external decoder tracking over hash-pinned Demian state matrices.

This is a measurement package.  It never changes Demian or feeds a readout
back into it.  The dynamic observers are deliberately labelled external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from development.run_demian_uncertainty_localization import calibration_metrics, fit_probe, predict_probe, _shuffled_labels


def _history_index(row):
    return sum(int(bit) << (len(row["symbols"]) - 1 - position)
               for position, bit in enumerate(row["symbols"]))


def fit_tracker(source, target, *, mode):
    """Fit target-to-source translation/scale/(proper) rotation on calibration rows."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or len(source) < 3:
        raise ValueError("tracker requires matched source and target clouds")
    if mode not in {"identity", "scale_track", "orthogonal_track"}:
        raise ValueError("unknown tracking mode")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source-source_mean, target-target_mean
    target_norm = np.linalg.norm(target_zero)
    if target_norm <= 1e-12:
        raise ValueError("target calibration cloud has zero spread")
    scale = float(np.linalg.norm(source_zero) / target_norm)
    rotation = np.eye(source.shape[1])
    if mode == "orthogonal_track":
        left, _, right = np.linalg.svd((target_zero*scale).T @ source_zero, full_matrices=False)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
    if mode == "identity":
        source_mean, target_mean, scale = np.zeros_like(source_mean), np.zeros_like(target_mean), 1.0
    return {"mode": mode, "source_mean": source_mean, "target_mean": target_mean,
            "scale": scale, "rotation": rotation, "fit_sample_count": int(len(source))}


def apply_tracker(tracker, features):
    return ((np.asarray(features, dtype=np.float64)-tracker["target_mean"])*tracker["scale"]
            @ tracker["rotation"] + tracker["source_mean"])


def time_features(projected, rows, checkpoints):
    """Fixed checkpoint identity is supplied to an external observer only."""
    projected = np.asarray(projected, dtype=np.float64)
    output = np.zeros((len(rows), projected.shape[1]*len(checkpoints)))
    position = {checkpoint: index for index, checkpoint in enumerate(checkpoints)}
    for index, row in enumerate(rows):
        start = position[row["checkpoint_id"]]*projected.shape[1]
        output[index, start:start+projected.shape[1]] = projected[index]
    return output


def regime_features(projected, rows, *, controls=None):
    """Small frozen external state-conditioned feature map (3x projection width)."""
    projected = np.asarray(projected, dtype=np.float64)
    if controls is None:
        controls = np.asarray([[np.log1p(row["state_norm"]), row["gate_mean"]] for row in rows], dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.shape != (len(rows), 2):
        raise ValueError("regime controls require two values per row")
    mean, scale = controls.mean(axis=0), controls.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (controls-mean)/scale
    return np.hstack((projected, projected*normalized[:, :1], projected*normalized[:, 1:2]))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_batch(batch):
    directory = Path(batch["artifact_directory"])
    if not batch.get("source_manifest_sha256") or _sha(directory/"manifest.json") != batch["source_manifest_sha256"]:
        raise ValueError("source manifest SHA256 mismatch")
    manifest = json.loads((directory/"manifest.json").read_text())
    for name in ("states.npz", "state_metadata.jsonl", "summary.json"):
        if manifest.get("file_sha256", {}).get(name) != _sha(directory/name):
            raise ValueError(f"source file SHA256 mismatch: {name}")
    states = np.load(directory/"states.npz")["states"].astype(np.float64)
    metadata = [json.loads(line) for line in (directory/"state_metadata.jsonl").read_text().splitlines()]
    summary = json.loads((directory/"summary.json").read_text())
    if len(states) != len(metadata):
        raise ValueError("state and metadata row counts differ")
    if not np.isfinite(states).all():
        raise ValueError("source states contain nonfinite values")
    return states, metadata, summary["integrity"]["history_split"]


def _cell(states, metadata, seed, order, checkpoint):
    indexes = [i for i, row in enumerate(metadata) if row["model_seed"] == seed and row["order"] == order and row["checkpoint_id"] == checkpoint]
    indexes.sort(key=lambda i: _history_index(metadata[i]))
    histories = [_history_index(metadata[i]) for i in indexes]
    if len(histories) != len(set(histories)):
        raise ValueError(f"duplicate history in cell {seed}/{order}/{checkpoint}")
    return states[indexes], [metadata[i] for i in indexes]


def _basis(width, dimension, seed):
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(width, dimension)))[0]


def _controls(states, rows):
    widths = rows[0]["channel_widths"]
    gate_start = sum(widths[:3])
    gate_stop = gate_start + widths[3]
    return np.asarray([[np.linalg.norm(state), state[gate_start:gate_stop].mean()] for state in states], dtype=np.float64)


def _record(metrics, predictions, *, base, target, prediction):
    score_index, start = len(metrics), len(predictions)
    predictions.extend([[score_index, index, float(actual), float(value)] for index, (actual, value) in enumerate(zip(target, prediction, strict=True))])
    metrics.append({**base, "score_index": score_index, "prediction_row_start": start,
                    "prediction_row_stop": len(predictions), **calibration_metrics(target, prediction)})


def run_protocol(protocol):
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    checkpoints = protocol["checkpoints"]
    dimension, alpha = int(protocol["projection_dimension"]), float(protocol["ridge_alpha"])
    metrics, predictions, transforms = [], [], {}
    for batch in protocol["batches"]:
        states, metadata, split = _load_batch(batch)
        train_ids, eval_ids = set(split["train_history_indices"]), set(split["evaluation_history_indices"])
        for seed in batch["model_seeds"]:
            for projection_seed in protocol["projection_seeds"]:
                basis = _basis(states.shape[1], dimension, int(projection_seed))
                cells = {(order, checkpoint): _cell(states, metadata, seed, order, checkpoint)
                         for order in ("train", "heldout") for checkpoint in checkpoints}
                source_all, source_rows_all = cells[("train", checkpoints[0])]
                source_train = np.asarray([i for i, row in enumerate(source_rows_all) if _history_index(row) in train_ids])
                source_eval = np.asarray([i for i, row in enumerate(source_rows_all) if _history_index(row) in eval_ids])
                x_source = source_all[source_train] @ basis
                source_labels = np.asarray([source_rows_all[i]["target_value"] for i in source_train])
                frozen = fit_probe(x_source, source_labels, alpha=alpha)
                shuffled = fit_probe(x_source, _shuffled_labels([source_rows_all[i] for i in source_train], int(seed)*10000+int(projection_seed)), alpha=alpha)
                transforms[f"{batch['id']}_{seed}_{projection_seed}_basis"] = basis
                for label_kind, fitted in (("true", frozen), ("conditional_shuffle", shuffled)):
                    for name, value in fitted.items():
                        transforms[f"{batch['id']}_{seed}_{projection_seed}_cue6_{label_kind}_{name}"] = value
                # Fit every temporal tracker once on base-order train histories.
                # The same tracker is applied to the heldout order: it is a
                # strict portability score, not target-domain calibration.
                base_trackers = {}
                base_target, base_target_rows = cells[("train", checkpoints[0])]
                for checkpoint in checkpoints:
                    base_target, base_target_rows = cells[("train", checkpoint)]
                    base_target_train = np.asarray([i for i, row in enumerate(base_target_rows) if _history_index(row) in train_ids])
                    if [_history_index(source_rows_all[i]) for i in source_train] != [_history_index(base_target_rows[i]) for i in base_target_train]:
                        raise ValueError("base tracker history pairing changed")
                    base_trackers[checkpoint] = {mode: fit_tracker(x_source, (base_target @ basis)[base_target_train], mode=mode)
                                                 for mode in ("identity", "scale_track", "orthogonal_track")}
                for order in ("train", "heldout"):
                    for checkpoint in checkpoints:
                        target_all, target_rows_all = cells[(order, checkpoint)]
                        target_train = np.asarray([i for i, row in enumerate(target_rows_all) if _history_index(row) in train_ids])
                        target_eval = np.asarray([i for i, row in enumerate(target_rows_all) if _history_index(row) in eval_ids])
                        y = np.asarray([target_rows_all[i]["target_value"] for i in target_eval])
                        target_projected = target_all @ basis
                        for mode in ("identity", "scale_track", "orthogonal_track"):
                            tracker = base_trackers[checkpoint][mode]
                            key = f"{batch['id']}_{seed}_{projection_seed}_base_{checkpoint}_{mode}"
                            for name, value in tracker.items():
                                transforms[f"{key}_{name}"] = value
                            corrected = apply_tracker(tracker, target_projected[target_eval])
                            for label_kind, readout in (("true_labels", frozen), ("conditional_shuffle", shuffled)):
                                _record(metrics, predictions, base={"batch": batch["id"], "model_seed": seed, "projection_seed": projection_seed,
                                    "projection_dimension": dimension, "checkpoint_id": checkpoint, "evaluation_order": order,
                                    "decoder": "frozen_cue6", "tracking_mode": mode, "label_kind": label_kind,
                                    "scope": "primary" if mode == "orthogonal_track" else "control",
                                    "fit_history_indices": sorted(train_ids), "evaluation_history_indices": sorted(eval_ids)},
                                    target=y, prediction=predict_probe(readout, corrected))
                        latest = np.asarray([target_rows_all[i]["latest_semantic_log_odds"] for i in target_eval])
                        _record(metrics, predictions, base={"batch": batch["id"], "model_seed": seed, "projection_seed": projection_seed,
                            "projection_dimension": None, "checkpoint_id": checkpoint, "evaluation_order": order,
                            "decoder": "analytic_latest", "tracking_mode": "none", "label_kind": "public_baseline", "scope": "control",
                            "fit_history_indices": [], "evaluation_history_indices": sorted(eval_ids)}, target=y, prediction=latest)
                if not protocol.get("include_dynamic_observers", True):
                    continue
                # Dynamic observers: fit only base-order train histories across every checkpoint.
                dynamic_states, dynamic_rows = [], []
                for checkpoint in checkpoints:
                    cell_states, cell_rows = cells[("train", checkpoint)]
                    for index, row in enumerate(cell_rows):
                        if _history_index(row) in train_ids:
                            dynamic_states.append(cell_states[index]); dynamic_rows.append(row)
                dynamic_states = np.asarray(dynamic_states) @ basis
                dynamic_labels = np.asarray([row["target_value"] for row in dynamic_rows])
                original_train = np.vstack([cells[("train", checkpoint)][0][i] for checkpoint in checkpoints for i, row in enumerate(cells[("train", checkpoint)][1]) if _history_index(row) in train_ids])
                controls = _controls(original_train, dynamic_rows)
                feature_sets = {"time_conditioned_external": time_features(dynamic_states, dynamic_rows, checkpoints),
                                "regime_conditioned_external": regime_features(dynamic_states, dynamic_rows, controls=controls)}
                regime_mean, regime_scale = controls.mean(axis=0), np.where(controls.std(axis=0) > 1e-12, controls.std(axis=0), 1.)
                for decoder_name, features in feature_sets.items():
                    readout = fit_probe(features, dynamic_labels, alpha=alpha)
                    for name, value in readout.items(): transforms[f"{batch['id']}_{seed}_{projection_seed}_{decoder_name}_{name}"] = value
                    for order in ("train", "heldout"):
                        for checkpoint in checkpoints:
                            cell_states, cell_rows = cells[(order, checkpoint)]
                            evaluate = [i for i, row in enumerate(cell_rows) if _history_index(row) in eval_ids]
                            projected = cell_states[evaluate] @ basis
                            if decoder_name == "time_conditioned_external":
                                features_eval = time_features(projected, [cell_rows[i] for i in evaluate], checkpoints)
                            else:
                                raw_controls = _controls(cell_states[evaluate], [cell_rows[i] for i in evaluate])
                                normalized = (raw_controls-regime_mean)/regime_scale
                                features_eval = np.hstack((projected, projected*normalized[:, :1], projected*normalized[:, 1:2]))
                            y = np.asarray([cell_rows[i]["target_value"] for i in evaluate])
                            _record(metrics, predictions, base={"batch": batch["id"], "model_seed": seed, "projection_seed": projection_seed,
                                "projection_dimension": dimension, "checkpoint_id": checkpoint, "evaluation_order": order,
                                "decoder": decoder_name, "tracking_mode": "none", "label_kind": "true_labels", "scope": "secondary_external_upper_bound",
                                "fit_history_indices": sorted(train_ids), "evaluation_history_indices": sorted(eval_ids)}, target=y, prediction=predict_probe(readout, features_eval))
    return {"source_commit": source_commit, "dependency_versions": {"numpy": np.__version__},
            "metrics": metrics, "predictions": np.asarray(predictions), "transforms": transforms,
            "integrity": {"analysis_only": True, "architecture_changed": False, "weights_changed": False,
                          "decoder_feedback_to_demian": False, "metric_count": len(metrics),
                          "claim_boundary": "external offline decoder tracking; not Demian adaptation, learning, or self-readout"}}


def write_artifacts(out, protocol, payload):
    out.mkdir(parents=True, exist_ok=False)
    (out/"metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False)+"\n" for row in payload["metrics"]))
    np.savez(out/"predictions.npz", score_and_history_target_prediction=payload["predictions"])
    np.savez(out/"transforms.npz", **payload["transforms"])
    (out/"summary.json").write_text(json.dumps({"experiment": protocol["protocol_id"], "source_commit": payload["source_commit"],
        "dependency_versions": payload["dependency_versions"], **payload["integrity"]}, indent=2, sort_keys=True)+"\n")
    hashes = {path.name: _sha(path) for path in out.iterdir()}
    (out/"manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "source_commit": payload["source_commit"], "dependency_versions": payload["dependency_versions"], "file_sha256": hashes}, indent=2, sort_keys=True)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol); write_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), **payload["integrity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
