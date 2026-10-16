"""Frozen-Demian delayed-cue external readout under declared read distractions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from development.closed_loop_demian import FrozenObservationEncoder
from development.closed_loop_world import ActionAcknowledgement, WorldObservation
from development.demian_v1_gate_state import DemianV1GateState
from development.run_demian_uncertainty_localization import fit_probe, predict_probe


def basis(width: int, seed: int) -> np.ndarray:
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(width, 16)))[0]


def latest_seen(history: np.ndarray, delay: int) -> int:
    """Return the last acknowledgement available at a delay-prefix query."""
    return int(history[delay - 1])


def _validate_protocol(protocol: dict) -> None:
    if sorted(protocol["symbols"]) != list(range(len(protocol["symbols"]))):
        raise ValueError("symbols must be contiguous categorical labels")
    if len(protocol["train_history_ids"]) * 2 != protocol["histories_per_cue_delay"]:
        raise ValueError("protocol requires an exactly balanced train/evaluation split")
    if len(set(protocol["train_history_ids"])) != len(protocol["train_history_ids"]):
        raise ValueError("duplicate training history id")
    if any(i < 0 or i >= protocol["histories_per_cue_delay"] for i in protocol["train_history_ids"]):
        raise ValueError("training history id outside declared corpus")


def _feature_vector(model: DemianV1GateState, state: tuple[torch.Tensor, ...], feature_source: str) -> np.ndarray:
    if feature_source == "surface":
        return model.state_vector(state).detach().numpy()
    if feature_source == "full_state":
        return torch.cat(state, dim=1).detach().numpy()
    raise ValueError(f"unknown feature_source: {feature_source}")


def collect(protocol: dict) -> tuple[np.ndarray, list[dict]]:
    """Collect query-time states without updating Demian or its encoder."""
    rows, states = [], []
    hidden_size, count = int(protocol["hidden_size"]), int(protocol["histories_per_cue_delay"])
    for seed in protocol["model_seeds"]:
        for cue in protocol["symbols"]:
            max_delay = max(protocol["delays"])
            rng = np.random.default_rng(int(seed) * 100000 + int(cue))
            histories = np.column_stack([rng.permutation(np.tile(np.asarray(protocol["symbols"]), count // len(protocol["symbols"])))
                                         for _ in range(max_delay)])
            for delay in protocol["delays"]:
                torch.manual_seed(int(seed))
                model = DemianV1GateState(hidden_size)
                encoder = FrozenObservationEncoder(hidden_size=hidden_size, symbol_count=len(protocol["symbols"]))
                state = tuple(part.repeat(count, 1) for part in model.initial_state(1, torch.device("cpu")))
                for tick in range(delay + 1):
                    observation = WorldObservation(tick, "cue" if tick == 0 else "delay", cue if tick == 0 else None, False)
                    acknowledgements = ([None] * count if tick == 0 else [
                        ActionAcknowledgement("read", int(symbol), int(symbol), None)
                        for symbol in histories[:, tick - 1]
                    ])
                    inputs = torch.cat([encoder.encode(observation, acknowledgement) for acknowledgement in acknowledgements])
                    state = model.step(encoder.inject(state, inputs))
                query = WorldObservation(delay + 1, "query", None, True)
                state = model.step(encoder.inject(state, torch.cat([encoder.encode(query, None) for _ in range(count)])))
                vectors = _feature_vector(model, state, protocol.get("feature_source", "surface"))
                for history_id in range(count):
                    rows.append({"seed": int(seed), "delay": int(delay), "cue": int(cue), "id": history_id,
                                 "latest": latest_seen(histories[history_id], int(delay))})
                    states.append(vectors[history_id])
        print(json.dumps({"seed": int(seed), "status": "collected"}), flush=True)
    return np.asarray(states), rows


def fit_and_score(states: np.ndarray, rows: list[dict], protocol: dict) -> tuple[list[dict], dict, list[dict]]:
    """Fit only declared training histories and return frozen objects/predictions."""
    train_ids = set(protocol["train_history_ids"])
    projection = basis(states.shape[1], int(protocol["projection_seed"]))
    metrics, fits, predictions = [], {}, []
    category_count = len(protocol["symbols"])
    for seed in protocol["model_seeds"]:
        seed_indices = [i for i, row in enumerate(rows) if row["seed"] == seed]
        train_indices = [i for i in seed_indices if rows[i]["id"] in train_ids]
        targets = np.eye(category_count)[[rows[i]["cue"] for i in train_indices]]
        demian_fit = fit_probe(states[train_indices] @ projection, targets, alpha=float(protocol["ridge_alpha"]))
        latest_fit = fit_probe(np.eye(category_count)[[rows[i]["latest"] for i in train_indices]], targets,
                               alpha=float(protocol["ridge_alpha"]))
        fits[str(seed)] = {"demian": demian_fit, "latest": latest_fit}
        for delay in protocol["delays"]:
            evaluation = [i for i in seed_indices if rows[i]["delay"] == delay and rows[i]["id"] not in train_ids]
            target = np.asarray([rows[i]["cue"] for i in evaluation])
            demian_scores = predict_probe(demian_fit, states[evaluation] @ projection)
            latest_scores = predict_probe(latest_fit, np.eye(category_count)[[rows[i]["latest"] for i in evaluation]])
            demian_class, latest_class = np.argmax(demian_scores, axis=1), np.argmax(latest_scores, axis=1)
            demian_accuracy, latest_accuracy = float((demian_class == target).mean()), float((latest_class == target).mean())
            no_storage_accuracy = float((target == 0).mean())
            metrics.append({"seed": int(seed), "delay": int(delay), "demian": demian_accuracy,
                            "latest": latest_accuracy, "no_storage": no_storage_accuracy,
                            "margin_over_latest": demian_accuracy - latest_accuracy,
                            "margin_over_no_storage": demian_accuracy - no_storage_accuracy})
            predictions.extend({"seed": int(seed), "delay": int(delay), "row": int(index), "target": int(label),
                                "demian_class": int(predicted), "latest_class": int(latest),
                                "demian_scores": score.tolist(), "latest_scores": latest_score.tolist()}
                               for index, label, predicted, latest, score, latest_score in
                               zip(evaluation, target, demian_class, latest_class, demian_scores, latest_scores, strict=True))
    return metrics, {"projection": projection, "fits": fits}, predictions


def _save_fits(path: Path, frozen: dict) -> None:
    arrays = {"projection": frozen["projection"]}
    for seed, models in frozen["fits"].items():
        for model_name, model in models.items():
            for field in ("mean", "scale", "weights"):
                arrays[f"{seed}_{model_name}_{field}"] = model[field]
    np.savez_compressed(path, **arrays)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    _validate_protocol(protocol)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite artifact directory: {args.out}")
    states, rows = collect(protocol)
    metrics, frozen, predictions = fit_and_score(states, rows, protocol)
    delay_gate = {}
    for delay in protocol["delays"]:
        cells = [cell for cell in metrics if cell["delay"] == delay]
        delay_gate[str(delay)] = sum(cell["demian"] >= .5 and cell["margin_over_latest"] >= .15 and
                                     cell["margin_over_no_storage"] >= .15 for cell in cells) >= 5
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "states.npz", states=states)
    _save_fits(args.out / "frozen_objects.npz", frozen)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.out / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    summary = {"protocol_id": protocol["protocol_id"], "metrics": metrics, "delay_gate_pass": delay_gate,
               "overall_pass": all(delay_gate.values()), "file_hashes": {}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["file_hashes"] = {path.name: _sha256(path) for path in args.out.iterdir()
                              if path.is_file() and path.name != "summary.json"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
