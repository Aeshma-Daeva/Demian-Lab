"""Frozen-fit residual descriptors for matched-target Demian transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from development.demian_v1_gate_state import V1_CHANNELS
from development.run_demian_uncertainty_decoder_tracking import apply_tracker
from development.run_demian_uncertainty_localization import fit_probe, predict_probe


def fit_channel_transport(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Fit a proper-orthogonal target-to-source channel map on calibration rows."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or len(source) < 3:
        raise ValueError("channel transport requires matched calibration clouds")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source - source_mean, target - target_mean
    target_norm = np.linalg.norm(target_zero)
    if target_norm <= 1e-12:
        raise ValueError("channel transport calibration cloud has zero spread")
    scale = float(np.linalg.norm(source_zero) / target_norm)
    left, _, right = np.linalg.svd((target_zero * scale).T @ source_zero, full_matrices=False)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return {"source_mean": source_mean, "target_mean": target_mean, "scale": scale,
            "rotation": rotation, "fit_sample_count": int(len(source))}


def apply_channel_transport(fit: dict[str, np.ndarray | float | int], target: np.ndarray) -> np.ndarray:
    return ((np.asarray(target, dtype=np.float64) - fit["target_mean"]) * float(fit["scale"])
            @ np.asarray(fit["rotation"]) + fit["source_mean"])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_matrix(batch: dict):
    directory = Path(batch["artifact_directory"])
    if _sha(directory / "manifest.json") != batch["source_manifest_sha256"]:
        raise ValueError("matrix manifest SHA256 mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("states.npz", "state_metadata.jsonl", "summary.json"):
        if manifest["file_sha256"].get(name) != _sha(directory / name):
            raise ValueError(f"matrix SHA256 mismatch: {name}")
    states = np.load(directory / "states.npz")["states"].astype(np.float64)
    metadata = [json.loads(line) for line in (directory / "state_metadata.jsonl").read_text().splitlines()]
    split = json.loads((directory / "summary.json").read_text())["integrity"]["history_split"]
    if len(states) != len(metadata) or not np.isfinite(states).all():
        raise ValueError("invalid matrix states")
    return states, metadata, split


def _paired(row: dict) -> int:
    return int(row.get("paired_history_index", sum(int(bit) << (len(row["symbols"]) - 1 - i)
                                                    for i, bit in enumerate(row["symbols"]))))


def _cell(states, metadata, seed, order, checkpoint):
    indexes = [i for i, row in enumerate(metadata) if row["model_seed"] == seed
               and row["order"] == order and row["checkpoint_id"] == checkpoint]
    indexes.sort(key=lambda i: _paired(metadata[i]))
    if len(indexes) != 64 or len({_paired(metadata[i]) for i in indexes}) != 64:
        raise ValueError(f"invalid history pairing: {seed}/{order}/{checkpoint}")
    return states[indexes], [metadata[i] for i in indexes]


def _slices(widths: list[int]) -> dict[str, slice]:
    offsets = np.cumsum([0, *widths])
    return {name: slice(int(offsets[i]), int(offsets[i + 1])) for i, name in enumerate(V1_CHANNELS)}


def _vector_r2(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    total = float(np.sum((actual - actual.mean(axis=0)) ** 2))
    return 1.0 - float(np.sum((actual - prediction) ** 2)) / total if total > 1e-12 else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _strict_pass(metrics: list[dict], seed: int, projection: int, checkpoint: str) -> bool:
    rows = [row for row in metrics if row["model_seed"] == seed and row.get("projection_seed") == projection
            and row["checkpoint_id"] == checkpoint and row["evaluation_order"] == "heldout"]
    true = next(row for row in rows if row["decoder"] == "frozen_cue6" and row["tracking_mode"] == "orthogonal_track"
                and row["label_kind"] == "true_labels")["r2"]
    shuffle = next(row for row in rows if row["decoder"] == "frozen_cue6" and row["tracking_mode"] == "orthogonal_track"
                   and row["label_kind"] == "conditional_shuffle")["r2"]
    latest = next(row for row in rows if row["decoder"] == "analytic_latest")["r2"]
    return bool(true > latest and true > shuffle)


def run_protocol(protocol: dict) -> dict:
    batch = protocol["batch"]
    states, metadata, split = _load_matrix(batch)
    tracking_directory = Path(protocol["tracking_artifact_directory"])
    if _sha(tracking_directory / "manifest.json") != protocol["tracking_manifest_sha256"]:
        raise ValueError("tracking manifest SHA256 mismatch")
    tracking = json.loads((tracking_directory / "manifest.json").read_text())
    if tracking["file_sha256"].get("metrics.jsonl") != _sha(tracking_directory / "metrics.jsonl"):
        raise ValueError("tracking metrics SHA256 mismatch")
    if tracking["file_sha256"].get("transforms.npz") != _sha(tracking_directory / "transforms.npz"):
        raise ValueError("tracking transforms SHA256 mismatch")
    metrics = [json.loads(line) for line in (tracking_directory / "metrics.jsonl").read_text().splitlines()]
    transforms = np.load(tracking_directory / "transforms.npz")
    train_ids, eval_ids = set(split["train_history_indices"]), set(split["evaluation_history_indices"])
    rows, saved_fits = [], {}
    for seed in batch["model_seeds"]:
        source_all, source_meta = _cell(states, metadata, seed, "train", "cue6")
        train_ix = np.asarray([i for i, row in enumerate(source_meta) if _paired(row) in train_ids])
        eval_ix = np.asarray([i for i, row in enumerate(source_meta) if _paired(row) in eval_ids])
        channel_slices = _slices(source_meta[0]["channel_widths"])
        for checkpoint in protocol["checkpoints"]:
            base_target_all, base_target_meta = _cell(states, metadata, seed, "train", checkpoint)
            target_all, target_meta = _cell(states, metadata, seed, "heldout", checkpoint)
            if [_paired(row) for row in source_meta] != [_paired(row) for row in base_target_meta]:
                raise ValueError("base-order history pairing changed")
            if [_paired(row) for row in source_meta] != [_paired(row) for row in target_meta]:
                raise ValueError("cross-order history pairing changed")
            channel_residuals = {}
            for channel, channel_slice in channel_slices.items():
                fit = fit_channel_transport(source_all[train_ix, channel_slice], base_target_all[train_ix, channel_slice])
                for name, value in fit.items():
                    saved_fits[f"{seed}_{checkpoint}_{channel}_transport_{name}"] = value
                channel_residuals[channel] = (apply_channel_transport(fit, target_all[eval_ix, channel_slice])
                                              - source_all[eval_ix, channel_slice])
            channel_energy = {channel: float(np.mean(np.sum(value ** 2, axis=1)))
                              for channel, value in channel_residuals.items()}
            total_channel_energy = sum(channel_energy.values())
            relationship = {}
            for left, right in (("carrier", "slow"), ("gate", "message")):
                left_norm = np.linalg.norm(channel_residuals[left], axis=1)
                right_norm = np.linalg.norm(channel_residuals[right], axis=1)
                # The predictor is fit on base-order calibration residuals only.
                base_left = (apply_channel_transport(fit_channel_transport(source_all[train_ix, channel_slices[left]], base_target_all[train_ix, channel_slices[left]]), base_target_all[train_ix, channel_slices[left]])
                             - source_all[train_ix, channel_slices[left]])
                base_right = (apply_channel_transport(fit_channel_transport(source_all[train_ix, channel_slices[right]], base_target_all[train_ix, channel_slices[right]]), base_target_all[train_ix, channel_slices[right]])
                              - source_all[train_ix, channel_slices[right]])
                predictor = fit_probe(base_left, base_right, alpha=float(protocol["relationship_ridge_alpha"]))
                for name, value in predictor.items():
                    saved_fits[f"{seed}_{checkpoint}_{left}_to_{right}_residual_{name}"] = value
                relationship[f"{left}_to_{right}_residual_prediction_r2"] = _vector_r2(base_right * 0 + channel_residuals[right], predict_probe(predictor, channel_residuals[left]))
                relationship[f"{left}_{right}_residual_norm_correlation"] = _correlation(left_norm, right_norm)
            for projection in protocol["projection_seeds"]:
                prefix = f"{batch['id']}_{seed}_{projection}_base_{checkpoint}_orthogonal_track"
                tracker = {name: transforms[f"{prefix}_{name}"] for name in ("source_mean", "target_mean", "scale", "rotation", "fit_sample_count")}
                basis = transforms[f"{batch['id']}_{seed}_{projection}_basis"]
                residual = apply_tracker(tracker, (target_all @ basis)[eval_ix]) - (source_all @ basis)[eval_ix]
                rows.append({"model_seed": seed, "checkpoint_id": checkpoint, "projection_seed": projection,
                             "strict_transport_pass": _strict_pass(metrics, seed, projection, checkpoint),
                             "full_projected_residual_energy": float(np.mean(np.sum(residual ** 2, axis=1))),
                             "channel_residual_energy_fraction": {channel: value / total_channel_energy if total_channel_energy else 0.0
                                                                 for channel, value in channel_energy.items()},
                             **relationship})
    return {"descriptors": rows, "fitted_objects": saved_fits, "integrity": {"analysis_only": True, "architecture_changed": False,
            "weights_changed": False, "state_count": len(states), "descriptor_count": len(rows),
            "matrix_manifest_sha256": batch["source_manifest_sha256"],
            "tracking_manifest_sha256": protocol["tracking_manifest_sha256"]}}


def write_artifacts(out: Path, protocol: dict, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "descriptors.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                                for row in payload["descriptors"]))
    np.savez(out / "fitted_objects.npz", **payload["fitted_objects"])
    (out / "summary.json").write_text(json.dumps({"experiment": protocol["protocol_id"],
        "claim_boundary": protocol["claim_boundary"], "integrity": payload["integrity"]}, indent=2, sort_keys=True) + "\n")
    hashes = {path.name: _sha(path) for path in out.iterdir()}
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "file_sha256": hashes}, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), **payload["integrity"]}, sort_keys=True))


if __name__ == "__main__":
    main()
