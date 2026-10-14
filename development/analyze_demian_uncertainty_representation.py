"""Exploratory representation and channel-relation analysis for fixed Demian traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from development.demian_v1_gate_state import V1_CHANNELS


def _centered(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64) - np.asarray(array, dtype=np.float64).mean(axis=0)


def _rms_spread(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(_centered(array) ** 2, axis=1))))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def pairwise_geometry_alignment(source: np.ndarray, target: np.ndarray) -> float:
    """Pearson alignment of upper-triangle pairwise distances."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape[0] != target.shape[0] or source.shape[0] < 3:
        raise ValueError("matched clouds require at least three paired histories")
    def distances(values):
        return np.linalg.norm(values[:, None, :] - values[None, :, :], axis=-1)[np.triu_indices(len(values), 1)]
    return _correlation(distances(source), distances(target))


def geometry_decomposition(source: np.ndarray, target: np.ndarray, *, rank: int) -> dict[str, float]:
    """Describe paired-cloud translation, scale, orientation and subspace overlap."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("source and target must be matched clouds with at least three histories")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    source_zero, target_zero = source-source_center, target-target_center
    source_spread, target_spread = _rms_spread(source), _rms_spread(target)
    if source_spread <= 1e-12 or target_spread <= 1e-12:
        raise ValueError("cloud spread must be nonzero")
    scale = target_spread / source_spread
    # The cloud has 64 histories but 1,312 coordinates. Economy QR reduces
    # the feature-space Procrustes nuclear norm to a <=64 square matrix.
    _, source_upper = np.linalg.qr(source_zero.T, mode="reduced")
    _, target_upper = np.linalg.qr(target_zero.T, mode="reduced")
    cross_singular_values = np.linalg.svd(source_upper @ target_upper.T, compute_uv=False)
    squared_residual = (float(np.sum((source_zero * scale) ** 2)) + float(np.sum(target_zero ** 2))
                        - 2.0 * scale * float(np.sum(cross_singular_values)))
    available_rank = min(rank, source_zero.shape[0] - 1, source_zero.shape[1])
    # This is overlap of the rank-k history-coordinate subspaces, not a claim
    # about the much larger feature-coordinate subspaces.
    source_basis = np.linalg.svd(source_zero, full_matrices=False)[0][:, :available_rank]
    target_basis = np.linalg.svd(target_zero, full_matrices=False)[0][:, :available_rank]
    singular_values = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    return {
        "sample_count": int(source.shape[0]),
        "centroid_shift": float(np.linalg.norm(target_center-source_center)),
        "source_spread": source_spread,
        "target_spread": target_spread,
        "scale_ratio": float(target_spread/source_spread),
        "procrustes_residual": float(np.sqrt(max(0.0, squared_residual) / source.shape[0]) / target_spread),
        "subspace_overlap": float(np.mean(singular_values ** 2)),
        "pairwise_geometry_alignment": pairwise_geometry_alignment(source, target),
        "rank": int(available_rank),
    }


def change_norm_relation(previous: dict[str, np.ndarray], current: dict[str, np.ndarray], left: str, right: str) -> dict[str, float]:
    """Descriptive relation between paired history-wise channel change norms."""
    left_delta = np.linalg.norm(np.asarray(current[left])-np.asarray(previous[left]), axis=1)
    right_delta = np.linalg.norm(np.asarray(current[right])-np.asarray(previous[right]), axis=1)
    if left_delta.shape != right_delta.shape:
        raise ValueError("channel histories must have the same count")
    return {"sample_count": int(len(left_delta)), "pearson_r": _correlation(left_delta, right_delta),
            "left_change_norm_mean": float(left_delta.mean()), "right_change_norm_mean": float(right_delta.mean())}


def _load_artifact(path: Path, expected_manifest_sha256: str):
    if not expected_manifest_sha256:
        raise ValueError("source manifest SHA256 is required")
    actual_manifest_sha256 = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("source manifest SHA256 mismatch")
    source_hashes = json.loads((path / "manifest.json").read_text()).get("file_sha256", {})
    for name in ("states.npz", "state_metadata.jsonl"):
        expected = source_hashes.get(name)
        actual = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise ValueError(f"source file SHA256 mismatch: {name}")
    states = np.load(path / "states.npz")["states"].astype(np.float64)
    metadata = [json.loads(line) for line in (path / "state_metadata.jsonl").read_text().splitlines()]
    if len(states) != len(metadata):
        raise ValueError("state and metadata row counts differ")
    return states, metadata


def _channels(states: np.ndarray, widths: list[int]) -> dict[str, np.ndarray]:
    offsets = np.cumsum([0, *widths])
    return {name: states[:, offsets[index]:offsets[index+1]] for index, name in enumerate(V1_CHANNELS)}


def _cell(states, metadata, seed, order, checkpoint):
    indices = [index for index, row in enumerate(metadata) if row["model_seed"] == seed and row["order"] == order and row["checkpoint_id"] == checkpoint]
    indices.sort(key=lambda index: int(metadata[index].get("paired_history_index", sum(int(bit) << (len(metadata[index]["symbols"])-1-position) for position, bit in enumerate(metadata[index]["symbols"])))))
    if not indices:
        raise ValueError(f"missing cell {seed}/{order}/{checkpoint}")
    return states[indices], [metadata[index] for index in indices]


def run_protocol(protocol: dict) -> dict:
    rank = int(protocol["subspace_rank"])
    rows, relation_rows = [], []
    for batch in protocol["batches"]:
        states, metadata = _load_artifact(Path(batch["artifact_directory"]), batch.get("source_manifest_sha256", ""))
        checkpoints = [point["id"] for point in batch["checkpoints"]]
        for seed in batch["model_seeds"]:
            for order in ("train", "heldout"):
                reference, reference_meta = _cell(states, metadata, seed, order, checkpoints[0])
                for checkpoint in checkpoints[1:]:
                    target, target_meta = _cell(states, metadata, seed, order, checkpoint)
                    if [row["symbols"] for row in reference_meta] != [row["symbols"] for row in target_meta]:
                        raise ValueError("history pairing changed across checkpoints")
                    rows.append({"batch": batch["id"], "model_seed": seed, "comparison": "within_order_time",
                                 "order": order, "source_checkpoint": checkpoints[0], "target_checkpoint": checkpoint,
                                 **geometry_decomposition(reference, target, rank=rank)})
                    old_channels, new_channels = _channels(reference, reference_meta[0]["channel_widths"]), _channels(target, target_meta[0]["channel_widths"])
                    for channel in V1_CHANNELS:
                        if channel == "gate":
                            continue
                        relation_rows.append({"batch": batch["id"], "model_seed": seed, "order": order,
                                              "source_checkpoint": checkpoints[0], "target_checkpoint": checkpoint,
                                              "relation": f"gate_change__{channel}_change",
                                              **change_norm_relation(old_channels, new_channels, "gate", channel)})
            for checkpoint in checkpoints:
                train, train_meta = _cell(states, metadata, seed, "train", checkpoint)
                heldout, heldout_meta = _cell(states, metadata, seed, "heldout", checkpoint)
                if [row["symbols"] for row in train_meta] != [row["symbols"] for row in heldout_meta]:
                    raise ValueError("history pairing changed across orders")
                rows.append({"batch": batch["id"], "model_seed": seed, "comparison": "cross_order",
                             "order": None, "source_checkpoint": checkpoint, "target_checkpoint": checkpoint,
                             **geometry_decomposition(train, heldout, rank=rank)})
    return {"source_commit": protocol.get("source_commit", "analysis-only"), "metrics": rows,
            "relation_metrics": relation_rows,
            "integrity": {"analysis_only": True, "architecture_changed": False, "weights_changed": False,
                          "metric_count": len(rows), "relation_metric_count": len(relation_rows),
                          "source_manifest_sha256": {batch["id"]: batch["source_manifest_sha256"] for batch in protocol["batches"]}}}


def write_artifacts(out: Path, protocol: dict, payload: dict) -> None:
    """Write analysis-only records without copying the source state arrays."""
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("metrics.jsonl", payload["metrics"]), ("relation_metrics.jsonl", payload["relation_metrics"])):
        (out / name).write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    (out / "summary.json").write_text(json.dumps({"experiment": protocol["protocol_id"], "claim_boundary": protocol["claim_boundary"],
        "metric_count": len(payload["metrics"]), "relation_metric_count": len(payload["relation_metrics"]), "integrity": payload["integrity"]},
        indent=2, sort_keys=True) + "\n")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in out.iterdir()}
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "source_commit": payload["source_commit"], "file_sha256": hashes}, indent=2, sort_keys=True) + "\n")


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
