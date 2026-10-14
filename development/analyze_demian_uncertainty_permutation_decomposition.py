"""Analysis-only continuous transport and centered geometry summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def geometry_descriptors(source, target, *, rank=16):
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("paired state clouds require equal two-dimensional shapes")
    source_centered, target_centered = source-source.mean(axis=0), target-target.mean(axis=0)
    count = min(rank, *source_centered.shape, *target_centered.shape)
    source_axes = np.linalg.svd(source_centered, full_matrices=False)[2][:count]
    target_axes = np.linalg.svd(target_centered, full_matrices=False)[2][:count]
    overlap = np.linalg.svd(source_axes @ target_axes.T, compute_uv=False).mean()
    return {
        "centroid_displacement": float(np.linalg.norm(source.mean(axis=0)-target.mean(axis=0))),
        "source_centered_spread": float(np.sqrt(np.mean(np.sum(source_centered**2, axis=1)))),
        "target_centered_spread": float(np.sqrt(np.mean(np.sum(target_centered**2, axis=1)))),
        "centered_subspace_overlap": float(overlap),
    }


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _score(rows, *, seed, projection, checkpoint, decoder, tracking_mode, label_kind):
    return next(row["r2"] for row in rows if row["model_seed"] == seed and row["projection_seed"] == projection
                and row["evaluation_order"] == "heldout" and row["checkpoint_id"] == checkpoint
                and row["decoder"] == decoder and row["tracking_mode"] == tracking_mode and row["label_kind"] == label_kind)


def analyze(root):
    root = Path(root)
    output = {"conditions": {}}
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        condition = condition_dir.name
        rows = _rows(condition_dir / "tracking" / "metrics.jsonl")
        seeds = sorted({row["model_seed"] for row in rows})
        projections = sorted({row["projection_seed"] for row in rows if row["projection_seed"] is not None})
        checkpoints = ["delay2", "delay8"]
        transport = {}
        for checkpoint in checkpoints:
            per_seed = []
            for seed in seeds:
                cells = []
                for projection in projections:
                    true = _score(rows, seed=seed, projection=projection, checkpoint=checkpoint,
                                  decoder="frozen_cue6", tracking_mode="orthogonal_track", label_kind="true_labels")
                    latest = _score(rows, seed=seed, projection=projection, checkpoint=checkpoint,
                                    decoder="analytic_latest", tracking_mode="none", label_kind="public_baseline")
                    shuffle = _score(rows, seed=seed, projection=projection, checkpoint=checkpoint,
                                     decoder="frozen_cue6", tracking_mode="orthogonal_track", label_kind="conditional_shuffle")
                    cells.append([true, latest, shuffle, true-latest, true-shuffle, true-max(latest, shuffle)])
                median = np.median(cells, axis=0)
                per_seed.append({"model_seed": seed, "true_r2": float(median[0]), "latest_r2": float(median[1]),
                                 "shuffle_r2": float(median[2]), "true_minus_latest": float(median[3]),
                                 "true_minus_shuffle": float(median[4]), "continuous_transport_margin": float(median[5])})
            transport[checkpoint] = {"per_seed": per_seed,
                                     "mean": {key: float(np.mean([row[key] for row in per_seed])) for key in per_seed[0] if key != "model_seed"}}
        states = np.load(condition_dir / "matrix" / "states.npz")["states"]
        metadata = _rows(condition_dir / "matrix" / "state_metadata.jsonl")
        geometry = {}
        for checkpoint in checkpoints:
            per_seed = []
            for seed in seeds:
                indexes = {order: sorted([i for i, row in enumerate(metadata) if row["model_seed"] == seed and row["checkpoint_id"] == checkpoint and row["order"] == order], key=lambda i: metadata[i]["paired_history_index"])
                           for order in ("train", "heldout")}
                per_seed.append({"model_seed": seed, **geometry_descriptors(states[indexes["train"]], states[indexes["heldout"]])})
            geometry[checkpoint] = {"per_seed": per_seed,
                                    "mean": {key: float(np.mean([row[key] for row in per_seed])) for key in per_seed[0] if key != "model_seed"}}
        output["conditions"][condition] = {"transport": transport, "geometry": geometry}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.root), indent=2, sort_keys=True) + "\n")
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["analysis_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
