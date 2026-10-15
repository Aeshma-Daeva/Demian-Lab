"""Analysis-only continuous transport and centered geometry summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from development.run_demian_uncertainty_decoder_tracking import apply_tracker


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


def frozen_geometry_descriptors(source_train, source_eval, target_eval, *, rank):
    """Evaluate target coordinates against a base-train-only centered subspace."""
    source_train = np.asarray(source_train, dtype=np.float64)
    source_eval, target_eval = np.asarray(source_eval, dtype=np.float64), np.asarray(target_eval, dtype=np.float64)
    if source_train.ndim != 2 or source_eval.shape != target_eval.shape or source_eval.shape[1] != source_train.shape[1]:
        raise ValueError("frozen geometry requires compatible state matrices")
    mean = source_train.mean(axis=0)
    axes = np.linalg.svd(source_train-mean, full_matrices=False)[2][:min(rank, *source_train.shape)]
    source_centered, target_centered = source_eval-source_eval.mean(axis=0), target_eval-target_eval.mean(axis=0)
    source_spread = np.sqrt(np.mean(np.sum(source_centered**2, axis=1)))
    target_spread = np.sqrt(np.mean(np.sum(target_centered**2, axis=1)))
    target_relative_to_base = target_eval-mean
    residual = target_relative_to_base - (target_relative_to_base @ axes.T) @ axes
    return {"tracker_residual_energy": float(np.mean(np.sum((target_eval-source_eval)**2, axis=1))),
            "centroid_residual_norm": float(np.linalg.norm(target_eval.mean(axis=0)-source_eval.mean(axis=0))),
            "centered_spread_ratio": float(target_spread/source_spread) if source_spread > 1e-12 else 1.,
            "subspace_reconstruction_error": float(np.mean(np.sum(residual**2, axis=1)))}


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


def analyze_frozen_geometry(root, *, rank=8):
    """Base-train-only delay8 geometry for each saved permutation condition."""
    root = Path(root)
    output = {"conditions": {}}
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        condition = condition_dir.name
        matrix_dir, tracking_dir = condition_dir / "matrix", condition_dir / "tracking"
        states = np.load(matrix_dir / "states.npz")["states"].astype(np.float64)
        metadata = _rows(matrix_dir / "state_metadata.jsonl")
        split = json.loads((matrix_dir / "summary.json").read_text())["integrity"]["history_split"]
        train_ids, eval_ids = set(split["train_history_indices"]), set(split["evaluation_history_indices"])
        transforms = np.load(tracking_dir / "transforms.npz")
        seeds = sorted({row["model_seed"] for row in metadata})
        projections = sorted({int(key.removeprefix(condition+"_").split("_")[1])
                              for key in transforms.files if key.startswith(condition+"_") and key.endswith("_basis")})
        per_seed = []
        for seed in seeds:
            per_projection = []
            for projection in projections:
                basis = transforms[f"{condition}_{seed}_{projection}_basis"]
                def cell(order, checkpoint, ids):
                    indexes = sorted([i for i, row in enumerate(metadata) if row["model_seed"] == seed and row["order"] == order and row["checkpoint_id"] == checkpoint and row["paired_history_index"] in ids], key=lambda i: metadata[i]["paired_history_index"])
                    return states[indexes] @ basis
                source_train, source_eval = cell("train", "cue6", train_ids), cell("train", "cue6", eval_ids)
                target_eval = cell("heldout", "delay8", eval_ids)
                prefix = f"{condition}_{seed}_{projection}_base_delay8_orthogonal_track_"
                tracker = {name: transforms[prefix+name] for name in ("source_mean", "target_mean", "scale", "rotation")}
                transformed = apply_tracker(tracker, target_eval)
                per_projection.append(frozen_geometry_descriptors(source_train, source_eval, transformed, rank=rank))
            per_seed.append({"model_seed": seed, **{key: float(np.median([row[key] for row in per_projection])) for key in per_projection[0]}})
        output["conditions"][condition] = {"per_seed": per_seed,
            "mean": {key: float(np.mean([row[key] for row in per_seed])) for key in per_seed[0] if key != "model_seed"}}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-geometry", action="store_true")
    args = parser.parse_args()
    payload = analyze_frozen_geometry(args.root) if args.frozen_geometry else analyze(args.root)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["analysis_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
