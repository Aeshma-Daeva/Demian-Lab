"""Prospective fixed-feature diagnosis using new seeds and frozen training fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from development.demian_cross_seed_transfer import BITS, HISTORIES, feature_maps, r2
from development.run_demian_cross_seed_transfer import (
    collect_demian, collect_reservoir, dump, load_collected, make_split,
    save_collection, sha,
)
from development.run_demian_uncertainty_localization import fit_probe, predict_probe


def contrast(distances):
    squared = np.asarray(distances, dtype=np.float64)**2
    return squared-squared.mean(axis=1, keepdims=True)


def fit_whitener(features):
    x = np.asarray(features, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite whitening features")
    mean = x.mean(0)
    covariance = (x-mean).T@(x-mean)/len(x)
    eigen, vectors = np.linalg.eigh(covariance)
    keep = eigen > max(0., eigen[-1]*1e-8)
    weights = np.zeros_like(eigen)
    weights[keep] = 1/np.sqrt(eigen[keep])
    return mean, (vectors*weights)@vectors.T


def maps_for(x, split):
    maps, info = feature_maps(x, split, np.zeros((x.shape[1], 1)))
    c = contrast(maps["anchor"])
    return {"distance": maps["anchor"], "contrast": c, "contrast_white": c,
            "affine_contrast_white": contrast(maps["linear"])}, info


def fit_variant(x, y, whiten):
    mean, matrix = fit_whitener(x) if whiten else (np.zeros(x.shape[1]), np.eye(x.shape[1]))
    fitted = fit_probe((x-mean)@matrix, y, alpha=1)
    return {"white_mean": mean, "white_matrix": matrix, **fitted}


def predict_variant(fitted, x):
    transformed = (x-fitted["white_mean"])@fitted["white_matrix"]
    return predict_probe({k: fitted[k] for k in ("mean", "scale", "weights")}, transformed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    original = json.loads(Path(protocol["source_protocol"]).read_text())
    if set(protocol["evaluation_seeds"]) & set(original["train_seeds"]+original["evaluation_seeds"]):
        raise ValueError("new evaluation seeds overlap earlier seeds")
    if protocol["train_seeds"] != original["train_seeds"] or protocol["delays"] != original["delays"]:
        raise ValueError("training boundary or checkpoint mismatch")
    torch.set_num_threads(1)
    source = Path(protocol["training_artifact"])
    corpus, _ = load_collected(source, original)
    out = Path(protocol["artifact_directory"])
    out.mkdir(parents=True, exist_ok=False)
    dump(out/"protocol.json", protocol)
    dump(out/"inherited_protocol.json", original)
    execution = {"status": "running", "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                 "numpy": np.__version__, "torch": str(torch.__version__), "threads": 1,
                 "training_collection_manifest_sha256": sha(source/"collection_manifest.json")}
    dump(out/"execution.json", execution)
    (out/"source").mkdir()
    for file in (Path(__file__), Path("development/demian_cross_seed_transfer.py"),
                 Path("development/run_demian_cross_seed_transfer.py"), Path("development/cross_seed_reservoir_reference.py"),
                 Path("development/demian_v1_gate_state.py"), Path("development/run_demian_uncertainty_localization.py")):
        (out/"source"/file.name).write_bytes(file.read_bytes())
    split = make_split(original["split_seed"])
    dump(out/"split.json", split)
    y = BITS@np.log(np.array(original["reliabilities"])/(1-np.array(original["reliabilities"])))
    fit_ids, evaluate = split["fit"], split["evaluation"]
    with np.load(source/"fit_demian_2.npz") as saved:
        shuffled = saved["shuffled_training_labels"]
    objects = {}
    for model in ("demian", "reservoir"):
        for ti, delay in enumerate(protocol["delays"]):
            maps = [maps_for(corpus[model][s][:, ti], split)[0] for s in protocol["train_seeds"]]
            for variant in protocol["variants"]:
                x = np.concatenate([m[variant][fit_ids] for m in maps])
                for labels, target in (("true", y[fit_ids]), ("shuffle", shuffled)):
                    fitted = fit_variant(x, np.tile(target, len(maps)), variant.endswith("white"))
                    key = f"{model}_{delay}_{variant}_{labels}"
                    objects[key] = fitted
                    np.savez_compressed(out/f"fit_{key}.npz", **fitted)
    fit_hashes = {p.name: sha(p) for p in out.glob("fit_*.npz")}
    dump(out/"frozen_fit_manifest.json", fit_hashes)
    print(json.dumps({"stage": "fits_frozen", "count": len(objects)}), flush=True)
    metrics, predictions = [], {"targets": y[evaluate], "history_ids": np.array(evaluate)}
    permutations = [np.random.default_rng(original["anchor_permutation_seed"]+i).permutation(16)
                    for i in range(original["anchor_permutation_count"])]
    for seed in protocol["evaluation_seeds"]:
        for model in ("demian", "reservoir"):
            if model == "demian":
                states, meta = collect_demian(seed, HISTORIES, original["reliabilities"], protocol["delays"], original["hidden_size"])
            else:
                states, meta = collect_reservoir(seed, HISTORIES, original["reliabilities"], protocol["delays"])
            save_collection(out, model, seed, states, meta)
            for ti, delay in enumerate(protocol["delays"]):
                maps, info = maps_for(states[:, ti], split)
                for variant in protocol["variants"]:
                    key = f"{model}_{delay}_{variant}"
                    x = maps[variant]
                    true = predict_variant(objects[f"{key}_true"], x)[evaluate]
                    shuffle = predict_variant(objects[f"{key}_shuffle"], x)[evaluate]
                    perm = np.stack([predict_variant(objects[f"{key}_true"], x[:, p])[evaluate] for p in permutations])
                    score, null = r2(y[evaluate], true), r2(y[evaluate], shuffle)
                    q95 = float(np.quantile([r2(y[evaluate], p) for p in perm], .95))
                    passed = not info["precision_limited"] and score >= .1 and score >= max(0., null, q95)+.1
                    metrics.append({"model": model, "seed": seed, "delay": delay, "variant": variant,
                                    "r2": score, "shuffle_r2": null, "permutation_q95": q95,
                                    "precision_limited": info["precision_limited"], "gate_pass": bool(passed)})
                    for label, value in (("true", true), ("shuffle", shuffle), ("permutations", perm)):
                        predictions[f"{seed}_{key}_{label}"] = value
        print(json.dumps({"stage": "evaluated", "seed": seed}), flush=True)
    if any(sha(out/name) != value for name, value in fit_hashes.items()):
        raise RuntimeError("frozen fit changed")
    dump(out/"metrics.json", metrics)
    np.savez_compressed(out/"predictions.npz", **predictions)
    gates = {}
    for model in ("demian", "reservoir"):
        counts = {str(d): sum(c["gate_pass"] for c in metrics if c["model"] == model and
                              c["variant"] == protocol["primary_variant"] and c["delay"] == d)
                  for d in protocol["primary_delays"]}
        gates[model] = {"passing_seeds_by_delay": counts, "overall_pass": all(n >= 5 for n in counts.values())}
    dump(out/"summary.json", {"protocol_id": protocol["protocol_id"], "primary_variant": protocol["primary_variant"],
                              "gates": gates, "claim_boundary": protocol["claim_boundary"]})
    dump(out/"execution.json", {**execution, "status": "complete"})
    dump(out/"manifest.json", {"files": {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
         if p.is_file() and p.name != "manifest.json"}})
    print(json.dumps({"stage": "complete", "gates": gates}), flush=True)


if __name__ == "__main__":
    main()
