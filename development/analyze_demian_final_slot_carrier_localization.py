"""Hash-pinned localization test: carrier endpoint residual versus global endpoint error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from development.run_demian_final_slot_relation_predictor import _cell, _sha, _spearman
from development.demian_v1_gate_state import V1_CHANNELS
from development.run_demian_uncertainty_decoder_tracking import apply_tracker, fit_tracker
from development.run_demian_uncertainty_localization import fit_probe, predict_probe, split_histories


def run(source: Path):
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("states_and_routes.npz", "metadata.jsonl", "summary.json"):
        if manifest["file_sha256"].get(name) != _sha(source / name):
            raise ValueError(f"source hash mismatch: {name}")
    protocol = manifest["protocol"]
    arrays = np.load(source / "states_and_routes.npz")
    rows = [json.loads(line) for line in (source / "metadata.jsonl").read_text().splitlines()]
    _, _, reference = _cell(arrays["states"], arrays["routes"], rows, int(protocol["model_seeds"][0]), "final05_base")
    split = split_histories([{**row, "symbols": row["canonical_symbols"]} for row in reference], int(protocol["history_split_seed"]))
    train_ids, eval_ids = set(split["train_history_indices"]), set(split["evaluation_history_indices"])
    width = int(protocol["hidden_size"])
    offsets = {name: slice(i*width, (i+1)*width) for i, name in enumerate(V1_CHANNELS)}
    seed_rows = []
    summary = json.loads((source / "summary.json").read_text())
    for seed in protocol["model_seeds"]:
        base, _, base_meta = _cell(arrays["states"], arrays["routes"], rows, int(seed), "final05_base")
        upstream, _, upstream_meta = _cell(arrays["states"], arrays["routes"], rows, int(seed), "final05_upstream_swap12")
        train = np.asarray([i for i, row in enumerate(base_meta) if row["paired_history_index"] in train_ids])
        evaluate = np.asarray([i for i, row in enumerate(base_meta) if row["paired_history_index"] in eval_ids])
        upstream_evaluate = np.asarray([i for i, row in enumerate(upstream_meta) if row["paired_history_index"] in eval_ids])
        errors, means, scales = {}, {}, {}
        for channel in V1_CHANNELS:
            s = offsets[channel]; tracker = fit_tracker(base[train, 0, s], base[train, -1, s], mode="orthogonal_track")
            def error(data, indexes):
                return np.mean((apply_tracker(tracker, data[indexes, -1, s]) - data[indexes, 0, s]) ** 2, axis=1)
            train_error = error(base, train); means[channel], scales[channel] = float(train_error.mean()), float(train_error.std())
            if not np.isfinite(scales[channel]) or scales[channel] <= 0:
                raise ValueError(f"zero/nonfinite base train scale: {seed}/{channel}")
            errors[channel] = {"base": error(base, evaluate), "upstream": error(upstream, upstream_evaluate), "train": train_error}
        g_train = np.mean([(errors[channel]["train"] - means[channel]) / scales[channel] for channel in V1_CHANNELS if channel != "carrier"], axis=0)
        nuisance = fit_probe(g_train[:, None], errors["carrier"]["train"], alpha=1.0)
        def residual(arm):
            g = np.mean([(errors[channel][arm] - means[channel]) / scales[channel] for channel in V1_CHANNELS if channel != "carrier"], axis=0)
            return errors["carrier"][arm] - predict_probe(nuisance, g[:, None])
        contrasts = {channel: float((errors[channel]["base"].mean() - errors[channel]["upstream"].mean()) / scales[channel]) for channel in V1_CHANNELS}
        global_base = np.mean([(errors[channel]["base"] - means[channel]) / scales[channel] for channel in V1_CHANNELS if channel != "carrier"], axis=0)
        global_upstream = np.mean([(errors[channel]["upstream"] - means[channel]) / scales[channel] for channel in V1_CHANNELS if channel != "carrier"], axis=0)
        global_contrast = float(global_base.mean() - global_upstream.mean())
        residual_contrast = float(residual("base").mean() - residual("upstream").mean())
        dominance = abs(contrasts["carrier"]) - max(abs(contrasts[channel]) for channel in V1_CHANNELS if channel != "carrier")
        outcome = float(summary["selection"]["outcomes"][str(seed)])
        seed_rows.append({"model_seed": int(seed), "outcome": outcome, "standardized_contrasts": contrasts,
                          "residualized_carrier_contrast": residual_contrast, "carrier_dominance": float(dominance), "global_composite_contrast": global_contrast,
                          "nuisance_fit": {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in nuisance.items()},
                          "base_train_means": means, "base_train_scales": scales})
    residuals, outcomes = np.asarray([row["residualized_carrier_contrast"] for row in seed_rows]), np.asarray([row["outcome"] for row in seed_rows])
    loo = [_spearman(np.delete(residuals, i), np.delete(outcomes, i)) for i in range(len(seed_rows))]
    gate = {"residual_negative": int((residuals < 0).sum()), "dominance_positive": int(sum(row["carrier_dominance"] > 0 for row in seed_rows)),
            "loo_negative": int(sum(value is not None and value < 0 for value in loo)), "passes": int((residuals < 0).sum()) >= 5 and int(sum(row["carrier_dominance"] > 0 for row in seed_rows)) >= 5 and int(sum(value is not None and value < 0 for value in loo)) >= 5}
    global_contrasts = np.asarray([row["global_composite_contrast"] for row in seed_rows])
    global_loo = [_spearman(np.delete(global_contrasts, i), np.delete(outcomes, i)) for i in range(len(seed_rows))]
    global_gate = {"contrast_negative": int((global_contrasts < 0).sum()), "loo_negative": int(sum(value is not None and value < 0 for value in global_loo)), "passes": int((global_contrasts < 0).sum()) >= 5 and int(sum(value is not None and value < 0 for value in global_loo)) >= 5}
    return {"source_manifest_sha256": _sha(source / "manifest.json"), "seed_rows": seed_rows, "leave_one_out_spearman": loo, "gate": gate, "global_leave_one_out_spearman": global_loo, "global_gate": global_gate,
            "claim_boundary": "localization discriminator; no causal inference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False); result = run(args.source)
    (args.out / "localization.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"files":{"localization.json": _sha(args.out / "localization.json")}}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["gate"], sort_keys=True))


if __name__ == "__main__":
    main()
