"""Hash-pinned, analysis-only standardized tie-break for relation screen v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from development.run_demian_final_slot_relation_predictor import _sha, tiebreak_rows


def run(source: Path, addendum: Path):
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("states_and_routes.npz", "metadata.jsonl", "metrics.jsonl", "summary.json"):
        if manifest["file_sha256"].get(name) != _sha(source / name):
            raise ValueError(f"source hash mismatch: {name}")
    protocol = manifest["protocol"]
    states = np.load(source / "states_and_routes.npz")
    rows = [json.loads(line) for line in (source / "metadata.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
    observed = {(row["metric"], row["model_seed"], row["arm"]): row["value"] for row in metrics if row["kind"] == "candidate"}
    records = tiebreak_rows(protocol, states["states"], states["routes"], rows)
    for record in records:
        for arm, value in (("final05_base", record["base_value"]), ("final05_upstream_swap12", record["upstream_value"])):
            if not np.isclose(value, observed[(record["metric"], record["model_seed"], arm)], rtol=1e-6, atol=1e-12):
                raise ValueError(f"metric replay mismatch: {record['metric']}/{record['model_seed']}/{arm}")
        record["contrast"] = record["base_value"] - record["upstream_value"]
        record["standardized_contrast"] = None if not np.isfinite(record["base_train_sd"]) or record["base_train_sd"] == 0 else record["contrast"] / record["base_train_sd"]
    eligible = set(json.loads((source / "summary.json").read_text())["selection"]["eligible_metrics_pending_tiebreak"])
    ranking = []
    for metric in sorted(eligible):
        values = [row["standardized_contrast"] for row in records if row["metric"] == metric and row["standardized_contrast"] is not None]
        ranking.append({"metric": metric, "valid_seed_count": len(values), "median_standardized_contrast": float(np.median(values)) if values else None})
    for row in ranking:
        row["median_standardized_magnitude"] = None if row["median_standardized_contrast"] is None else abs(row["median_standardized_contrast"])
    ranking.sort(key=lambda row: (-float("inf") if row["median_standardized_magnitude"] is None else -row["median_standardized_magnitude"], row["metric"]))
    winner = ranking[0]["metric"] if len(ranking) == 1 or (len(ranking) > 1 and ranking[0]["median_standardized_magnitude"] != ranking[1]["median_standardized_magnitude"]) else None
    return {"source_directory": str(source), "source_manifest_sha256": _sha(source / "manifest.json"), "addendum_sha256": _sha(addendum), "records": records, "ranking": ranking, "winner": winner,
            "claim_boundary": "analysis-only descriptive selection; no channel or route mechanism claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True); parser.add_argument("--addendum", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    result = run(args.source, args.addendum)
    (args.out / "tie_break.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"files":{"tie_break.json": _sha(args.out / "tie_break.json")}}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": result["winner"], "ranking": result["ranking"]}, sort_keys=True))


if __name__ == "__main__":
    main()
