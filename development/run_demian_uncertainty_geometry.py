"""Label-free paired state-geometry decomposition for Demian uncertainty traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from development.run_demian_uncertainty_length_invariance import build_neutral_cue_corpus
from development.demian_uncertainty_probe import run_uncertainty_episode
from development.demian_v1_gate_state import V1_CHANNELS
from development.demian_uncertainty_world import AmbiguousEvidenceWorld


def _vector(part: object) -> np.ndarray:
    return part.detach().cpu().numpy().reshape(-1).astype(np.float64)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 1e-12 else 0.0


def _metrics(reference: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    reference_norm = float(np.linalg.norm(reference))
    changed_norm = float(np.linalg.norm(changed))
    return {
        "norm_ratio": changed_norm / reference_norm if reference_norm > 1e-12 else 0.0,
        "cosine": _cosine(reference, changed),
        "distance": float(np.linalg.norm(changed - reference)),
    }


def _pairwise_geometry(reference: np.ndarray, changed: np.ndarray) -> dict[str, float]:
    if len(reference) < 2:
        return {"pairwise_distance_correlation": 0.0, "pairwise_distance_ratio": 0.0}
    ref_distances = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    changed_distances = np.linalg.norm(changed[:, None, :] - changed[None, :, :], axis=2)
    upper = np.triu_indices(len(reference), k=1)
    left, right = ref_distances[upper], changed_distances[upper]
    correlation = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-12 and np.std(right) > 1e-12 else 0.0
    return {"pairwise_distance_correlation": correlation, "pairwise_distance_ratio": float(np.mean(right) / np.mean(left)) if float(np.mean(left)) > 1e-12 else 0.0}


def _states(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    corpus = [row for row in build_neutral_cue_corpus(protocol) if row["family"] == "heldout"]
    rows: list[dict[str, Any]] = []
    for seed in protocol["model_seeds"]:
        for spec in corpus:
            world = AmbiguousEvidenceWorld(hidden_target=int(spec["target"]), evidence=spec["evidence"], delay_steps=int(spec["delay_steps"]))
            frames = world.frames()
            count = len(spec["evidence"]) + int(spec["delay_steps"])
            trace = run_uncertainty_episode(model_seed=int(seed), hidden_size=int(protocol["hidden_size"]), public_frames=frames[:count], checkpoint_phases=("delay",))
            state = trace["checkpoints"][f"delay:{count - 1}"].state
            components = {name: _vector(value) for name, value in zip(V1_CHANNELS, state, strict=True)}
            rows.append({"model_seed": int(seed), "episode_id": int(spec["episode_id"]), "condition": spec["condition"], "suffix": spec["suffix"], "components": components, "full_state": np.concatenate(list(components.values()))})
    return rows


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    states = _states(protocol)
    pair_metrics: list[dict[str, Any]] = []
    channel_metrics: list[dict[str, Any]] = []
    for seed in protocol["model_seeds"]:
        seed_rows = [row for row in states if row["model_seed"] == int(seed)]
        base = {row["episode_id"]: row for row in seed_rows if row["condition"] == "base"}
        comparisons = [("extra_delay", "delay_plus_two", None)] + [("neutral_suffix", "neutral_cue_plus_two", suffix) for suffix in ("00", "01", "10", "11")]
        for comparison, condition, suffix in comparisons:
            matched = [row for row in seed_rows if row["condition"] == condition and row["suffix"] == suffix]
            ref = np.vstack([base[row["episode_id"]]["full_state"] for row in matched])
            changed = np.vstack([row["full_state"] for row in matched])
            pair_metrics.append({"model_seed": int(seed), "comparison": comparison, "suffix": suffix, **_pairwise_geometry(ref, changed)})
            for channel in ("full_state", *V1_CHANNELS):
                for reference_row, changed_row in zip((base[row["episode_id"]] for row in matched), matched, strict=True):
                    reference = reference_row["full_state"] if channel == "full_state" else reference_row["components"][channel]
                    changed_part = changed_row["full_state"] if channel == "full_state" else changed_row["components"][channel]
                    channel_metrics.append({"model_seed": int(seed), "episode_id": int(changed_row["episode_id"]), "comparison": comparison, "suffix": suffix, "channel": channel, **_metrics(reference, changed_part)})
    return {"pair_metrics": pair_metrics, "channel_metrics": channel_metrics, "integrity": {"all_finite": all(np.isfinite(value) for row in pair_metrics + channel_metrics for key, value in row.items() if isinstance(value, float)), "scope": "label-free paired geometry only"}}


def write_artifacts(out: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("pair_metrics.jsonl", payload["pair_metrics"]), ("channel_metrics.jsonl", payload["channel_metrics"])):
        with (out / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"experiment": protocol["protocol_id"], "pair_metric_count": len(payload["pair_metrics"]), "channel_metric_count": len(payload["channel_metrics"]), "integrity": payload["integrity"]}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)


if __name__ == "__main__":
    main()
