"""Schedule-balanced corrective probe for delayed public-evidence retention."""

from __future__ import annotations

import itertools
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from development.demian_uncertainty_probe import encode_evidence, run_uncertainty_episode
from development.demian_uncertainty_representation import evaluate_ridge_probe
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, reference_belief
from development.demian_v1_gate_state import V1_CHANNELS


def build_balanced_corpus(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every public symbol history under every reliability order."""
    schedules = [tuple(float(value) for value in schedule) for schedule in protocol["reliability_schedules"]]
    if not schedules or len({len(schedule) for schedule in schedules}) != 1:
        raise ValueError("nonempty reliability schedules with equal length are required")
    if len({tuple(sorted(schedule)) for schedule in schedules}) != 1:
        raise ValueError("every schedule must share one reliability multiset")
    histories = list(itertools.product((0, 1), repeat=len(schedules[0])))
    generator = random.Random(int(protocol["allocation_seed"]))
    corpus: list[dict[str, Any]] = []
    for schedule_id, schedule in enumerate(schedules):
        targets = [0] * (len(histories) // 2) + [1] * (len(histories) // 2)
        generator.shuffle(targets)
        for history_index, (symbols, target) in enumerate(zip(histories, targets, strict=True)):
            corpus.append({
                "episode_id": len(corpus),
                "schedule_id": schedule_id,
                "target": target,
                "history_index": history_index,
                "evidence": tuple(zip(symbols, schedule, strict=True)),
            })
    return corpus


def stratified_episode_split(
    rows: list[dict[str, Any]], *, split_seed: int, heldout_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole episodes inside each schedule-by-target allocation cell."""
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must lie strictly between zero and one")
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["schedule_id"]), int(row["target"]))].append(int(row["episode_id"]))
    generator = random.Random(split_seed)
    heldout_ids: set[int] = set()
    for episode_ids in grouped.values():
        if len(episode_ids) < 2:
            raise ValueError("each schedule-target cell needs at least two episodes")
        ordered = sorted(episode_ids)
        generator.shuffle(ordered)
        heldout_ids.update(ordered[:max(1, round(len(ordered) * heldout_fraction))])
    return (
        [dict(row) for row in rows if int(row["episode_id"]) not in heldout_ids],
        [dict(row) for row in rows if int(row["episode_id"]) in heldout_ids],
    )


def _flatten(parts: tuple[object, ...]) -> np.ndarray:
    return np.concatenate([part.detach().cpu().numpy().reshape(-1) for part in parts]).astype(np.float64)


def collect_pre_resolution_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Run only through delay: hidden target never enters Demian's input path."""
    hidden_size = int(protocol["hidden_size"])
    schedules = protocol["reliability_schedules"]
    corpus = build_balanced_corpus(protocol)
    rows: list[dict[str, Any]] = []
    for model_seed in protocol["model_seeds"]:
        for spec in corpus:
            for delay_steps in protocol["delay_steps"]:
                world = AmbiguousEvidenceWorld(
                    hidden_target=int(spec["target"]),
                    evidence=spec["evidence"],
                    delay_steps=int(delay_steps),
                )
                frames = world.frames()
                pre_resolution_count = len(spec["evidence"]) + int(delay_steps)
                trace = run_uncertainty_episode(
                    model_seed=int(model_seed),
                    hidden_size=hidden_size,
                    public_frames=frames[:pre_resolution_count],
                    checkpoint_phases=("delay",),
                )
                checkpoint = trace["checkpoints"][f"delay:{pre_resolution_count - 1}"]
                components = dict(zip(V1_CHANNELS, checkpoint.state, strict=True))
                current = frames[pre_resolution_count - 1]
                schedule_vector = np.zeros(len(schedules), dtype=np.float64)
                schedule_vector[int(spec["schedule_id"])] = 1.0
                rows.append({
                    "model_seed": int(model_seed),
                    "episode_id": int(spec["episode_id"]),
                    "schedule_id": int(spec["schedule_id"]),
                    "target": int(spec["target"]),
                    "delay_steps": int(delay_steps),
                    "target_value": float(reference_belief(frames[:len(spec["evidence"])])["log_odds"]),
                    "features": {
                        "full_state": _flatten(checkpoint.state),
                        **{name: _flatten((value,)) for name, value in components.items()},
                        "current_observation": encode_evidence(current, hidden_size).detach().cpu().numpy().reshape(-1).astype(np.float64),
                        "latest_observation": encode_evidence(frames[len(spec["evidence"]) - 1], hidden_size).detach().cpu().numpy().reshape(-1).astype(np.float64),
                        "schedule_only": schedule_vector,
                    },
                })
    return rows


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = collect_pre_resolution_rows(protocol)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_seed"], row["delay_steps"])].append(row)
    scores: list[dict[str, Any]] = []
    for (model_seed, delay_steps), items in sorted(grouped.items()):
        for split_seed in protocol["split_seeds"]:
            train, heldout = stratified_episode_split(
                items,
                split_seed=int(split_seed),
                heldout_fraction=float(protocol["heldout_fraction"]),
            )
            train_y = np.asarray([row["target_value"] for row in train], dtype=np.float64)
            heldout_y = np.asarray([row["target_value"] for row in heldout], dtype=np.float64)
            for feature_set in sorted(train[0]["features"]):
                result = evaluate_ridge_probe(
                    np.vstack([row["features"][feature_set] for row in train]), train_y,
                    np.vstack([row["features"][feature_set] for row in heldout]), heldout_y,
                    alpha=float(protocol["ridge_alpha"]),
                )
                scores.append({"model_seed": model_seed, "delay_steps": delay_steps, "split_seed": int(split_seed), "feature_set": feature_set, **result})
            shuffled = np.random.default_rng(model_seed + delay_steps * 10_000 + int(split_seed)).permutation(train_y)
            result = evaluate_ridge_probe(
                np.vstack([row["features"]["full_state"] for row in train]), shuffled,
                np.vstack([row["features"]["full_state"] for row in heldout]), heldout_y,
                alpha=float(protocol["ridge_alpha"]),
            )
            scores.append({"model_seed": model_seed, "delay_steps": delay_steps, "split_seed": int(split_seed), "feature_set": "full_state_shuffled_label", **result})
    summary = {
        feature_set: {"mean_r2": float(np.mean([row["r2"] for row in scores if row["feature_set"] == feature_set]))}
        for feature_set in sorted({row["feature_set"] for row in scores})
    }
    return {"rows": rows, "scores": scores, "summary": summary, "integrity": {"all_finite": all(np.isfinite(row["r2"]) for row in scores), "probe_scope": "per-model-seed, schedule-target-stratified complete episodes"}}


def write_artifacts(out: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(protocol, sort_keys=True).encode()
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(encoded).hexdigest(), "row_count": len(payload["rows"]), "score_count": len(payload["scores"])}, indent=2, sort_keys=True) + "\n")
    with (out / "scores.jsonl").open("x") as stream:
        for row in payload["scores"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"schema_version": 1, "experiment": "demian-uncertainty-corrective", "claim_boundary": "corrective held-out probe only; no retained-evidence, uncertainty, causal, routing, utility, learning, or adaptation claim without review", "row_count": len(payload["rows"]), "score_count": len(payload["scores"]), "summary": payload["summary"], "integrity": payload["integrity"]}, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), "scores": len(payload["scores"])}))


if __name__ == "__main__":
    main()
