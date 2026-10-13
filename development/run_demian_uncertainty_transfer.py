"""Frozen low-capacity transfer probe for delayed public evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from development.demian_uncertainty_probe import encode_evidence, run_uncertainty_episode
from development.demian_uncertainty_representation import evaluate_ridge_probe, fixed_random_projection
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, reference_belief


def _schedule_rows(
    schedules: list[list[float]], *, family: str, allocation_seed: int, episode_offset: int,
    history_count: int | None = None,
) -> list[dict[str, Any]]:
    if not schedules:
        raise ValueError(f"{family} requires at least one reliability schedule")
    rows: list[dict[str, Any]] = []
    generator = random.Random(allocation_seed)
    for schedule_index, raw_schedule in enumerate(schedules):
        schedule = tuple(float(value) for value in raw_schedule)
        histories = list(itertools.product((0, 1), repeat=len(schedule)))
        if history_count is not None:
            if not 0 < history_count <= len(histories) or history_count % 2:
                raise ValueError("history_count must be positive, even, and within the schedule support")
            generator.shuffle(histories)
            histories = histories[:history_count]
        targets = [0] * (len(histories) // 2) + [1] * (len(histories) // 2)
        generator.shuffle(targets)
        for history_index, (symbols, target) in enumerate(zip(histories, targets, strict=True)):
            rows.append({
                "episode_id": episode_offset + len(rows),
                "family": family,
                "schedule_index": schedule_index,
                "target": target,
                "history_index": history_index,
                "history_length": len(schedule),
                "reliability_schedule": schedule,
                "evidence": tuple(zip(symbols, schedule, strict=True)),
            })
    return rows


def build_transfer_corpus(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Build independently allocated train, unseen-order, and unseen-length corpora."""
    allocation_seed = int(protocol["allocation_seed"])
    train_schedules = protocol["train_reliability_schedules"]
    unseen_order_schedules = protocol["unseen_order_schedules"]
    unseen_length_schedules = protocol["unseen_length_schedules"]
    train_length = len(train_schedules[0])
    if any(len(schedule) != train_length for schedule in train_schedules + unseen_order_schedules):
        raise ValueError("train and unseen-order schedules must share one history length")
    train_multiset = tuple(sorted(float(value) for value in train_schedules[0]))
    if any(tuple(sorted(float(value) for value in schedule)) != train_multiset for schedule in train_schedules + unseen_order_schedules):
        raise ValueError("train and unseen-order schedules must share one reliability multiset")
    train_orders = {tuple(float(value) for value in schedule) for schedule in train_schedules}
    unseen_orders = {tuple(float(value) for value in schedule) for schedule in unseen_order_schedules}
    if train_orders & unseen_orders:
        raise ValueError("unseen-order schedules must be disjoint from train schedules")
    train = _schedule_rows(train_schedules, family="train", allocation_seed=allocation_seed, episode_offset=0)
    unseen_order = _schedule_rows(
        unseen_order_schedules, family="unseen_order", allocation_seed=allocation_seed + 1, episode_offset=len(train),
    )
    unseen_length = _schedule_rows(
        unseen_length_schedules,
        family="unseen_length",
        allocation_seed=allocation_seed + 2,
        episode_offset=len(train) + len(unseen_order),
        history_count=int(protocol["unseen_length_history_count"]),
    )
    return train + unseen_order + unseen_length


def _flatten(parts: tuple[object, ...]) -> np.ndarray:
    return np.concatenate([part.detach().cpu().numpy().reshape(-1) for part in parts]).astype(np.float64)


def _schedule_only(spec: dict[str, Any], *, maximum_history_length: int) -> np.ndarray:
    padded = list(spec["reliability_schedule"]) + [0.0] * (maximum_history_length - int(spec["history_length"]))
    return np.asarray([float(spec["history_length"]), *padded], dtype=np.float64)


def collect_transfer_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect post-delay states without exposing hidden target or resolution frames."""
    hidden_size = int(protocol["hidden_size"])
    corpus = build_transfer_corpus(protocol)
    maximum_history_length = max(int(spec["history_length"]) for spec in corpus)
    rows: list[dict[str, Any]] = []
    for model_seed in protocol["model_seeds"]:
        for spec in corpus:
            for delay_steps in protocol["delay_steps"]:
                world = AmbiguousEvidenceWorld(hidden_target=int(spec["target"]), evidence=spec["evidence"], delay_steps=int(delay_steps))
                frames = world.frames()
                pre_resolution_count = len(spec["evidence"]) + int(delay_steps)
                trace = run_uncertainty_episode(
                    model_seed=int(model_seed), hidden_size=hidden_size, public_frames=frames[:pre_resolution_count], checkpoint_phases=("delay",),
                )
                checkpoint = trace["checkpoints"][f"delay:{pre_resolution_count - 1}"]
                current = frames[pre_resolution_count - 1]
                latest = frames[len(spec["evidence"]) - 1]
                rows.append({
                    **spec,
                    "model_seed": int(model_seed),
                    "delay_steps": int(delay_steps),
                    "target_value": float(reference_belief(frames[:len(spec["evidence"])])["log_odds"]),
                    "features": {
                        "full_state": _flatten(checkpoint.state),
                        "current_observation": encode_evidence(current, hidden_size).detach().cpu().numpy().reshape(-1).astype(np.float64),
                        "latest_observation": encode_evidence(latest, hidden_size).detach().cpu().numpy().reshape(-1).astype(np.float64),
                        "schedule_only": _schedule_only(spec, maximum_history_length=maximum_history_length),
                    },
                })
    return rows


def _conditional_shuffle(rows: list[dict[str, Any]], *, seed: int) -> np.ndarray:
    """Permute labels inside declared public-marginal groups only."""
    grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        evidence = row["evidence"]
        grouped[(
            int(row["history_length"]),
            tuple(sorted(float(reliability) for _, reliability in evidence)),
            int(evidence[-1][0]),
            float(evidence[-1][1]),
        )].append(index)
    shuffled = np.asarray([float(row["target_value"]) for row in rows], dtype=np.float64)
    generator = np.random.default_rng(seed)
    for indexes in grouped.values():
        shuffled[indexes] = generator.permutation(shuffled[indexes])
    return shuffled


def _score(
    train: list[dict[str, Any]], evaluate: list[dict[str, Any]], *, feature_set: str, alpha: float,
    projection_dimension: int | None = None, projection_seed: int | None = None, shuffled_labels: np.ndarray | None = None,
) -> dict[str, float]:
    train_features = np.vstack([row["features"][feature_set] for row in train])
    evaluate_features = np.vstack([row["features"][feature_set] for row in evaluate])
    if projection_dimension is not None and projection_seed is not None:
        train_features = fixed_random_projection(train_features, output_dim=projection_dimension, seed=projection_seed)
        evaluate_features = fixed_random_projection(evaluate_features, output_dim=projection_dimension, seed=projection_seed)
    train_labels = shuffled_labels if shuffled_labels is not None else np.asarray([row["target_value"] for row in train], dtype=np.float64)
    evaluate_labels = np.asarray([row["target_value"] for row in evaluate], dtype=np.float64)
    return evaluate_ridge_probe(train_features, train_labels, evaluate_features, evaluate_labels, alpha=alpha)


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = collect_transfer_rows(protocol)
    scores: list[dict[str, Any]] = []
    for model_seed in protocol["model_seeds"]:
        for delay_steps in protocol["delay_steps"]:
            cell = [row for row in rows if row["model_seed"] == int(model_seed) and row["delay_steps"] == int(delay_steps)]
            train = [row for row in cell if row["family"] == "train"]
            evaluations = {family: [row for row in cell if row["family"] == family] for family in ("unseen_order", "unseen_length")}
            shuffled = _conditional_shuffle(train, seed=int(model_seed) + int(delay_steps) * 10_000)
            for family, evaluate in evaluations.items():
                for dimension in protocol["projection_dimensions"]:
                    for projection_seed in protocol["projection_seeds"]:
                        scores.append({
                            "model_seed": int(model_seed), "delay_steps": int(delay_steps), "evaluation_family": family,
                            "feature_set": "projected_full_state", "projection_dimension": int(dimension), "projection_seed": int(projection_seed),
                            **_score(train, evaluate, feature_set="full_state", alpha=float(protocol["ridge_alpha"]), projection_dimension=int(dimension), projection_seed=int(projection_seed)),
                        })
                        scores.append({
                            "model_seed": int(model_seed), "delay_steps": int(delay_steps), "evaluation_family": family,
                            "feature_set": "full_state_conditional_shuffle", "projection_dimension": int(dimension), "projection_seed": int(projection_seed),
                            **_score(train, evaluate, feature_set="full_state", alpha=float(protocol["ridge_alpha"]), projection_dimension=int(dimension), projection_seed=int(projection_seed), shuffled_labels=shuffled),
                        })
                for feature_set in ("current_observation", "latest_observation", "schedule_only"):
                    scores.append({
                        "model_seed": int(model_seed), "delay_steps": int(delay_steps), "evaluation_family": family,
                        "feature_set": feature_set, "projection_dimension": None, "projection_seed": None,
                        **_score(train, evaluate, feature_set=feature_set, alpha=float(protocol["ridge_alpha"])),
                    })
    summary = {
        f"{family}/{feature_set}": {"mean_r2": float(np.mean([
            row["r2"] for row in scores
            if (row["evaluation_family"], row["feature_set"]) == (family, feature_set)
        ]))}
        for family, feature_set in sorted({(row["evaluation_family"], row["feature_set"]) for row in scores})
    }
    return {
        "rows": rows, "scores": scores, "summary": summary,
        "integrity": {"all_finite": all(np.isfinite(row[metric]) for row in scores for metric in ("r2", "mae", "pearson_r", "normalized_mae")), "probe_scope": "train-only frozen coefficients, projection, and standardization per model seed and delay"},
    }


def write_artifacts(out: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(protocol, sort_keys=True).encode()
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(encoded).hexdigest(), "row_count": len(payload["rows"]), "score_count": len(payload["scores"])}, indent=2, sort_keys=True) + "\n")
    with (out / "scores.jsonl").open("x") as stream:
        for row in payload["scores"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"schema_version": 1, "experiment": "demian-uncertainty-transfer", "claim_boundary": "low-capacity held-out transfer probe only; no causal, routing, utility, learning, or adaptation claim without review", "row_count": len(payload["rows"]), "score_count": len(payload["scores"]), "summary": payload["summary"], "integrity": payload["integrity"]}, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
