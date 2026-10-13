"""Paired neutral-cue length-invariance gate for delayed public evidence."""

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
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, EvidenceObservation, reference_belief


def _base_specs(schedules: list[list[float]], *, family: str, allocation_seed: int, offset: int) -> list[dict[str, Any]]:
    generator = random.Random(allocation_seed)
    specs: list[dict[str, Any]] = []
    for schedule_index, raw_schedule in enumerate(schedules):
        schedule = tuple(float(value) for value in raw_schedule)
        histories = list(itertools.product((0, 1), repeat=len(schedule)))
        targets = [0] * (len(histories) // 2) + [1] * (len(histories) // 2)
        generator.shuffle(targets)
        for history_index, (symbols, target) in enumerate(zip(histories, targets, strict=True)):
            observations = [EvidenceObservation(index, "evidence", symbol, reliability) for index, (symbol, reliability) in enumerate(zip(symbols, schedule, strict=True))]
            specs.append({"episode_id": offset + len(specs), "family": family, "schedule_index": schedule_index, "target": target, "history_index": history_index, "base_evidence": tuple(zip(symbols, schedule, strict=True)), "target_value": float(reference_belief(observations)["log_odds"])})
    return specs


def build_neutral_cue_corpus(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Create paired base, extra-delay, and all neutral-suffix variants."""
    train = _base_specs(protocol["train_reliability_schedules"], family="train", allocation_seed=int(protocol["allocation_seed"]), offset=0)
    heldout = _base_specs(protocol["heldout_reliability_schedules"], family="heldout", allocation_seed=int(protocol["allocation_seed"]) + 1, offset=len(train))
    base_schedules = protocol["train_reliability_schedules"] + protocol["heldout_reliability_schedules"]
    if len({len(schedule) for schedule in base_schedules}) != 1:
        raise ValueError("all base schedules require one shared length")
    if len({tuple(sorted(float(value) for value in schedule)) for schedule in base_schedules}) != 1:
        raise ValueError("train and heldout schedules require one shared reliability multiset")
    if {tuple(schedule) for schedule in protocol["train_reliability_schedules"]} & {tuple(schedule) for schedule in protocol["heldout_reliability_schedules"]}:
        raise ValueError("heldout schedules must be disjoint from train schedules")
    rows: list[dict[str, Any]] = []
    for spec in train + heldout:
        base = spec["base_evidence"]
        rows.append({**spec, "condition": "base", "suffix": None, "evidence": base, "delay_steps": int(protocol["base_delay_steps"])})
        if spec["family"] == "train":
            continue
        rows.append({**spec, "condition": "delay_plus_two", "suffix": None, "evidence": base, "delay_steps": int(protocol["base_delay_steps"]) + 2})
        for suffix in itertools.product((0, 1), repeat=2):
            rows.append({**spec, "condition": "neutral_cue_plus_two", "suffix": "".join(map(str, suffix)), "evidence": base + tuple((symbol, 0.5) for symbol in suffix), "delay_steps": int(protocol["base_delay_steps"])})
    return rows


def _flatten(parts: tuple[object, ...]) -> np.ndarray:
    return np.concatenate([part.detach().cpu().numpy().reshape(-1) for part in parts]).astype(np.float64)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector.copy()


def _schedule_only(row: dict[str, Any], *, width: int) -> np.ndarray:
    schedule = [reliability for _, reliability in row["evidence"]]
    return np.asarray([len(row["evidence"]), *schedule, *([0.0] * (width - len(schedule)))], dtype=np.float64)


def collect_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    corpus = build_neutral_cue_corpus(protocol)
    width = max(len(row["evidence"]) for row in corpus)
    rows: list[dict[str, Any]] = []
    for model_seed in protocol["model_seeds"]:
        for spec in corpus:
            world = AmbiguousEvidenceWorld(hidden_target=int(spec["target"]), evidence=spec["evidence"], delay_steps=int(spec["delay_steps"]))
            frames = world.frames()
            count = len(spec["evidence"]) + int(spec["delay_steps"])
            trace = run_uncertainty_episode(model_seed=int(model_seed), hidden_size=int(protocol["hidden_size"]), public_frames=frames[:count], checkpoint_phases=("delay",))
            checkpoint = trace["checkpoints"][f"delay:{count - 1}"]
            informative = frames[len(spec["base_evidence"]) - 1]
            state = _flatten(checkpoint.state)
            rows.append({**spec, "model_seed": int(model_seed), "features": {
                "full_state": state,
                "full_state_l2_normalized": _l2_normalize(state),
                "current_observation": encode_evidence(frames[count - 1], int(protocol["hidden_size"])).detach().cpu().numpy().reshape(-1).astype(np.float64),
                "latest_informative_observation": encode_evidence(informative, int(protocol["hidden_size"])).detach().cpu().numpy().reshape(-1).astype(np.float64),
                "latest_observation": encode_evidence(frames[len(spec["evidence"]) - 1], int(protocol["hidden_size"])).detach().cpu().numpy().reshape(-1).astype(np.float64),
                "schedule_only": _schedule_only(spec, width=width),
            }})
    return rows


def _split_base(rows: list[dict[str, Any]], *, split_seed: int, heldout_fraction: float) -> tuple[set[int], set[int]]:
    base = [row for row in rows if row["condition"] == "base" and row["family"] == "train"]
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in base:
        grouped[(int(row["schedule_index"]), int(row["target"]))].append(int(row["episode_id"]))
    generator = random.Random(split_seed)
    test: set[int] = set()
    for ids in grouped.values():
        generator.shuffle(ids)
        test.update(ids[:max(1, round(len(ids) * heldout_fraction))])
    return {int(row["episode_id"]) for row in base} - test, test


def _conditional_shuffle(rows: list[dict[str, Any]], *, seed: int) -> np.ndarray:
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        last_symbol, last_reliability = row["base_evidence"][-1]
        groups[(tuple(sorted(reliability for _, reliability in row["base_evidence"])), last_symbol, last_reliability)].append(index)
    labels = np.asarray([row["target_value"] for row in rows], dtype=np.float64)
    generator = np.random.default_rng(seed)
    for indexes in groups.values():
        labels[indexes] = generator.permutation(labels[indexes])
    return labels


def _score(train: list[dict[str, Any]], evaluate: list[dict[str, Any]], *, feature: str, alpha: float, dimension: int | None = None, projection_seed: int | None = None, shuffle: np.ndarray | None = None, condition_normalized: bool = False) -> dict[str, float]:
    x_train = np.vstack([row["features"][feature] for row in train])
    x_eval = np.vstack([row["features"][feature] for row in evaluate])
    if dimension is not None:
        x_train = fixed_random_projection(x_train, output_dim=dimension, seed=int(projection_seed))
        x_eval = fixed_random_projection(x_eval, output_dim=dimension, seed=int(projection_seed))
    if condition_normalized:
        x_train = (x_train - x_train.mean(axis=0)) / np.where(x_train.std(axis=0) > 1e-12, x_train.std(axis=0), 1.0)
        x_eval = (x_eval - x_eval.mean(axis=0)) / np.where(x_eval.std(axis=0) > 1e-12, x_eval.std(axis=0), 1.0)
    y_train = shuffle if shuffle is not None else np.asarray([row["target_value"] for row in train], dtype=np.float64)
    return evaluate_ridge_probe(x_train, y_train, x_eval, np.asarray([row["target_value"] for row in evaluate], dtype=np.float64), alpha=alpha)


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = collect_rows(protocol)
    scores: list[dict[str, Any]] = []
    for model_seed in protocol["model_seeds"]:
        seed_rows = [row for row in rows if row["model_seed"] == int(model_seed)]
        train = [row for row in seed_rows if row["family"] == "train" and row["condition"] == "base"]
        shuffle = _conditional_shuffle(train, seed=int(model_seed))
        for condition in ("base", "delay_plus_two", "neutral_cue_plus_two"):
            variants = [None] if condition != "neutral_cue_plus_two" else ["00", "01", "10", "11"]
            for suffix in variants:
                evaluate = [row for row in seed_rows if row["family"] == "heldout" and row["condition"] == condition and row["suffix"] == suffix]
                for dimension in protocol["projection_dimensions"]:
                    for projection_seed in protocol["projection_seeds"]:
                        common = {"model_seed": int(model_seed), "evaluation_condition": condition, "suffix": suffix, "projection_dimension": int(dimension), "projection_seed": int(projection_seed)}
                        scores.append({**common, "feature_set": "projected_full_state", **_score(train, evaluate, feature="full_state", alpha=float(protocol["ridge_alpha"]), dimension=int(dimension), projection_seed=int(projection_seed))})
                        scores.append({**common, "feature_set": "projected_full_state_l2_normalized", **_score(train, evaluate, feature="full_state_l2_normalized", alpha=float(protocol["ridge_alpha"]), dimension=int(dimension), projection_seed=int(projection_seed))})
                        scores.append({**common, "feature_set": "projected_full_state_condition_normalized", **_score(train, evaluate, feature="full_state", alpha=float(protocol["ridge_alpha"]), dimension=int(dimension), projection_seed=int(projection_seed), condition_normalized=True)})
                        scores.append({**common, "feature_set": "full_state_conditional_shuffle", **_score(train, evaluate, feature="full_state", alpha=float(protocol["ridge_alpha"]), dimension=int(dimension), projection_seed=int(projection_seed), shuffle=shuffle)})
                for feature in ("current_observation", "latest_informative_observation", "latest_observation", "schedule_only"):
                    scores.append({"model_seed": int(model_seed), "evaluation_condition": condition, "suffix": suffix, "projection_dimension": None, "projection_seed": None, "feature_set": feature, **_score(train, evaluate, feature=feature, alpha=float(protocol["ridge_alpha"]))})
    summary = {f"{condition}/{feature}": {"mean_r2": float(np.mean([row["r2"] for row in scores if row["evaluation_condition"] == condition and row["feature_set"] == feature]))} for condition, feature in sorted({(row["evaluation_condition"], row["feature_set"]) for row in scores})}
    return {"rows": rows, "scores": scores, "summary": summary, "integrity": {"all_finite": all(np.isfinite(row[metric]) for row in scores for metric in ("r2", "mae", "pearson_r", "normalized_mae")), "probe_scope": "heldout reliability order; train-only frozen projection and coefficients; neutral suffixes dependent within base history"}}


def write_artifacts(out: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(protocol, sort_keys=True).encode()
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(encoded).hexdigest(), "row_count": len(payload["rows"]), "score_count": len(payload["scores"])}, indent=2, sort_keys=True) + "\n")
    with (out / "scores.jsonl").open("x") as stream:
        for row in payload["scores"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"schema_version": 1, "experiment": "demian-uncertainty-length-invariance", "claim_boundary": "paired frozen-probe calibration measurement only; no causal, routing, utility, learning, or adaptation claim without review", "row_count": len(payload["rows"]), "score_count": len(payload["scores"]), "summary": payload["summary"], "integrity": payload["integrity"]}, indent=2, sort_keys=True, allow_nan=False) + "\n")


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
