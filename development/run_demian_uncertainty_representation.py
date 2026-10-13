"""Held-out frozen-probe evaluation for pre-resolution Demian states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from development.demian_uncertainty_probe import encode_evidence, run_uncertainty_episode
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, reference_belief, sample_evidence
from development.demian_v1_gate_state import V1_CHANNELS
from development.demian_uncertainty_representation import evaluate_ridge_probe, split_episode_rows


def _flatten(parts: tuple[object, ...]) -> np.ndarray:
    return np.concatenate([part.detach().cpu().numpy().reshape(-1) for part in parts]).astype(np.float64)


def resolve_episode_specs(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve an explicitly unique set of public histories before probe fitting."""
    schedules = [list(schedule) for schedule in protocol["reliability_schedules"]]
    if "episode_seeds" in protocol:
        return [
            {
                "episode_id": episode_id,
                "target": int(episode_seed) % 2,
                "evidence_seed": int(episode_seed),
                "schedule": schedules[episode_id % len(schedules)],
            }
            for episode_id, episode_seed in enumerate(protocol["episode_seeds"])
        ]
    episode_count = int(protocol["episode_count"])
    next_seed = int(protocol["episode_seed_start"])
    specs: list[dict[str, Any]] = []
    histories: set[tuple[tuple[int, float], ...]] = set()
    attempts = 0
    while len(specs) < episode_count:
        episode_id = len(specs)
        target = episode_id % 2
        schedule = schedules[(episode_id // 2) % len(schedules)]
        evidence = sample_evidence(hidden_target=target, reliabilities=schedule, seed=next_seed)
        next_seed += 1
        attempts += 1
        history = tuple(evidence)
        if history in histories:
            if attempts > episode_count * 10_000:
                raise ValueError("could not resolve enough unique public histories")
            continue
        histories.add(history)
        specs.append({
            "episode_id": episode_id,
            "target": target,
            "evidence_seed": next_seed - 1,
            "schedule": schedule,
            "public_history": history,
        })
    return specs


def collect_pre_resolution_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Capture final-delay internal state and evaluator-only public-evidence labels."""
    hidden_size = int(protocol["hidden_size"])
    rows: list[dict[str, Any]] = []
    schedules = [list(schedule) for schedule in protocol["reliability_schedules"]]
    if hidden_size < 10 or not schedules or any(not schedule for schedule in schedules):
        raise ValueError("positive schedules and encoder-compatible hidden size are required")
    for model_seed in protocol["model_seeds"]:
        for spec in resolve_episode_specs(protocol):
            episode_id = int(spec["episode_id"])
            target = int(spec["target"])
            schedule = list(spec["schedule"])
            evidence_seed = int(spec["evidence_seed"])
            evidence = sample_evidence(
                hidden_target=target,
                reliabilities=schedule,
                seed=evidence_seed,
            )
            for delay_steps in protocol["delay_steps"]:
                if int(delay_steps) <= 0:
                    raise ValueError("representation probe requires a positive delay")
                world = AmbiguousEvidenceWorld(
                    hidden_target=target,
                    evidence=evidence,
                    delay_steps=int(delay_steps),
                )
                frames = world.frames()
                trace = run_uncertainty_episode(
                    model_seed=int(model_seed),
                    hidden_size=hidden_size,
                    public_frames=frames,
                    checkpoint_phases=("delay",),
                )
                checkpoint = trace["checkpoints"][f"delay:{len(evidence) + int(delay_steps) - 1}"]
                components = dict(zip(V1_CHANNELS, checkpoint.state, strict=True))
                belief = reference_belief(frames[:len(evidence)])
                current = frames[len(evidence) + int(delay_steps) - 1]
                base = {
                    "model_seed": int(model_seed),
                    "episode_id": episode_id,
                    "episode_seed": evidence_seed,
                    "delay_steps": int(delay_steps),
                    "checkpoint_phase": "delay",
                    "features": {
                        "full_state": _flatten(checkpoint.state),
                        **{name: _flatten((value,)) for name, value in components.items()},
                        "current_observation": encode_evidence(current, hidden_size).detach().cpu().numpy().reshape(-1).astype(np.float64),
                        "latest_observation": encode_evidence(frames[len(evidence) - 1], hidden_size).detach().cpu().numpy().reshape(-1).astype(np.float64),
                    },
                }
                for label in ("log_odds", "entropy"):
                    rows.append({**base, "target": label, "target_value": float(belief[label])})
    return rows


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Evaluate fixed linear probes independently within each model realization."""
    rows = collect_pre_resolution_rows(protocol)
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_seed"], row["delay_steps"], row["target"])].append(row)
    scores: list[dict[str, Any]] = []
    for (model_seed, delay_steps, target), items in sorted(grouped.items()):
        train, heldout = split_episode_rows(items, heldout_fraction=float(protocol["heldout_fraction"]))
        y_train = np.asarray([row["target_value"] for row in train], dtype=np.float64)
        y_heldout = np.asarray([row["target_value"] for row in heldout], dtype=np.float64)
        feature_sets = sorted(train[0]["features"])
        for feature_set in feature_sets:
            result = evaluate_ridge_probe(
                np.vstack([row["features"][feature_set] for row in train]),
                y_train,
                np.vstack([row["features"][feature_set] for row in heldout]),
                y_heldout,
                alpha=float(protocol["ridge_alpha"]),
            )
            scores.append({
                "model_seed": model_seed,
                "delay_steps": delay_steps,
                "target": target,
                "feature_set": feature_set,
                "train_episode_count": len({row["episode_id"] for row in train}),
                "heldout_episode_count": len({row["episode_id"] for row in heldout}),
                **result,
            })
        shuffled = np.random.default_rng(model_seed + 10_000 * delay_steps).permutation(y_train)
        result = evaluate_ridge_probe(
            np.vstack([row["features"]["full_state"] for row in train]),
            shuffled,
            np.vstack([row["features"]["full_state"] for row in heldout]),
            y_heldout,
            alpha=float(protocol["ridge_alpha"]),
        )
        scores.append({
            "model_seed": model_seed,
            "delay_steps": delay_steps,
            "target": target,
            "feature_set": "full_state_shuffled_label",
            "train_episode_count": len({row["episode_id"] for row in train}),
            "heldout_episode_count": len({row["episode_id"] for row in heldout}),
            **result,
        })
    summary: dict[str, dict[str, float]] = {}
    for feature_set in sorted({row["feature_set"] for row in scores}):
        matching = [row for row in scores if row["feature_set"] == feature_set]
        summary[feature_set] = {
            "mean_r2": float(np.mean([row["r2"] for row in matching])),
            "seed_delay_cell_count": len(matching),
        }
    return {
        "rows": rows,
        "scores": scores,
        "summary": summary,
        "integrity": {
            "all_finite": all(math.isfinite(float(value)) for row in scores for value in (row["r2"], row["constant_r2"], row["mae"])),
            "probe_scope": "per-model-seed; held-out complete public episodes",
        },
    }


def _serializable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "features"} for row in rows]


def write_artifacts(out: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    protocol_encoded = json.dumps(protocol, sort_keys=True).encode()
    (out / "manifest.json").write_text(json.dumps({
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_encoded).hexdigest(),
        "row_count": len(payload["rows"]),
        "score_count": len(payload["scores"]),
    }, indent=2, sort_keys=True) + "\n")
    with (out / "scores.jsonl").open("x") as stream:
        for row in payload["scores"]:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({
        "schema_version": 1,
        "experiment": "demian-uncertainty-representation",
        "claim_boundary": "held-out frozen-probe measurement only; no causal, routing, utility, learning, or adaptation claim",
        "row_count": len(payload["rows"]),
        "score_count": len(payload["scores"]),
        "summary": payload["summary"],
        "integrity": payload["integrity"],
    }, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), "scores": len(payload["scores"]), "integrity": payload["integrity"]}))


if __name__ == "__main__":
    main()
