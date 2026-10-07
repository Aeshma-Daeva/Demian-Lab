#!/usr/bin/env python3
"""Run a small untrained closed-loop comparator pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from development.closed_loop_comparators import build_budget_tracks, run_comparator_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument("--reference-hidden-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--delay-steps", type=int, default=31)
    parser.add_argument("--track", choices=("state", "parameters"), default="state")
    parser.add_argument("--environment-exposure", choices=("natural", "forced_read"), default="natural")
    parser.add_argument("--out", type=Path, default=Path("data/diagnostics/closed_loop_comparator_pilot/summary.json"))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    tracks = build_budget_tracks(reference_hidden_size=args.reference_hidden_size, seed=seeds[0])
    selected_track = tracks[args.track]
    specifications = selected_track["specifications"]
    runs = []
    for architecture, specification in specifications.items():
        if not specification["eligible"]:
            continue
        for condition in ("baseline", "internal", "environment", "both"):
            runs.append(
                run_comparator_condition(
                    architecture=architecture,  # type: ignore[arg-type]
                    condition=condition,  # type: ignore[arg-type]
                    seeds=seeds,
                    hidden_size=specification["hidden_size"],  # type: ignore[arg-type]
                    steps=args.steps,
                    delay_steps=args.delay_steps,
                    environment_exposure=args.environment_exposure,
                )
            )
    payload = {
        "schema_version": 1,
        "experiment": "closed-loop-comparator-pilot",
        "evidence_status": "untrained_initialized_dynamics_only",
        "track": args.track,
        "budget_tracks": tracks,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
