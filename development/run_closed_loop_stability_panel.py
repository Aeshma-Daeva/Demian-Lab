#!/usr/bin/env python3
"""Run fixed-input recovery measurements for state-matched comparators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from development.closed_loop_comparators import build_budget_tracks
from development.closed_loop_stability import run_stability_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument("--reference-hidden-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--delay-steps", type=int, default=510)
    parser.add_argument("--amplitudes", default="0,0.001,0.01,0.1")
    parser.add_argument("--intervention-ticks", default="128,256,384")
    parser.add_argument("--out", type=Path, default=Path("data/diagnostics/closed_loop_stability_panel/summary.json"))
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    tracks = build_budget_tracks(reference_hidden_size=args.reference_hidden_size, seed=seeds[0])
    specifications = tracks["state"]["specifications"]
    architectures = ["rnn", "gru", "demian_route_ablation", "demian"]
    hidden_sizes = {architecture: specifications[architecture]["hidden_size"] for architecture in architectures}
    payload = run_stability_panel(
        architectures=architectures,  # type: ignore[arg-type]
        seeds=seeds,
        hidden_sizes=hidden_sizes,  # type: ignore[arg-type]
        steps=args.steps,
        delay_steps=args.delay_steps,
        amplitudes=[float(value) for value in args.amplitudes.split(",")],
        intervention_ticks=[int(value) for value in args.intervention_ticks.split(",")],
    )
    payload["budget_track"] = tracks["state"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
