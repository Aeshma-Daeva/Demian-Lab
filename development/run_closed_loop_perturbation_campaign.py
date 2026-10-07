#!/usr/bin/env python3
"""Run the deterministic four-group closed-loop perturbation screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from development.closed_loop_perturbation import run_matched_perturbation_campaign, run_perturbation_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94-193")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--delay-steps", type=int, default=2046)
    parser.add_argument("--sample-every", type=int, default=32)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostics/demian_closed_loop_perturbation_h256_t2048_s100_20261007/summary.json"),
    )
    args = parser.parse_args()
    start, end = (int(value) for value in args.seeds.split("-", maxsplit=1))
    runner = run_matched_perturbation_campaign if args.matched else run_perturbation_campaign
    payload = runner(
        seeds=list(range(start, end + 1)),
        hidden_size=args.hidden_size,
        steps=args.steps,
        delay_steps=args.delay_steps,
        sample_every=args.sample_every,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
