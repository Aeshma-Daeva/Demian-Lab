#!/usr/bin/env python3
"""Run the deterministic unselected closed-loop Demian baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from development.closed_loop_baseline import run_closed_loop_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--delay-steps", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostics/demian_closed_loop_baseline_20261006/summary.json"),
    )
    args = parser.parse_args()
    payload = run_closed_loop_baseline(
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
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
