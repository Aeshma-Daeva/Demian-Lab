#!/usr/bin/env python3
"""Run the unselected Demian v1 route-trajectory pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from development.route_trajectory import run_route_trajectory_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--protocol", type=Path, default=Path("data/ROUTE_TRAJECTORY_PROTOCOL.json"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostics/demian_route_trajectory_pilot_20261006/summary.json"),
    )
    args = parser.parse_args()
    payload = run_route_trajectory_pilot(
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        hidden_size=args.hidden_size,
        steps=args.steps,
        protocol_path=args.protocol,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
