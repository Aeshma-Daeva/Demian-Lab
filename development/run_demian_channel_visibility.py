"""Run a frozen Demian-only paired channel-visibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from development.closed_loop_demian import InputFrame
from development.closed_loop_world import ActionAcknowledgement, WorldObservation
from development.demian_channel_visibility import run_channel_visibility_cell
from development.demian_v1_gate_state import V1_CHANNELS


ROOT = Path(__file__).resolve().parents[1]


def _frames(*, seed: int, steps: int) -> list[InputFrame]:
    acknowledgement = ActionAcknowledgement("noop", None, None, None)
    return [
        InputFrame(
            observation=WorldObservation(
                tick=tick,
                phase="cue" if tick == 0 else "query" if tick == steps - 1 else "delay",
                cue_symbol=seed % 3 if tick == 0 else None,
                query=tick == steps - 1,
            ),
            acknowledgement=acknowledgement,
        )
        for tick in range(steps)
    ]


def _validate(protocol: dict) -> None:
    required = {
        "protocol_id",
        "seeds",
        "channels",
        "direction_seeds",
        "hidden_size",
        "steps",
        "intervention_tick",
        "amplitude",
        "visibility_threshold",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    if not protocol["seeds"] or len(set(protocol["seeds"])) != len(protocol["seeds"]):
        raise ValueError("protocol requires unique nonempty seeds")
    if list(protocol["channels"]) != list(V1_CHANNELS):
        raise ValueError("protocol channels must be the ordered six-channel state")
    if not protocol["direction_seeds"] or len(set(protocol["direction_seeds"])) != len(protocol["direction_seeds"]):
        raise ValueError("protocol requires unique nonempty direction seeds")
    if protocol["hidden_size"] < 1 or protocol["steps"] < 2:
        raise ValueError("protocol requires positive width and at least two steps")
    if not 0 <= protocol["intervention_tick"] < protocol["steps"]:
        raise ValueError("intervention_tick must index the fixed stream")
    for name in ("amplitude", "visibility_threshold"):
        if not math.isfinite(protocol[name]) or protocol[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _compact_cell(result: dict[str, object]) -> dict[str, object]:
    trajectory = result["trajectory"]
    assert isinstance(trajectory, list)
    return {
        "seed": result["seed"],
        "channel": result["channel"],
        "direction_seed": result["config"]["direction_seed"],  # type: ignore[index]
        "initial_channel_pulse_norm": result["initial_channel_pulse_norm"],
        "visibility_tick": result["visibility_tick"],
        "visibility_transition": result["visibility_transition"],
        "visibility_delay": result["visibility_delay"],
        "peak_paired_surface_separation": max(point["paired_surface_separation"] for point in trajectory),  # type: ignore[index]
        "final_paired_surface_separation": trajectory[-1]["paired_surface_separation"],  # type: ignore[index]
        "peak_pair_asymmetry": max(point["pair_asymmetry"] for point in trajectory),  # type: ignore[index]
        "trajectory": trajectory,
    }


def run_visibility_protocol(protocol: dict) -> dict[str, object]:
    """Evaluate the complete declared seed × channel × direction grid."""
    _validate(protocol)
    cells = []
    for seed in protocol["seeds"]:
        frames = _frames(seed=seed, steps=protocol["steps"])
        for channel in protocol["channels"]:
            for direction_seed in protocol["direction_seeds"]:
                result = run_channel_visibility_cell(
                    seed=seed,
                    hidden_size=protocol["hidden_size"],
                    frames=frames,
                    channel=channel,
                    intervention_tick=protocol["intervention_tick"],
                    amplitude=protocol["amplitude"],
                    direction_seed=direction_seed,
                    visibility_threshold=protocol["visibility_threshold"],
                )
                cells.append(_compact_cell(result))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell["channel"])].append(cell)
    channel_summary = {
        channel: {
            "cells": len(rows),
            "visible_cells": sum(row["visibility_tick"] is not None for row in rows),
            "mean_visibility_delay_visible_only": mean(
                row["visibility_delay"] for row in rows if row["visibility_delay"] is not None
            )
            if any(row["visibility_delay"] is not None for row in rows)
            else None,
            "mean_peak_paired_surface_separation": mean(row["peak_paired_surface_separation"] for row in rows),
        }
        for channel, rows in grouped.items()
    }
    return {"cells": cells, "channel_summary": channel_summary}


def write_visibility_artifacts(out: Path, protocol: dict, payload: dict[str, object]) -> None:
    """Create a fresh compact artifact directory; never overwrite a run."""
    out.mkdir(parents=True, exist_ok=False)
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    manifest = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "claim_boundary": protocol.get("claim_boundary", "local visibility only"),
        "cell_count": len(payload["cells"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (out / "cells.jsonl").open("x") as stream:
        for cell in payload["cells"]:
            stream.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "schema_version": 1,
        "experiment": "demian-channel-visibility",
        "claim_boundary": manifest["claim_boundary"],
        "cell_count": manifest["cell_count"],
        "channel_summary": payload["channel_summary"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/DEMIAN_CHANNEL_VISIBILITY_PROTOCOL.json")
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    payload = run_visibility_protocol(protocol)
    write_visibility_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), "cells": len(payload["cells"])}))


if __name__ == "__main__":
    main()
