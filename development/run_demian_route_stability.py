"""Run preregistered fixed-forcing Demian route-stability cells."""

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
from development.demian_route_stability import ROUTE_SCALE_ATTRIBUTES, run_route_stability_cell
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
        "development_seeds",
        "heldout_seeds",
        "hidden_size",
        "steps",
        "channels",
        "route_id",
        "intervention_ticks",
        "amplitudes",
        "direction_seeds",
        "random_direction_seed_offset",
        "recovery_threshold",
        "recovery_dwell",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    for split in ("development_seeds", "heldout_seeds"):
        values = protocol[split]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{split} must be nonempty and unique")
    if set(protocol["development_seeds"]) & set(protocol["heldout_seeds"]):
        raise ValueError("development and heldout seeds must not overlap")
    if protocol["hidden_size"] < 1 or protocol["steps"] < 2:
        raise ValueError("protocol requires positive width and at least two steps")
    if not protocol["channels"] or any(channel not in V1_CHANNELS for channel in protocol["channels"]):
        raise ValueError("protocol channels must be implemented Demian channels")
    if protocol["route_id"] not in ROUTE_SCALE_ATTRIBUTES:
        raise ValueError("protocol route must support a scale intervention")
    route_multipliers = protocol.get("route_multipliers", [0.0])
    if not route_multipliers or any(
        not math.isfinite(value) or value < 0 for value in route_multipliers
    ):
        raise ValueError("route multipliers must be finite and nonnegative")
    if not protocol["intervention_ticks"] or any(
        not 0 <= tick < protocol["steps"] for tick in protocol["intervention_ticks"]
    ):
        raise ValueError("intervention ticks must index the prerecorded stream")
    if not protocol["amplitudes"] or any(not math.isfinite(value) or value <= 0 for value in protocol["amplitudes"]):
        raise ValueError("amplitudes must be finite and positive")
    if not protocol["direction_seeds"] or len(set(protocol["direction_seeds"])) != len(protocol["direction_seeds"]):
        raise ValueError("direction seeds must be nonempty and unique")
    if not math.isfinite(protocol["recovery_threshold"]) or protocol["recovery_threshold"] <= 0:
        raise ValueError("recovery threshold must be finite and positive")
    if protocol["recovery_dwell"] < 1:
        raise ValueError("recovery dwell must be positive")
    if not isinstance(protocol.get("include_random_route_interaction", False), bool):
        raise ValueError("include_random_route_interaction must be boolean")


def run_route_stability_protocol(protocol: dict) -> dict[str, object]:
    """Evaluate each declared split × seed × time × amplitude cell."""
    _validate(protocol)
    cells: list[dict[str, object]] = []
    for split, seeds in (("development", protocol["development_seeds"]), ("heldout", protocol["heldout_seeds"])):
        for seed in seeds:
            frames = _frames(seed=seed, steps=protocol["steps"])
            for channel in protocol["channels"]:
                for intervention_tick in protocol["intervention_ticks"]:
                    for amplitude in protocol["amplitudes"]:
                        for route_multiplier in protocol.get("route_multipliers", [0.0]):
                            for direction_seed in protocol["direction_seeds"]:
                                cell = run_route_stability_cell(
                                    seed=seed,
                                    hidden_size=protocol["hidden_size"],
                                    frames=frames,
                                    channel=channel,
                                    route_id=protocol["route_id"],
                                    intervention_tick=intervention_tick,
                                    amplitude=amplitude,
                                    direction_seed=direction_seed,
                                    random_direction_seed=direction_seed + protocol["random_direction_seed_offset"],
                                    recovery_threshold=protocol["recovery_threshold"],
                                    recovery_dwell=protocol["recovery_dwell"],
                                    route_multiplier=route_multiplier,
                                    include_random_route_interaction=protocol.get("include_random_route_interaction", False),
                                )
                                cells.append({"split": split, **cell})
    return {"cells": cells, "summary": _summarize(cells)}


def _summarize(cells: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for cell in cells:
        config = cell["config"]
        assert isinstance(config, dict)
        groups[(str(cell["split"]), str(cell["channel"]), float(config["route_multiplier"]))].append(cell)
    summary = {}
    for (split, channel, route_multiplier), rows in groups.items():
        arm_summary = {}
        for arm_name in ("lesion_only", "targeted_lesion", "sham", "random_direction"):
            peaks: list[float] = []
            recovered = 0
            for row in rows:
                arms = row["arms"]
                assert isinstance(arms, dict)
                arm = arms[arm_name]
                assert isinstance(arm, dict)
                trajectory = arm["trajectory"]
                assert isinstance(trajectory, list)
                gains = [point["finite_time_state_gain"] for point in trajectory]
                if all(gain is not None for gain in gains):
                    peaks.append(max(float(gain) for gain in gains))
                recovery = arm["recovery"]
                assert isinstance(recovery, dict)
                recovered += int(bool(recovery["recovered"]))
            arm_summary[arm_name] = {
                "cell_count": len(rows),
                "mean_peak_finite_time_state_gain": None if not peaks else mean(peaks),
                "recovered_cells": recovered,
                "censored_cells": len(rows) - recovered,
            }
        interaction_peaks = [
            max(float(point["state_interaction_norm"]) for point in row["interaction_trajectory"])
            for row in rows
        ]
        group_summary = {
            "arms": arm_summary,
            "mean_peak_state_interaction_norm": mean(interaction_peaks),
        }
        random_interaction_peaks = [
            max(float(point["state_interaction_norm"]) for point in row["random_interaction_trajectory"])
            for row in rows
            if "random_interaction_trajectory" in row
        ]
        if random_interaction_peaks:
            group_summary["mean_peak_random_state_interaction_norm"] = mean(random_interaction_peaks)
            group_summary["mean_peak_selectivity_gap"] = (
                group_summary["mean_peak_state_interaction_norm"] - group_summary["mean_peak_random_state_interaction_norm"]
            )
        summary[f"{split}:{channel}:route_multiplier={route_multiplier:g}"] = group_summary
    return summary


def write_route_stability_artifacts(out: Path, protocol: dict, payload: dict[str, object]) -> None:
    """Write a new non-overwriting run directory with protocol and cell evidence."""
    out.mkdir(parents=True, exist_ok=False)
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    manifest = {
        "schema_version": 1,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "claim_boundary": protocol.get("claim_boundary", "fixed-forcing route intervention only"),
        "cell_count": len(payload["cells"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (out / "cells.jsonl").open("x") as stream:
        for cell in payload["cells"]:
            stream.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": "demian-fixed-forcing-route-stability",
                "claim_boundary": manifest["claim_boundary"],
                "cell_count": manifest["cell_count"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/DEMIAN_ROUTE_STABILITY_PROTOCOL.json")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    payload = run_route_stability_protocol(protocol)
    write_route_stability_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), "cells": len(payload["cells"])}))


if __name__ == "__main__":
    main()
