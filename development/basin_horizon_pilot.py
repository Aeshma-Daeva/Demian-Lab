#!/usr/bin/env python3
"""Finite-window basin and horizon pilot for the six-channel Demian system."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from development.afp_v2_characterization import AFPV2Config, classify_internal_dynamics
from development.demian_v1_gate_state import V1_CHANNELS, V1State, DemianV1GateState, clone_state
from development.demian_v1_measurement import V1TraceConfig
from development.lab_schemas import (
    BasinHorizonPilotSpec,
    BasinHorizonRunRecord,
    FiniteWindowProtocol,
    load_basin_horizon_spec,
)
from development.probe_demian_v1_continuation import run_v1_continuation_probe


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in values) / len(values))


def _trace_deltas(trace: Sequence[Sequence[float]]) -> list[float]:
    return [
        _rms([float(right) - float(left) for left, right in zip(previous, current, strict=True)])
        for previous, current in zip(trace, trace[1:])
    ]


def _state_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return _rms([float(a) - float(b) for a, b in zip(left, right, strict=True)])


def perturb_reference_state(
    state: V1State,
    radius: float,
    direction_seed: int | None,
    scale_floor: float,
) -> V1State:
    """Sample a sphere direction normalized independently by channel RMS."""

    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    if radius == 0.0:
        if direction_seed is not None:
            raise ValueError("radius-zero reference must not carry a direction seed")
        return clone_state(state)
    if direction_seed is None:
        raise ValueError("positive-radius perturbations require a direction seed")

    generator = torch.Generator(device=state[0].device).manual_seed(direction_seed)
    perturbed = []
    for channel in state:
        direction = torch.randn(
            channel.shape,
            generator=generator,
            device=channel.device,
            dtype=channel.dtype,
        )
        direction_rms = torch.sqrt(torch.mean(direction.square()))
        channel_rms = max(float(torch.sqrt(torch.mean(channel.square())).item()), scale_floor)
        delta = radius * channel_rms * direction / direction_rms
        perturbed.append(channel.detach().clone() + delta)
    return tuple(perturbed)  # type: ignore[return-value]


def classify_checkpoint(
    surface_trace: Sequence[Sequence[float]],
    full_state_trace: Sequence[Sequence[float]],
    checkpoint: int,
    protocol: FiniteWindowProtocol,
) -> dict[str, Any]:
    """Classify one finite prefix with explicit dwell, persistence, and censoring."""

    if checkpoint < 3 or checkpoint > len(surface_trace) or checkpoint > len(full_state_trace):
        raise ValueError("checkpoint must address at least three recorded states")
    surface = list(surface_trace[:checkpoint])
    full_state = list(full_state_trace[:checkpoint])
    deltas = _trace_deltas(surface)
    dwell = int(protocol.settling_dwell_steps)
    settlement_start = next(
        (
            start
            for start in range(len(deltas) - dwell + 1)
            if all(value <= protocol.surface_delta_tolerance for value in deltas[start : start + dwell])
            and _state_distance(surface[start], surface[start + dwell]) / dwell
            <= protocol.surface_drift_tolerance
        ),
        None,
    )
    settlement_step = None if settlement_start is None else settlement_start + dwell + 1

    if settlement_step is None:
        return {
            "checkpoint": checkpoint,
            "surface_status": "unsettled_at_checkpoint",
            "settlement_step": None,
            "escape_step": None,
            "censoring": "unsettled_at_checkpoint",
            "internal_regime": None,
            "afp_status": "not_evaluable",
            "continuation_effect": None,
            "predictive_information": None,
            "task_utility": None,
        }

    settled_states = surface[settlement_start : settlement_start + dwell + 1]
    settled_center = [
        sum(float(row[index]) for row in settled_states) / len(settled_states)
        for index in range(len(settled_states[0]))
    ]
    escape_step = next(
        (
            state_index + 1
            for state_index in range(settlement_step, len(surface))
            if _state_distance(surface[state_index], settled_center) > protocol.surface_escape_tolerance
        ),
        None,
    )
    if escape_step is not None:
        return {
            "checkpoint": checkpoint,
            "surface_status": "escaped_after_settlement",
            "settlement_step": settlement_step,
            "escape_step": escape_step,
            "censoring": "surface_escape",
            "internal_regime": None,
            "afp_status": "not_afp_v2",
            "continuation_effect": None,
            "predictive_information": None,
            "task_utility": None,
        }

    persistence_end = settlement_step + int(protocol.persistence_steps)
    if persistence_end > checkpoint:
        return {
            "checkpoint": checkpoint,
            "surface_status": "operationally_converged",
            "settlement_step": settlement_step,
            "escape_step": None,
            "censoring": "right_censored",
            "internal_regime": None,
            "afp_status": "not_evaluable",
            "continuation_effect": None,
            "predictive_information": None,
            "task_utility": None,
        }

    persistence_trace = full_state[settlement_step - 1 : persistence_end]
    internal_config = AFPV2Config(tail_steps=len(persistence_trace))
    internal = classify_internal_dynamics(persistence_trace, internal_config)
    internal_regime = str(internal["classification"])
    if internal_regime == "internal_structured_persistent":
        afp_status = "afp_v2_candidate"
    elif internal_regime == "internal_fixed":
        afp_status = "operational_full_state_fixed_point"
    else:
        afp_status = "not_afp_v2"
    return {
        "checkpoint": checkpoint,
        "surface_status": "operationally_converged",
        "settlement_step": settlement_step,
        "escape_step": None,
        "censoring": "complete",
        "internal_regime": internal_regime,
        "internal_metrics": internal["metrics"],
        "internal_classifier_config": asdict(internal_config),
        "afp_status": afp_status,
        "continuation_effect": None,
        "predictive_information": None,
        "task_utility": None,
    }


def _run_trace(
    model: DemianV1GateState,
    state: V1State,
    steps: int,
) -> tuple[list[list[float]], list[list[float]]]:
    surfaces: list[list[float]] = []
    full_states: list[list[float]] = []
    with torch.no_grad():
        for _ in range(steps):
            state = model.step(state)
            components = model.state_components(state)
            surfaces.append(model.state_vector(state).reshape(-1).detach().cpu().tolist())
            full_states.append(
                [
                    value
                    for name in V1_CHANNELS
                    for value in components[name].reshape(-1).detach().cpu().tolist()
                ]
            )
    return surfaces, full_states


def _aggregate_checkpoint_rows(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, int], dict[str, int]] = {}
    censoring: dict[str, int] = {}
    for run in runs:
        for checkpoint in run["checkpoints"]:
            key = (str(run["update_mode"]), float(run["radius"]), int(checkpoint["checkpoint"]))
            counts = grouped.setdefault(key, {})
            label = str(checkpoint["afp_status"])
            counts[label] = counts.get(label, 0) + 1
            censor = str(checkpoint["censoring"])
            censoring[censor] = censoring.get(censor, 0) + 1
    return {
        "regime_frequency": [
            {
                "update_mode": mode,
                "radius": radius,
                "checkpoint": checkpoint,
                "counts": dict(sorted(counts.items())),
                "sample_count": sum(counts.values()),
            }
            for (mode, radius, checkpoint), counts in sorted(grouped.items())
        ],
        "censoring_counts": dict(sorted(censoring.items())),
    }


def run_basin_horizon_pilot(
    spec: BasinHorizonPilotSpec,
    *,
    model_seeds: Sequence[int] | None = None,
    include_continuation_controls: bool = False,
) -> dict[str, Any]:
    """Run every sampled trajectory once and classify all checkpoint prefixes."""

    hidden_size = int(spec.system.get("hidden_size", 16))
    maximum_horizon = max(spec.finite_window_protocol.checkpoints)
    modes = spec.conditions["architecture_modes"]
    selected_seeds = list(model_seeds or spec.replication.model_seeds)
    if not selected_seeds or not set(selected_seeds).issubset(spec.replication.model_seeds):
        raise ValueError("model_seeds must be a non-empty subset of the registered replication seeds")
    runs: list[dict[str, Any]] = []
    continuation_controls: list[dict[str, Any]] = []

    for model_seed in selected_seeds:
        for update_mode in modes:
            torch.manual_seed(model_seed)
            reference_model = DemianV1GateState(
                hidden_size,
                gate_disabled=update_mode == "gate_disabled",
                gate_frozen=update_mode == "gate_frozen",
            ).to(dtype=torch.float32)
            reference_state = reference_model.initial_state(1, torch.device("cpu"))
            with torch.no_grad():
                for _ in range(spec.basin_probe.reference_steps):
                    reference_state = reference_model.step(reference_state)

            reference_id = f"seed={model_seed}:{update_mode}:reference=0"
            continuation = None
            if include_continuation_controls:
                continuation = run_v1_continuation_probe(
                    V1TraceConfig(
                        seed=model_seed,
                        hidden_size=hidden_size,
                        steps=spec.basin_probe.reference_steps + maximum_horizon,
                        update_mode=update_mode,
                    ),
                    pause_steps=spec.basin_probe.reference_steps,
                    resume_steps=maximum_horizon,
                )
                continuation_controls.append(
                    {
                        "reference_id": reference_id,
                        "model_seed": model_seed,
                        "update_mode": update_mode,
                        "claim_boundary": (
                            "Each arm tests one reconstruction; body_surface does not establish that every "
                            "surface-only reconstruction is insufficient."
                        ),
                        **continuation,
                    }
                )
            conditions: list[tuple[float, int | None]] = [(0.0, None)]
            conditions.extend(
                (float(radius), direction_seed)
                for radius in spec.basin_probe.radii
                if radius > 0.0
                for direction_seed in spec.basin_probe.direction_seeds
            )
            for radius, direction_seed in conditions:
                trajectory_id = f"{reference_id}:r={radius:g}"
                if direction_seed is not None:
                    trajectory_id += f":direction={direction_seed}"
                    dependency_group_id = f"{reference_id}:direction={direction_seed}"
                else:
                    dependency_group_id = reference_id
                state = perturb_reference_state(
                    reference_state,
                    radius,
                    direction_seed,
                    spec.basin_probe.channel_scale_floor,
                )
                surfaces, full_states = _run_trace(
                    copy.deepcopy(reference_model),
                    state,
                    maximum_horizon,
                )
                checkpoint_rows = [
                    classify_checkpoint(
                        surfaces,
                        full_states,
                        checkpoint,
                        spec.finite_window_protocol,
                    )
                    for checkpoint in spec.finite_window_protocol.checkpoints
                ]
                record = BasinHorizonRunRecord.model_validate(
                    {
                        "run_id": trajectory_id,
                        "spec_id": spec.spec_id,
                        "model_seed": model_seed,
                        "update_mode": update_mode,
                        "reference_id": reference_id,
                        "trajectory_id": trajectory_id,
                        "dependency_group_id": dependency_group_id,
                        "forcing_protocol": "autonomous_after_reference_state",
                        "radius": radius,
                        "direction_seed": direction_seed,
                        "checkpoints": checkpoint_rows,
                        "continuation": continuation if radius == 0.0 else None,
                        "steps_run": maximum_horizon,
                        "development_stage": (
                            "threshold_development"
                            if model_seed in spec.selection_control.pilot_seeds
                            else "held_out_confirmation"
                        ),
                    }
                )
                runs.append(record.model_dump(mode="json"))

    return {
        "schema_version": 1,
        "experiment": "demian-basin-horizon-afpv2-pilot",
        "spec": spec.model_dump(mode="json"),
        "executed_model_seeds": selected_seeds,
        "run_count": len(runs),
        "runs": runs,
        "aggregates": _aggregate_checkpoint_rows(runs),
        "continuation_controls": continuation_controls,
        "evidence_status": {
            "observation": "Finite checkpoint classifications only.",
            "hypothesis": "AFP-v2 may occur under declared local state-space conditions.",
            "control": "Radius zero is paired with channel-normalized sphere perturbations.",
            "interpretation": "Empirical local frequencies are not basin volumes or global prevalence.",
            "untested_speculation": "Internal regimes may predict task-relevant continuation.",
        },
    }


def _flat_checkpoint_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in payload["runs"]:
        for checkpoint in run["checkpoints"]:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "trajectory_id": run["trajectory_id"],
                    "dependency_group_id": run["dependency_group_id"],
                    "reference_id": run["reference_id"],
                    "model_seed": run["model_seed"],
                    "update_mode": run["update_mode"],
                    "development_stage": run["development_stage"],
                    "radius": run["radius"],
                    "direction_seed": run["direction_seed"],
                    "checkpoint": checkpoint["checkpoint"],
                    "surface_status": checkpoint["surface_status"],
                    "settlement_step": checkpoint["settlement_step"],
                    "escape_step": checkpoint["escape_step"],
                    "censoring": checkpoint["censoring"],
                    "internal_regime": checkpoint["internal_regime"],
                    "afp_status": checkpoint["afp_status"],
                    "continuation_effect": checkpoint["continuation_effect"],
                    "predictive_information": checkpoint["predictive_information"],
                    "task_utility": checkpoint["task_utility"],
                }
            )
    return rows


def write_pilot_artifacts(payload: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write reviewable JSON and searchable checkpoint rows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoints.parquet"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(_flat_checkpoint_rows(payload)), checkpoint_path)
    return {"summary": summary_path, "checkpoints": checkpoint_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/superpowers/specs/2026-10-05-demian-basin-horizon-pilot.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/diagnostics/demian_basin_horizon_pilot_20261005"),
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--continuation-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = load_basin_horizon_spec(args.spec)
    seeds = None
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    payload = run_basin_horizon_pilot(
        spec,
        model_seeds=seeds,
        include_continuation_controls=args.continuation_controls,
    )
    artifacts = write_pilot_artifacts(payload, args.out)
    print(artifacts["summary"])
    print(artifacts["checkpoints"])


if __name__ == "__main__":
    main()
