"""Deterministic first-pass internal and external perturbation campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Literal, Sequence

import torch

from development.closed_loop_demian import ClosedLoopDemianRunner
from development.demian_v1_gate_state import V1State


PERTURBATION_RATE = 0.01
INTERNAL_AMPLITUDE = 0.01
EnvironmentMode = Literal["clear", "replace", "mixed", "none"]


@dataclass(frozen=True)
class PerturbationCondition:
    name: Literal["baseline", "internal", "environment", "both"]
    seeds: tuple[int, ...]
    internal_amplitude: float
    environment_mode: EnvironmentMode
    internal_ticks: dict[int, tuple[int, ...]]
    environment_ticks: dict[int, tuple[int, ...]]


def build_four_group_conditions(seeds: Sequence[int], *, steps: int) -> list[PerturbationCondition]:
    if len(seeds) % 4:
        raise ValueError("seed count must divide evenly into four groups")
    if steps < 1:
        raise ValueError("steps must be positive")
    group_size = len(seeds) // 4
    groups = [tuple(seeds[index : index + group_size]) for index in range(0, len(seeds), group_size)]

    def ticks(group: tuple[int, ...], *, salt: int) -> dict[int, tuple[int, ...]]:
        return {seed: _scheduled_ticks(seed, steps=steps, salt=salt) for seed in group}

    empty = lambda group: {seed: () for seed in group}
    return [
        PerturbationCondition("baseline", groups[0], 0.0, "none", empty(groups[0]), empty(groups[0])),
        PerturbationCondition("internal", groups[1], INTERNAL_AMPLITUDE, "none", ticks(groups[1], salt=11), empty(groups[1])),
        PerturbationCondition("environment", groups[2], 0.0, "mixed", empty(groups[2]), ticks(groups[2], salt=17)),
        PerturbationCondition("both", groups[3], INTERNAL_AMPLITUDE, "mixed", ticks(groups[3], salt=23), ticks(groups[3], salt=29)),
    ]


def run_perturbation_campaign(
    *,
    seeds: Sequence[int],
    hidden_size: int,
    steps: int,
    delay_steps: int,
    sample_every: int,
) -> dict[str, object]:
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    conditions = build_four_group_conditions(seeds, steps=steps)
    groups = [_run_condition(condition, hidden_size, steps, delay_steps, sample_every) for condition in conditions]
    return {
        "schema_version": 1,
        "experiment": "demian-v1-four-group-perturbation-pilot",
        "config": {
            "hidden_size": hidden_size,
            "steps": steps,
            "delay_steps": delay_steps,
            "sample_every": sample_every,
            "internal_amplitude": INTERNAL_AMPLITUDE,
            "perturbation_rate": PERTURBATION_RATE,
        },
        "replication_scope": {
            "parameter_seed_count": len(seeds),
            "trajectory_count": len(seeds),
            "group_seed_count": len(seeds) // 4,
            "independent_unit": "parameter_seed",
            "design_note": "Disjoint seed groups make this an exploratory perturbation screen, not a matched causal comparison.",
        },
        "groups": groups,
        "evidence_status": {
            "observation": "The campaign compares declared perturbation conditions across disjoint parameter-seed groups.",
            "hypothesis_status": "untested",
            "interpretation": "Differences cannot be attributed to perturbation type without a matched-seed follow-up.",
        },
    }


def _run_condition(
    condition: PerturbationCondition,
    hidden_size: int,
    steps: int,
    delay_steps: int,
    sample_every: int,
) -> dict[str, object]:
    runs = []
    for position, seed in enumerate(condition.seeds):
        runner = ClosedLoopDemianRunner(
            seed=seed,
            hidden_size=hidden_size,
            cue_symbol=seed % 3,
            delay_steps=delay_steps,
        )
        internal_tick_set = set(condition.internal_ticks[seed])
        environment_tick_set = set(condition.environment_ticks[seed])
        environment_kind: EnvironmentMode = _environment_kind(condition.environment_mode, position)
        trace = []
        events = []
        route_l2_total: dict[str, float] = {}
        for step_index in range(steps):
            if runner.world.completed:
                break
            if step_index in internal_tick_set:
                runner.state = _apply_internal_pulse(runner.state, seed, step_index, condition.internal_amplitude)
                events.append({"tick": step_index, "kind": "internal_pulse", "amplitude": condition.internal_amplitude})
            if step_index in environment_tick_set:
                events.append(_apply_environment_disturbance(runner, environment_kind, step_index))
            record = runner.step()
            route_l2 = {
                route_id: math.sqrt(sum(value * value for value in values))
                for route_id, values in (record.route_step.values if record.route_step else {}).items()
            }
            for route_id, value in route_l2.items():
                route_l2_total[route_id] = route_l2_total.get(route_id, 0.0) + value
            if step_index % sample_every == 0 or step_index == steps - 1 or runner.world.completed:
                trace.append(
                    {
                        "tick": record.observation.tick,
                        "phase": record.observation.phase,
                        "proposal": asdict(record.proposal),
                        "acceptance": asdict(record.acceptance),
                        "executed_acknowledgement": asdict(record.executed_acknowledgement),
                        "surface": record.surface,
                        "route_l2": route_l2,
                    }
                )
        executed_tick_count = steps if not runner.world.completed else step_index + 1
        runs.append(
            {
                "seed": seed,
                "cue_symbol": seed % 3,
                "trace": trace,
                "events": events,
                "executed_tick_count": executed_tick_count,
                "route_l2_mean": {route_id: value / executed_tick_count for route_id, value in route_l2_total.items()},
                "final_register": runner.world.register,
                "final_score": runner.world.score,
            }
        )
    return {
        "name": condition.name,
        "seed_count": len(condition.seeds),
        "internal_amplitude": condition.internal_amplitude,
        "environment_mode": condition.environment_mode,
        "scheduled_internal_events": sum(len(values) for values in condition.internal_ticks.values()),
        "scheduled_environment_events": sum(len(values) for values in condition.environment_ticks.values()),
        "runs": runs,
    }


def _scheduled_ticks(seed: int, *, steps: int, salt: int) -> tuple[int, ...]:
    count = max(1, round(steps * PERTURBATION_RATE))
    return tuple(sorted(random.Random(f"{seed}:{salt}").sample(range(steps), count)))


def _apply_internal_pulse(state: V1State, seed: int, tick: int, amplitude: float) -> V1State:
    if amplitude == 0.0:
        return state
    pulses = []
    for channel_index, component in enumerate(state):
        generator = torch.Generator(device=component.device).manual_seed(seed * 100_000 + tick * 10 + channel_index)
        direction = torch.randn(component.shape, generator=generator, device=component.device, dtype=component.dtype)
        pulses.append(component + direction / torch.linalg.vector_norm(direction) * (amplitude / math.sqrt(len(state))))
    return tuple(pulses)  # type: ignore[return-value]


def _environment_kind(mode: EnvironmentMode, position: int) -> EnvironmentMode:
    if mode != "mixed":
        return mode
    return "clear" if position % 2 == 0 else "replace"


def _apply_environment_disturbance(
    runner: ClosedLoopDemianRunner,
    mode: EnvironmentMode,
    tick: int,
) -> dict[str, object]:
    before = runner.world.register
    if mode == "clear":
        runner.world.reset_storage()
    elif mode == "replace":
        basis = runner.world.cue_symbol if before is None else before
        runner.world.register = (basis + 1) % runner.world.symbol_count
    else:
        raise ValueError("environment disturbance requires clear or replace mode")
    return {"tick": tick, "kind": f"register_{mode}", "before": before, "after": runner.world.register}
