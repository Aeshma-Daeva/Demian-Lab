"""Unselected closed-loop baseline for the declared Demian environment boundary."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Sequence

from development.closed_loop_demian import ClosedLoopDemianRunner


def run_closed_loop_baseline(
    *,
    seeds: Sequence[int],
    hidden_size: int,
    steps: int,
    delay_steps: int = 2,
    sample_every: int = 1,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    runs: list[dict[str, object]] = []
    for seed in seeds:
        runner = ClosedLoopDemianRunner(
            seed=seed,
            hidden_size=hidden_size,
            cue_symbol=seed % 3,
            delay_steps=delay_steps,
        )
        trace: list[dict[str, object]] = []
        route_l2_total: dict[str, float] = {}
        executed_tick_count = 0
        for step_index in range(steps):
            if runner.world.completed:
                break
            record = runner.step()
            executed_tick_count += 1
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
        runs.append(
            {
                "run_id": f"demian_v1_closed_loop:seed={seed}:unselected",
                "seed": seed,
                "cue_symbol": seed % 3,
                "trace": trace,
                "executed_tick_count": executed_tick_count,
                "route_l2_mean": {
                    route_id: value / executed_tick_count for route_id, value in route_l2_total.items()
                },
                "final_register": runner.world.register,
                "final_score": runner.world.score,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "demian-v1-unselected-closed-loop-baseline",
        "protocol_id": "demian-v1-cue-delay-query-v1",
        "interface_version": "frozen-fast-input-control-decoder-v1",
        "config": {
            "seeds": list(seeds),
            "hidden_size": hidden_size,
            "steps": steps,
            "delay_steps": delay_steps,
            "sample_every": sample_every,
        },
        "replication_scope": {
            "parameter_seed_count": len(seeds),
            "trajectory_count": len(seeds),
            "dependent_tick_count": sum(run["executed_tick_count"] for run in runs),
            "executed_tick_count": sum(run["executed_tick_count"] for run in runs),
            "sampled_tick_count": sum(len(run["trace"]) for run in runs),
            "independent_unit": "parameter_seed",
        },
        "runs": runs,
        "evidence_status": {
            "observation": "A deterministic external boundary records action proposals, execution acknowledgements, environmental observations, and internal route magnitudes.",
            "hypothesis_status": "untested",
            "control": "Exact checkpoint restore and fixed-observation replay are tested separately from closed-loop feedback.",
            "interpretation": "Unselected behavior is an interface baseline, not evidence of task competence or route causality.",
            "untested_speculation": "Environmental feedback may organize internal route trajectories into reproducible situated regimes.",
        },
    }
