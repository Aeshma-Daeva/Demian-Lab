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
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if steps < 1:
        raise ValueError("steps must be positive")
    runs: list[dict[str, object]] = []
    for seed in seeds:
        runner = ClosedLoopDemianRunner(
            seed=seed,
            hidden_size=hidden_size,
            cue_symbol=seed % 3,
            delay_steps=2,
        )
        trace: list[dict[str, object]] = []
        for _ in range(steps):
            if runner.world.completed:
                break
            record = runner.step()
            route_l2 = {
                route_id: math.sqrt(sum(value * value for value in values))
                for route_id, values in (record.route_step.values if record.route_step else {}).items()
            }
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
                "final_register": runner.world.register,
                "final_score": runner.world.score,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "demian-v1-unselected-closed-loop-baseline",
        "protocol_id": "demian-v1-cue-delay-query-v1",
        "interface_version": "frozen-fast-input-control-decoder-v1",
        "config": {"seeds": list(seeds), "hidden_size": hidden_size, "steps": steps},
        "replication_scope": {
            "parameter_seed_count": len(seeds),
            "trajectory_count": len(seeds),
            "dependent_tick_count": sum(len(run["trace"]) for run in runs),
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
