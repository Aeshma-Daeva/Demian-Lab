#!/usr/bin/env python3
"""Versioned route-trajectory protocol and deterministic unselected pilot."""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from pathlib import Path
from statistics import mean
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from development.demian_v1_gate_state import V1_CHANNELS, V1_ROUTE_CATALOG
from development.demian_v1_measurement import V1TraceConfig, V1TraceResult, run_v1_measurement


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProtocolRoute(ProtocolModel):
    id: str
    sources: list[str]
    target: str
    operation: Literal["additive", "multiplicative"]
    injection_point: Literal[
        "target_state",
        "gate_context_preactivation",
        "fast_preactivation",
        "multiplicative_modulator",
    ]


class RouteHypothesis(ProtocolModel):
    id: str = Field(pattern=r"^H-ROUTE-[0-9]{3}$")
    statement: str
    operational_prediction: str
    evidence_status: Literal["untested"]
    falsifier: str


class RecordingRules(ProtocolModel):
    retain_absolute_l2: bool
    normalizations: list[str]
    temporal_summaries: list[str]
    active_l2_epsilon: float = Field(gt=0)


class InterventionDefinition(ProtocolModel):
    id: str
    semantics: str
    required_controls: list[str]


class RouteTrajectoryProtocol(ProtocolModel):
    schema_version: int = Field(ge=1)
    protocol_id: str
    system: Literal["demian_v1_gate_state"]
    route_catalog_version: str
    route_catalog: list[ProtocolRoute]
    hypotheses: list[RouteHypothesis]
    recording: RecordingRules
    interventions: list[InterventionDefinition]
    interpretation_rules: list[str]

    @model_validator(mode="after")
    def match_implemented_catalog(self) -> RouteTrajectoryProtocol:
        implemented = {
            route.id: (
                list(route.sources),
                route.target,
                route.operation,
                route.injection_point,
            )
            for route in V1_ROUTE_CATALOG
        }
        declared = {
            route.id: (
                route.sources,
                route.target,
                route.operation,
                route.injection_point,
            )
            for route in self.route_catalog
        }
        if declared != implemented:
            raise ValueError("protocol must match the implemented route catalog")
        if len(declared) != len(self.route_catalog):
            raise ValueError("route catalog IDs must be unique")
        return self


def load_route_protocol(path: Path) -> RouteTrajectoryProtocol:
    return RouteTrajectoryProtocol.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = _l2(left) * _l2(right)
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def _route_summary(
    trace: V1TraceResult,
    protocol: RouteTrajectoryProtocol,
) -> dict[str, dict[str, object]]:
    catalog = {route.id: route for route in protocol.route_catalog}
    summary: dict[str, dict[str, object]] = {}
    for route_id in catalog:
        route = catalog[route_id]
        if route.operation == "multiplicative":
            norms = [
                _l2([value - 1.0 for value in step.values[route_id]])
                for step in trace.route_steps
            ]
        else:
            norms = [_l2(step.values[route_id]) for step in trace.route_steps]
        target = catalog[route_id].target
        target_norms = [_l2(values) for values in trace.channels[target]]
        target_history = [trace.initial_channels[target], *trace.channels[target]]
        target_update_norms = [
            _l2([right - left for left, right in zip(before, after, strict=True)])
            for before, after in zip(target_history, target_history[1:])
        ]
        declared_scale = trace.channel_scales[target] * math.sqrt(len(target_history[0]))
        active = [value > protocol.recording.active_l2_epsilon for value in norms]
        longest = current = 0
        for is_active in active:
            current = current + 1 if is_active else 0
            longest = max(longest, current)
        route_summary: dict[str, object] = {
            "operation": route.operation,
            "norm_basis": (
                "deviation_from_neutral"
                if route.operation == "multiplicative"
                else "contribution_vector"
            ),
            "mean_l2": mean(norms),
            "active_fraction": mean(float(value) for value in active),
            "longest_active_fraction": longest / len(active),
        }
        if route.operation == "additive":
            route_summary["mean_target_state_ratio"] = mean(
                value / max(target_norm, 1e-12)
                for value, target_norm in zip(norms, target_norms, strict=True)
            )
            route_summary["mean_target_update_ratio"] = mean(
                value / max(update_norm, 1e-12)
                for value, update_norm in zip(norms, target_update_norms, strict=True)
            )
            route_summary["mean_declared_scale_ratio"] = mean(norms) / max(
                declared_scale, 1e-12
            )
        summary[route_id] = route_summary
    return summary


def _target_summary(
    trace: V1TraceResult,
    protocol: RouteTrajectoryProtocol,
) -> dict[str, dict[str, float]]:
    catalog = {route.id: route for route in protocol.route_catalog}
    summaries: dict[str, dict[str, float]] = {}
    for target in V1_CHANNELS:
        route_ids = [
            route.id
            for route in protocol.route_catalog
            if route.target == target
            and route.operation == "additive"
            and route.injection_point == "target_state"
        ]
        cancellations: list[float] = []
        alignments: list[float] = []
        for step in trace.route_steps:
            values = [step.values[route_id] for route_id in route_ids]
            total_norm = sum(_l2(value) for value in values)
            summed = [sum(items) for items in zip(*values, strict=True)]
            cancellations.append(1.0 - _l2(summed) / max(total_norm, 1e-12))
            pairs = [
                _cosine(values[left], values[right])
                for left in range(len(values))
                for right in range(left + 1, len(values))
            ]
            alignments.append(mean(pairs) if pairs else 1.0)
        summaries[target] = {
            "mean_cancellation": min(1.0, max(0.0, mean(cancellations))),
            "mean_pair_alignment": mean(alignments),
            "integration_route_count": float(len(route_ids)),
        }
        assert all(catalog[route_id].target == target for route_id in route_ids)
    return summaries


def run_route_trajectory_pilot(
    *,
    seeds: Sequence[int],
    hidden_size: int,
    steps: int,
    protocol_path: Path,
) -> dict[str, object]:
    if not seeds:
        raise ValueError("at least one seed is required")
    protocol = load_route_protocol(protocol_path)
    runs: list[dict[str, object]] = []
    for seed in seeds:
        config = V1TraceConfig(
            seed=seed,
            hidden_size=hidden_size,
            steps=steps,
            parameter_provenance="unselected_random",
            history_condition="uninterrupted",
            record_routes=True,
        )
        trace = run_v1_measurement(config)
        runs.append(
            {
                "run_id": f"demian_v1_route_trajectory:seed={seed}:active:uninterrupted",
                "seed": seed,
                "parameter_provenance": config.parameter_provenance,
                "history_condition": config.history_condition,
                "update_mode": config.update_mode,
                "route_summary": _route_summary(trace, protocol),
                "target_summary": _target_summary(trace, protocol),
                "route_steps": [asdict(step) for step in trace.route_steps],
            }
        )
    return {
        "schema_version": 1,
        "experiment": "demian-v1-route-trajectory-unselected-pilot",
        "protocol_id": protocol.protocol_id,
        "route_catalog_version": protocol.route_catalog_version,
        "config": {"seeds": list(seeds), "hidden_size": hidden_size, "steps": steps},
        "replication_scope": {
            "parameter_seed_count": len(seeds),
            "trajectory_count": len(seeds),
            "route_step_count": len(seeds) * steps,
            "independent_unit": "parameter_seed",
            "dependent_observations": ["route_steps_within_trajectory"],
        },
        "runs": runs,
        "evidence_status": {
            "observation": "Time-resolved route contributions were recorded without selecting regimes.",
            "hypothesis_status": "untested",
            "control": "Trace/no-trace identity and target-state reconstruction tests.",
            "interpretation": "Route sequences are descriptive until matched interventions establish delayed effects.",
            "untested_speculation": "Distinct route trajectories may coexist under similar exposed trajectories.",
        },
    }


def write_route_pilot_artifacts(payload: dict[str, object], summary_path: Path) -> Path:
    """Write compact review JSON plus lossless timestamped route vectors."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    summary = copy.deepcopy(payload)
    rows: list[dict[str, object]] = []
    runs = summary["runs"]
    if not isinstance(runs, list):
        raise ValueError("pilot payload runs must be a list")
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("pilot runs must be objects")
        route_steps = run["route_steps"]
        if not isinstance(route_steps, list):
            raise ValueError("pilot route_steps must be a list")
        record_count = 0
        for route_step in route_steps:
            step = int(route_step["step"])
            for route_id, values in route_step["values"].items():
                rows.append(
                    {
                        "run_id": str(run["run_id"]),
                        "seed": int(run["seed"]),
                        "step": step,
                        "route_id": str(route_id),
                        "values": [float(value) for value in values],
                    }
                )
                record_count += 1
        run["route_steps"] = {
            "artifact": "route_steps.parquet",
            "record_count": record_count,
        }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = summary_path.with_name("route_steps.parquet")
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, compression="zstd")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return parquet_path
