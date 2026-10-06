#!/usr/bin/env python3
"""Typed trace collection for the explicit six-channel Demian v1 substrate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from development.demian_v1_gate_state import (
    V1_CHANNELS,
    DemianV1GateState,
    clamp_v1_channel,
    match_v1_surface,
)


UpdateMode = Literal["active", "gate_disabled", "gate_frozen"]
PostStepClamp = Literal[
    "none", "fast", "slow", "control", "message", "carrier", "gate"
]


@dataclass(frozen=True)
class V1TraceConfig:
    seed: int
    hidden_size: int
    steps: int
    parameter_provenance: str = "unselected_random"
    history_condition: str = "uninterrupted"
    update_mode: UpdateMode = "active"
    post_step_clamp: PostStepClamp = "none"
    perturb_step: int | None = None
    perturb_scale: float = 0.0
    device: str = "cpu"
    dtype: str = "float32"
    record_routes: bool = True


@dataclass(frozen=True)
class V1RouteStepRecord:
    step: int
    values: dict[str, list[float]]


@dataclass(frozen=True)
class V1TraceResult:
    config: V1TraceConfig
    surfaces: list[list[float]]
    full_states: list[list[float]]
    channels: dict[str, list[list[float]]]
    metrics: list[dict[str, float]]
    initial_channels: dict[str, list[float]] = field(default_factory=dict)
    channel_scales: dict[str, float] = field(default_factory=dict)
    route_steps: list[V1RouteStepRecord] = field(default_factory=list)


def _validate_config(config: V1TraceConfig) -> None:
    if config.steps < 3:
        raise ValueError("steps must be at least 3")
    if config.hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    if not config.parameter_provenance.strip():
        raise ValueError("parameter_provenance must be non-empty")
    if config.update_mode not in {"active", "gate_disabled", "gate_frozen"}:
        raise ValueError(f"unknown update mode: {config.update_mode}")
    if config.post_step_clamp not in {"none", *V1_CHANNELS}:
        raise ValueError(f"unknown post-step clamp: {config.post_step_clamp}")
    if config.perturb_step is not None and not 1 <= config.perturb_step <= config.steps:
        raise ValueError("perturb_step must fall inside the trace")
    if config.dtype != "float32":
        raise ValueError("phase-one measurement supports float32 only")


def run_v1_measurement(config: V1TraceConfig) -> V1TraceResult:
    """Run one deterministic trace with explicit intervention semantics."""

    _validate_config(config)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    model = DemianV1GateState(
        config.hidden_size,
        gate_disabled=config.update_mode == "gate_disabled",
        gate_frozen=config.update_mode == "gate_frozen",
        trace_routes=config.record_routes,
    ).to(device=device, dtype=torch.float32)
    state = model.initial_state(1, device)
    initial_channels = {
        name: values.reshape(-1).detach().cpu().tolist()
        for name, values in model.state_components(state).items()
    }
    channel_scales = {
        "fast": float(model.init_scale),
        "slow": float(model.init_scale),
        "control": float(model.init_scale),
        "message": float(model.initial_message_scale),
        "carrier": float(model.initial_carrier_scale),
        "gate": float(model.initial_gate_scale),
    }
    surfaces: list[list[float]] = []
    full_states: list[list[float]] = []
    channels: dict[str, list[list[float]]] = {name: [] for name in V1_CHANNELS}
    metrics: list[dict[str, float]] = []
    route_steps: list[V1RouteStepRecord] = []

    with torch.no_grad():
        for step in range(1, config.steps + 1):
            reconstruction_error = 0.0
            if config.perturb_step == step and config.perturb_scale > 0.0:
                surface = model.state_vector(state)
                generator = torch.Generator(device=device).manual_seed(config.seed + step)
                noise = torch.randn(
                    surface.shape,
                    generator=generator,
                    device=device,
                    dtype=surface.dtype,
                )
                target = surface + config.perturb_scale * noise
                state = match_v1_surface(model, state, target)
                reconstruction_error = float(torch.max(torch.abs(model.state_vector(state) - target)).item())
            state = model.step(state)
            if config.post_step_clamp != "none":
                state = clamp_v1_channel(state, config.post_step_clamp)
            components = model.state_components(state)
            surface = model.state_vector(state).reshape(-1).detach().cpu().tolist()
            surfaces.append(surface)
            flattened: list[float] = []
            for name in V1_CHANNELS:
                values = components[name].reshape(-1).detach().cpu().tolist()
                channels[name].append(values)
                flattened.extend(values)
            full_states.append(flattened)
            row = dict(model.step_aux())
            row["surface_reconstruction_error"] = reconstruction_error
            metrics.append(row)
            route_trace = model.route_trace()
            if route_trace is not None:
                route_steps.append(
                    V1RouteStepRecord(
                        step=route_trace.step,
                        values={
                            route_id: value.reshape(-1).detach().cpu().tolist()
                            for route_id, value in route_trace.values.items()
                        },
                    )
                )

    return V1TraceResult(
        config=config,
        initial_channels=initial_channels,
        channel_scales=channel_scales,
        surfaces=surfaces,
        full_states=full_states,
        channels=channels,
        metrics=metrics,
        route_steps=route_steps,
    )
