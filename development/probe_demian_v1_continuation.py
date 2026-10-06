#!/usr/bin/env python3
"""Tail-aligned continuation controls for explicit six-channel Demian v1."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch

from development.demian_v1_gate_state import (
    V1_CHANNELS,
    V1State,
    DemianV1GateState,
    clamp_v1_channel,
    clone_state,
    match_v1_surface,
)
from development.demian_v1_measurement import V1TraceConfig


def _step(model: DemianV1GateState, state: V1State, config: V1TraceConfig) -> V1State:
    state = model.step(state)
    if config.post_step_clamp != "none":
        state = clamp_v1_channel(state, config.post_step_clamp)
    return state


def _apply_history(
    model: DemianV1GateState,
    state: V1State,
    config: V1TraceConfig,
    step: int,
) -> V1State:
    if config.perturb_step != step or config.perturb_scale <= 0.0:
        return state
    surface = model.state_vector(state)
    generator = torch.Generator(device=surface.device).manual_seed(config.seed + step)
    noise = torch.randn(
        surface.shape,
        generator=generator,
        device=surface.device,
        dtype=surface.dtype,
    )
    return match_v1_surface(model, state, surface + config.perturb_scale * noise)


def _surfaces(
    model: DemianV1GateState,
    state: V1State,
    config: V1TraceConfig,
    steps: int,
    start_step: int,
) -> list[torch.Tensor]:
    rows = []
    with torch.no_grad():
        for step in range(start_step, start_step + steps):
            state = _apply_history(model, state, config, step)
            state = _step(model, state, config)
            rows.append(model.state_vector(state).reshape(-1).detach().cpu())
    return rows


def _summarize(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, float]:
    gaps = [
        float(torch.norm(left - right).item() / math.sqrt(max(left.numel(), 1)))
        for left, right in zip(reference, candidate, strict=True)
    ]
    return {
        "final_gap_vs_uninterrupted": gaps[-1],
        "mean_step_gap_vs_uninterrupted": float(sum(gaps) / len(gaps)),
    }


def _surface_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.norm(left - right).item() / math.sqrt(max(left.numel(), 1)))


def _random_norm_matched(state: V1State, seed: int) -> tuple[V1State, float]:
    generator = torch.Generator(device=state[0].device).manual_seed(seed)
    result = []
    mismatches = []
    for component in state:
        target_norm = torch.norm(component)
        random = torch.randn(
            component.shape,
            generator=generator,
            device=component.device,
            dtype=component.dtype,
        )
        random_norm = torch.norm(random)
        matched = torch.zeros_like(component) if target_norm == 0 else random * (target_norm / random_norm)
        result.append(matched)
        mismatches.append(float(torch.abs(torch.norm(matched) - target_norm).item()))
    return tuple(result), max(mismatches)  # type: ignore[return-value]


def run_v1_continuation_probe(
    config: V1TraceConfig,
    pause_steps: int,
    resume_steps: int,
) -> dict[str, Any]:
    """Compare exact, surface, channel, sham, and random continuation states."""

    if pause_steps < 1:
        raise ValueError("pause_steps must be positive")
    if resume_steps < 3:
        raise ValueError("resume_steps must be at least 3")
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    source = DemianV1GateState(
        config.hidden_size,
        gate_disabled=config.update_mode == "gate_disabled",
        gate_frozen=config.update_mode == "gate_frozen",
    ).to(device=device, dtype=torch.float32)
    state = source.initial_state(1, device)
    reachable_history = [
        (0, clone_state(state), source.state_vector(state).reshape(-1).detach().clone())
    ]
    with torch.no_grad():
        for step in range(1, pause_steps + 1):
            state = _apply_history(source, state, config, step)
            state = _step(source, state, config)
            if step < pause_steps:
                reachable_history.append(
                    (
                        step,
                        clone_state(state),
                        source.state_vector(state).reshape(-1).detach().clone(),
                    )
                )
    paused_model = copy.deepcopy(source)
    paused_state = clone_state(state)
    paused_surface = paused_model.state_vector(paused_state).detach().clone()
    reference = _surfaces(
        copy.deepcopy(paused_model),
        clone_state(paused_state),
        config,
        resume_steps,
        pause_steps + 1,
    )

    empty = tuple(torch.zeros_like(component) for component in paused_state)
    arm_states: dict[str, V1State] = {
        "full_checkpoint": clone_state(paused_state),
        "sham": clone_state(paused_state),
        "body_surface": match_v1_surface(paused_model, empty, paused_surface),
    }
    reachable_step, reachable_state, reachable_surface = min(
        reachable_history,
        key=lambda item: _surface_gap(item[2], paused_surface.reshape(-1)),
    )
    reachable_error = _surface_gap(reachable_surface, paused_surface.reshape(-1))
    arm_states["reachable_surface_match"] = clone_state(reachable_state)
    random_state, random_mismatch = _random_norm_matched(paused_state, config.seed + 20_000)
    arm_states["random_norm_matched"] = random_state
    for index, name in enumerate(V1_CHANNELS):
        parts = [torch.zeros_like(component) for component in paused_state]
        parts[index] = paused_state[index].detach().clone()
        arm_states[f"{name}_only"] = tuple(parts)  # type: ignore[assignment]

    arms: dict[str, dict[str, Any]] = {}
    for name, arm_state in arm_states.items():
        arm_model = copy.deepcopy(paused_model)
        start_index = int(arm_model._step_index)
        candidate = _surfaces(
            arm_model,
            clone_state(arm_state),
            config,
            resume_steps,
            pause_steps + 1,
        )
        row: dict[str, Any] = _summarize(reference, candidate)
        row.update(
            {
                "step_index_start": start_index,
                "update_mode": config.update_mode,
                "surface_reconstruction_error": (
                    float(torch.max(torch.abs(paused_model.state_vector(arm_state) - paused_surface)).item())
                    if name == "body_surface"
                    else 0.0
                ),
                "max_norm_mismatch": random_mismatch if name == "random_norm_matched" else 0.0,
            }
        )
        if name == "reachable_surface_match":
            row.update(
                {
                    "source_step": reachable_step,
                    "surface_match_error": reachable_error,
                    "surface_match_tolerance": 1e-7,
                    "surface_match_within_tolerance": reachable_error <= 1e-7,
                }
            )
        arms[name] = row

    return {
        "config": {
            "seed": config.seed,
            "hidden_size": config.hidden_size,
            "update_mode": config.update_mode,
            "post_step_clamp": config.post_step_clamp,
        },
        "pause_steps": pause_steps,
        "resume_steps": resume_steps,
        "control_steps": [pause_steps + 1, pause_steps + resume_steps],
        "reference_final_surface": reference[-1].tolist(),
        "arms": arms,
    }
