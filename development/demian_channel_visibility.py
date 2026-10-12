"""Paired channel-local visibility measurements for the six-channel Demian state."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter
from development.closed_loop_demian import InputFrame
from development.demian_v1_gate_state import V1_CHANNELS


def channel_pulse(
    state: tuple[torch.Tensor, ...], *, channel: str, amplitude: float, direction_seed: int, sign: int
) -> tuple[torch.Tensor, ...]:
    """Apply one signed relative-norm pulse to exactly one state channel."""
    if channel not in V1_CHANNELS:
        raise ValueError(f"unknown Demian channel: {channel}")
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or 1")
    index = V1_CHANNELS.index(channel)
    source = state[index]
    norm = source.norm()
    if norm == 0:
        raise ValueError(f"cannot pulse zero-norm channel: {channel}")
    generator = torch.Generator(device=source.device).manual_seed(direction_seed)
    direction = torch.randn(source.shape, generator=generator, dtype=source.dtype, device=source.device)
    direction = direction / direction.norm()
    result = list(state)
    result[index] = source + sign * amplitude * norm * direction
    return tuple(result)


def full_state_pulse(
    state: tuple[torch.Tensor, ...], *, amplitude: float, reference_norm: torch.Tensor, direction_seed: int, sign: int
) -> tuple[torch.Tensor, ...]:
    """Apply a full-state pulse with norm matched to a declared reference."""
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or 1")
    if reference_norm.ndim != 0 or reference_norm <= 0:
        raise ValueError("reference norm must be a positive scalar tensor")
    generator = torch.Generator(device=state[0].device).manual_seed(direction_seed)
    width = sum(component.numel() for component in state)
    direction = torch.randn(width, generator=generator, dtype=state[0].dtype, device=state[0].device)
    direction = direction / direction.norm()
    offset = sign * amplitude * reference_norm * direction
    result = []
    cursor = 0
    for component in state:
        component_width = component.numel()
        result.append(component + offset[cursor : cursor + component_width].reshape_as(component))
        cursor += component_width
    return tuple(result)


def first_visibility_tick(surface_separations: Sequence[float], *, threshold: float) -> int | None:
    """Return first sampled index crossing a finite positive visibility threshold."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    return next((index for index, separation in enumerate(surface_separations) if separation >= threshold), None)


def _state_vector(adapter: ClosedLoopModelAdapter) -> torch.Tensor:
    return torch.cat([component.detach().reshape(-1) for component in adapter.state])


def _surface_vector(adapter: ClosedLoopModelAdapter) -> torch.Tensor:
    return adapter.model.state_vector(adapter.state).detach().reshape(-1)


def _measurement(
    *,
    tick: int,
    transitions_after_pulse: int | None,
    baseline: ClosedLoopModelAdapter,
    plus: ClosedLoopModelAdapter,
    minus: ClosedLoopModelAdapter,
) -> dict[str, object]:
    base_surface, plus_surface, minus_surface = (_surface_vector(adapter) for adapter in (baseline, plus, minus))
    base_state, plus_state, minus_state = (_state_vector(adapter) for adapter in (baseline, plus, minus))
    return {
        "tick": tick,
        "transitions_after_pulse": transitions_after_pulse,
        "input_replay": True,
        "plus_surface_separation": float((plus_surface - base_surface).norm()),
        "minus_surface_separation": float((minus_surface - base_surface).norm()),
        "paired_surface_separation": float((plus_surface - minus_surface).norm() / 2),
        "pair_asymmetry": float((plus_surface + minus_surface - 2 * base_surface).norm()),
        "plus_state_separation": float((plus_state - base_state).norm()),
        "minus_state_separation": float((minus_state - base_state).norm()),
        "paired_state_separation": float((plus_state - minus_state).norm() / 2),
    }


def run_channel_visibility_cell(
    *,
    seed: int,
    hidden_size: int,
    frames: Sequence[InputFrame],
    channel: str,
    intervention_tick: int,
    amplitude: float,
    direction_seed: int,
    visibility_threshold: float,
) -> dict[str, object]:
    """Replay one input stream through baseline and equal/opposite channel pulses."""
    if not frames:
        raise ValueError("at least one recorded input frame is required")
    if not 0 <= intervention_tick < len(frames):
        raise ValueError("intervention_tick must index frames")
    baseline = ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size)
    plus = ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size)
    minus = ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size)
    trajectory: list[dict[str, object]] = []
    initial_channel_pulse_norm = None
    for tick, frame in enumerate(frames):
        if tick == intervention_tick:
            plus.state = channel_pulse(
                plus.state, channel=channel, amplitude=amplitude, direction_seed=direction_seed, sign=1
            )
            minus.state = channel_pulse(
                minus.state, channel=channel, amplitude=amplitude, direction_seed=direction_seed, sign=-1
            )
            index = V1_CHANNELS.index(channel)
            initial_channel_pulse_norm = float((plus.state[index] - baseline.state[index]).norm())
            trajectory.append(
                _measurement(
                    tick=tick,
                    transitions_after_pulse=0,
                    baseline=baseline,
                    plus=plus,
                    minus=minus,
                )
            )
        baseline.advance(frame.observation, frame.acknowledgement)
        plus.advance(frame.observation, frame.acknowledgement)
        minus.advance(frame.observation, frame.acknowledgement)
        trajectory.append(
            _measurement(
                tick=tick,
                transitions_after_pulse=tick - intervention_tick + 1 if tick >= intervention_tick else None,
                baseline=baseline,
                plus=plus,
                minus=minus,
            )
        )
    visible = next(
        (
            point
            for point in trajectory
            if point["transitions_after_pulse"] is not None
            and point["paired_surface_separation"] >= visibility_threshold
        ),
        None,
    )
    return {
        "schema_version": 1,
        "experiment": "demian-channel-visibility",
        "seed": seed,
        "channel": channel,
        "config": {
            "hidden_size": hidden_size,
            "intervention_tick": intervention_tick,
            "amplitude": amplitude,
            "direction_seed": direction_seed,
            "visibility_threshold": visibility_threshold,
        },
        "initial_channel_pulse_norm": initial_channel_pulse_norm,
        "visibility_tick": None if visible is None else visible["tick"],
        "visibility_transition": None if visible is None else visible["transitions_after_pulse"],
        "visibility_delay": None if visible is None else visible["transitions_after_pulse"],
        "trajectory": trajectory,
    }
