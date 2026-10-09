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


def first_visibility_tick(surface_separations: Sequence[float], *, threshold: float) -> int | None:
    """Return first sampled index crossing a finite positive visibility threshold."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    return next((index for index, separation in enumerate(surface_separations) if separation >= threshold), None)


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
    paired_separations: list[float] = []
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
        base_step = baseline.advance(frame.observation, frame.acknowledgement)
        plus_step = plus.advance(frame.observation, frame.acknowledgement)
        minus_step = minus.advance(frame.observation, frame.acknowledgement)
        base_surface = torch.tensor(base_step.surface)
        plus_surface = torch.tensor(plus_step.surface)
        minus_surface = torch.tensor(minus_step.surface)
        plus_separation = float((plus_surface - base_surface).norm())
        minus_separation = float((minus_surface - base_surface).norm())
        paired_separation = float((plus_surface - minus_surface).norm() / 2)
        paired_separations.append(paired_separation)
        trajectory.append(
            {
                "tick": tick,
                "input_replay": True,
                "plus_surface_separation": plus_separation,
                "minus_surface_separation": minus_separation,
                "paired_surface_separation": paired_separation,
                "pair_asymmetry": float((plus_surface + minus_surface - 2 * base_surface).norm()),
            }
        )
    visible_index = first_visibility_tick(paired_separations, threshold=visibility_threshold)
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
        "visibility_tick": visible_index,
        "visibility_delay": None if visible_index is None else visible_index - intervention_tick,
        "trajectory": trajectory,
    }
