"""Fixed-input recovery controls for closed-loop comparator dynamics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from development.closed_loop_comparators import (
    Architecture,
    ClosedLoopModelAdapter,
    ComparatorRunner,
    apply_relative_pulse,
)


def run_fixed_input_recovery(
    *,
    architecture: Architecture,
    seed: int,
    hidden_size: int,
    steps: int,
    delay_steps: int,
    intervention_tick: int,
    amplitude: float,
    direction_seed: int | None = None,
) -> dict[str, object]:
    """Compare sham and pulsed continuations under the same recorded inputs."""
    if not 0 <= intervention_tick < steps:
        raise ValueError("intervention_tick must fall within steps")
    source = ComparatorRunner(
        architecture=architecture,
        seed=seed,
        hidden_size=hidden_size,
        cue_symbol=seed % 3,
        delay_steps=delay_steps,
    )
    frames = [source.step().input_frame for _ in range(steps)]
    reference = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
    perturbed = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
    trajectory: list[dict[str, object]] = []
    initial_separation = 0.0
    for tick, frame in enumerate(frames):
        if tick == intervention_tick:
            before = _flatten_state(perturbed)
            apply_relative_pulse(
                perturbed,
                amplitude=amplitude,
                seed=seed if direction_seed is None else direction_seed,
                tick=tick,
            )
            initial_separation = _l2_difference(before, _flatten_state(perturbed))
        reference_step = reference.advance(frame.observation, frame.acknowledgement)
        perturbed_step = perturbed.advance(frame.observation, frame.acknowledgement)
        reference_state = _flatten_state(reference)
        perturbed_state = _flatten_state(perturbed)
        separation = _l2_difference(reference_state, perturbed_state)
        trajectory.append(
            {
                "tick": tick,
                "input_replay": True,
                "state_separation": separation,
                "state_cosine": _cosine(reference_state, perturbed_state),
                "surface_separation": _l2_difference(reference_step.surface, perturbed_step.surface),
                "component_separation": _component_separation(reference, perturbed),
            }
        )
    gains = [point["state_separation"] / initial_separation if initial_separation else 0.0 for point in trajectory]
    peak_gain = max(gains, default=0.0)
    peak_tick = trajectory[gains.index(peak_gain)]["tick"] if gains else None
    final = trajectory[-1]
    return {
        "schema_version": 1,
        "experiment": "fixed-input-recovery",
        "architecture": architecture,
        "seed": seed,
        "config": {
            "hidden_size": hidden_size,
            "steps": steps,
            "delay_steps": delay_steps,
            "intervention_tick": intervention_tick,
            "amplitude": amplitude,
            "direction_seed": seed if direction_seed is None else direction_seed,
        },
        "trajectory": trajectory,
        "metrics": {
            "initial_separation": initial_separation,
            "peak_gain": peak_gain,
            "peak_tick": peak_tick,
            "final_gain": gains[-1] if gains else 0.0,
            "final_state_separation": final["state_separation"],
            "final_state_cosine": final["state_cosine"],
            "final_surface_separation": final["surface_separation"],
        },
    }


def run_stability_panel(
    *,
    architectures: Sequence[Architecture],
    seeds: Sequence[int],
    hidden_sizes: Mapping[Architecture, int],
    steps: int,
    delay_steps: int,
    amplitudes: Sequence[float],
    intervention_ticks: Sequence[int],
) -> dict[str, object]:
    """Run preregistered amplitude/time recovery cells under fixed input replay."""
    return {
        "schema_version": 1,
        "experiment": "fixed-input-stability-panel",
        "input_semantics": "Each baseline frame, including acknowledgement, is replayed identically in paired arms.",
        "runs": [
            run_fixed_input_recovery(
                architecture=architecture,
                seed=seed,
                hidden_size=hidden_sizes[architecture],
                steps=steps,
                delay_steps=delay_steps,
                intervention_tick=intervention_tick,
                amplitude=amplitude,
            )
            for architecture in architectures
            for seed in seeds
            for amplitude in amplitudes
            for intervention_tick in intervention_ticks
        ],
    }


def _flatten_state(adapter: ClosedLoopModelAdapter) -> list[float]:
    return [value for component in adapter.state for value in component.detach().reshape(-1).cpu().tolist()]


def _component_separation(reference: ClosedLoopModelAdapter, perturbed: ClosedLoopModelAdapter) -> list[float]:
    return [
        math.sqrt(sum((left - right) ** 2 for left, right in zip(a.reshape(-1).tolist(), b.reshape(-1).tolist())))
        for a, b in zip(reference.state, perturbed.state)
    ]


def _l2_difference(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 1.0
