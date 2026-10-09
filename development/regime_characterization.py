"""Directional tangent rates under fixed drive; Phase 0 instrumentation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter
from development.demian_v1_gate_state import DemianV1GateState, V1_CHANNELS


class StateMap:
    """Pure tensor transition/readout for the current constant-parameter adapters."""

    def __init__(self, adapter: ClosedLoopModelAdapter, *, recurrent_multiplier: float = 1.0):
        if adapter.architecture == "mlp":
            raise ValueError("memoryless MLP has no recurrent tangent state")
        self.adapter = adapter
        adapter.model.double().requires_grad_(False)
        adapter.action_head.double().requires_grad_(False)
        adapter.state = tuple(value.detach().double() for value in adapter.state)
        self.shapes = [value.shape for value in adapter.state]
        self.sizes = [value.numel() for value in adapter.state]
        self.channels = list(V1_CHANNELS) if len(adapter.state) == 6 else ["hidden"]
        self.initial_state = torch.cat([value.reshape(-1) for value in adapter.state])
        if recurrent_multiplier != 1.0:
            if adapter.architecture != "rnn":
                raise ValueError("recurrent multiplier is defined only for vanilla RNN weight_hh")
            with torch.no_grad():
                adapter.model.weight_hh.mul_(recurrent_multiplier)
        if isinstance(adapter.model, DemianV1GateState):
            if adapter.model.binding_start_step != 1 or adapter.model.gate_frozen:
                raise ValueError("time-dependent or frozen-gate runtime needs an augmented state map")

    def components(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(value.reshape(shape) for value, shape in zip(state.split(self.sizes), self.shapes))

    def transition(self, state: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        model = self.adapter.model
        if self.adapter.architecture in {"rnn", "gru"}:
            return model(encoded, state.reshape(1, -1)).reshape(-1)
        # step() writes only diagnostics/index for these configurations; preserve them.
        metadata = (model._step_index, model._step_aux, model._route_trace)
        try:
            components = self.components(state)
            injected = (components[0] + encoded, *components[1:])
            return torch.cat([value.reshape(-1) for value in model.step(injected)])
        finally:
            model._step_index, model._step_aux, model._route_trace = metadata

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        if self.adapter.architecture in {"rnn", "gru"}:
            return state
        return self.adapter.model.state_vector(self.components(state)).reshape(-1)

    def operating_point(self, state: torch.Tensor, encoded: torch.Tensor, floor: float) -> dict:
        """Measure slopes at declared gate/candidate operations (not state magnitudes)."""
        slopes: dict[str, float] = {}
        handles = []

        def record(name, function):
            def hook(module, inputs, output):
                with torch.no_grad():
                    activated = function(output)
                    derivative = activated * (1 - activated) if function is torch.sigmoid else 1 - activated.square()
                    slopes[name] = float((derivative.abs() < floor).double().mean())

            return hook

        if isinstance(self.adapter.model, DemianV1GateState):
            for channel in self.channels:
                for suffix, function in (("gate", torch.sigmoid), ("mix", torch.tanh)):
                    name = f"{channel}_{suffix}"
                    handles.append(getattr(self.adapter.model, name).register_forward_hook(record(name, function)))
            handles.append(self.adapter.model.gate_input.register_forward_hook(record("gate_context", torch.tanh)))
        try:
            with torch.no_grad():
                next_state = self.transition(state, encoded)
        finally:
            for handle in handles:
                handle.remove()
        if self.adapter.architecture == "rnn":
            slopes["rnn_tanh"] = float(((1 - next_state.square()).abs() < floor).double().mean())
        # GRU fused internal gates and cross-route nonlinearities need separate instrumentation.
        return {
            "component_norms": [float(value.norm()) for value in self.components(next_state)],
            "component_increment_norms": [
                float((a - b).norm()) for a, b in zip(self.components(next_state), self.components(state))
            ],
            "operation_saturation": slopes,
            "saturation_coverage": "gate_candidate_subset" if slopes else "unavailable_fused_GRU",
        }


def _direction(state: torch.Tensor, seed: int) -> torch.Tensor:
    vector = torch.randn(state.shape, dtype=state.dtype, generator=torch.Generator().manual_seed(seed))
    return vector / vector.norm()


def check_jvp(
    transition: Callable,
    state: torch.Tensor,
    encoded: torch.Tensor,
    *,
    direction_seed: int,
    epsilons: Sequence[float] = (1e-4, 1e-5, 1e-6),
    tolerance: float = 1e-5,
) -> dict:
    vector = _direction(state, direction_seed)
    _, tangent = torch.autograd.functional.jvp(lambda value: transition(value, encoded), state, vector)
    checks = []
    for epsilon in epsilons:
        with torch.no_grad():
            finite_difference = (
                transition(state + epsilon * vector, encoded) - transition(state - epsilon * vector, encoded)
            ) / (2 * epsilon)
        error = float((finite_difference - tangent).norm() / tangent.norm().clamp_min(1e-15))
        checks.append({"epsilon": epsilon, "relative_error": error})
    return {
        "passed": all(math.isfinite(item["relative_error"]) and item["relative_error"] <= tolerance for item in checks),
        "checks": checks,
    }


def tangent_window(
    transition: Callable,
    readout: Callable,
    initial_state: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    *,
    direction_seed: int,
    visibility_floor: float = 1e-12,
    sample_every: int = 32,
    operating_point: Callable | None = None,
) -> dict:
    """Benettin single-vector renormalization; reports directional finite-window rates."""
    if not inputs or sample_every < 1:
        raise ValueError("nonempty inputs and positive sample_every required")
    state = initial_state.detach().clone()
    vector = _direction(state, direction_seed)
    _, initial_visible = torch.autograd.functional.jvp(readout, state, vector)
    initial_visibility = float(initial_visible.norm())
    log_growth = 0.0
    samples = []
    for tick, encoded in enumerate(inputs):
        point = operating_point(state, encoded) if operating_point is not None else None
        next_state, tangent = torch.autograd.functional.jvp(lambda value: transition(value, encoded), state, vector)
        norm = float(tangent.norm())
        if not torch.isfinite(next_state).all() or not math.isfinite(norm):
            return {"status": "numerical_failure", "failure_tick": tick, "rate": None, "readout_rate": None}
        if norm == 0:
            return {"status": "tangent_collapse", "failure_tick": tick, "rate": None, "readout_rate": None}
        log_growth += math.log(norm)
        state = next_state.detach()
        vector = (tangent / norm).detach()
        if tick % sample_every == 0 or tick == len(inputs) - 1:
            samples.append(
                {
                    "tick": tick,
                    "state_norm": float(state.norm()),
                    "cumulative_log_growth": log_growth,
                    "operating_point": point,
                }
            )
    _, final_visible = torch.autograd.functional.jvp(readout, state, vector)
    final_visibility = float(final_visible.norm())
    visible = initial_visibility > visibility_floor and final_visibility > visibility_floor
    return {
        "status": "ok",
        "rate": log_growth / len(inputs),
        "readout_rate": (log_growth + math.log(final_visibility / initial_visibility)) / len(inputs)
        if visible
        else None,
        "readout_status": "ok"
        if visible
        else (
            "initial_visibility_below_floor"
            if initial_visibility <= visibility_floor
            else "final_visibility_below_floor"
        ),
        "initial_visibility": initial_visibility,
        "final_visibility": final_visibility,
        "log_growth": log_growth,
        "measured_ticks": len(inputs),
        "direction_seed": direction_seed,
        "samples": samples,
    }
