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
        """Operation slopes, including fused GRU gates and compound Demian routes."""
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("slope floor must be finite and positive")
        slopes: dict[str, float] = {}
        activated: dict[str, torch.Tensor] = {}
        handles = []

        def measure(name, value, sigmoid=False):
            derivative = value * (1 - value) if sigmoid else 1 - value.square()
            slopes[name] = float((derivative.abs() < floor).double().mean())
            activated[name] = value
            return value

        def record(name, sigmoid):
            def hook(module, inputs, output):
                with torch.no_grad():
                    measure(name, torch.sigmoid(output) if sigmoid else torch.tanh(output), sigmoid)

            return hook

        model = self.adapter.model
        if isinstance(model, DemianV1GateState):
            for channel in self.channels:
                for suffix in ("gate", "mix"):
                    name = f"{channel}_{suffix}"
                    handles.append(getattr(model, name).register_forward_hook(record(name, suffix == "gate")))
            for name in (
                "gate_input",
                "fast_to_message",
                "message_to_carrier",
                "fast_to_slow",
                "carrier_to_slow",
                "fast_slow_to_control",
                "control_readout",
                "message_to_fast",
                "carrier_to_fast",
                "message_readout",
                "carrier_readout",
                "gate_readout",
            ):
                handles.append(getattr(model, name).register_forward_hook(record(name, False)))
        reconstruction_error = 0.0
        try:
            with torch.no_grad():
                next_state = self.transition(state, encoded)
                if self.adapter.architecture == "gru":
                    hidden = state.reshape(1, -1)
                    input_terms = torch.nn.functional.linear(encoded, model.weight_ih, model.bias_ih).chunk(3, -1)
                    hidden_terms = torch.nn.functional.linear(hidden, model.weight_hh, model.bias_hh).chunk(3, -1)
                    reset = measure("gru_reset", torch.sigmoid(input_terms[0] + hidden_terms[0]), True)
                    update = measure("gru_update", torch.sigmoid(input_terms[1] + hidden_terms[1]), True)
                    candidate = measure("gru_candidate", torch.tanh(input_terms[2] + reset * hidden_terms[2]))
                    reconstructed = ((1 - update) * candidate + update * hidden).reshape(-1)
                    reconstruction_error = float((reconstructed - next_state).abs().max())
                elif isinstance(model, DemianV1GateState):
                    new_gate = self.components(next_state)[-1]
                    pressure = measure("gate_pressure", torch.sigmoid(new_gate), True)
                    surface_mod = 1 + model.gate_to_surface_scale * (2 * pressure - 1)
                    integrated = measure(
                        "fast_integrated",
                        torch.tanh(
                            activated["fast_mix"]
                            + model.control_to_fast_scale * activated["control_readout"]
                            + surface_mod * model.message_to_fast_scale * activated["message_to_fast"]
                            + surface_mod * model.carrier_to_fast_scale * activated["carrier_to_fast"]
                        ),
                    )
                    fast = self.components(state)[0] + encoded
                    reconstructed = (1 - activated["fast_gate"]) * fast + activated[
                        "fast_gate"
                    ] * model.state_gain * integrated
                    reconstruction_error = float((reconstructed - self.components(next_state)[0]).abs().max())
                    self.readout(next_state)  # Capture the three exposed-readout nonlinearities.
                else:
                    measure("rnn_tanh", next_state)
        finally:
            for handle in handles:
                handle.remove()
        return {
            "component_norms": [float(value.norm()) for value in self.components(next_state)],
            "component_increment_norms": [
                float((a - b).norm()) for a, b in zip(self.components(next_state), self.components(state))
            ],
            "operation_saturation": slopes,
            "saturation_coverage": "complete_declared_operations",
            "transition_reconstruction_max_error": reconstruction_error,
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
        sampled = tick % sample_every == 0 or tick == len(inputs) - 1
        point = operating_point(state, encoded) if operating_point is not None and sampled else None
        next_state, tangent = torch.autograd.functional.jvp(lambda value: transition(value, encoded), state, vector)
        norm = float(tangent.norm())
        if not torch.isfinite(next_state).all() or not math.isfinite(norm):
            return {"status": "numerical_failure", "failure_tick": tick, "rate": None, "readout_rate": None}
        if norm == 0:
            return {"status": "tangent_collapse", "failure_tick": tick, "rate": None, "readout_rate": None}
        log_growth += math.log(norm)
        state = next_state.detach()
        vector = (tangent / norm).detach()
        if sampled:
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
