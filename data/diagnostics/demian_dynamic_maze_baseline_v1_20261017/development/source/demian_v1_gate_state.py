#!/usr/bin/env python3
"""Demian v1 gate-state substrate prototype.

This module is a focused architectural prototype, not a replacement for the
existing v9 five-channel evolution workbench. It makes the replicated
gate-state mechanism explicit by adding a first-class internal gate channel and
using it to modulate message/carrier routes directly.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from development.substrates.legacy import DemianNativeV9Substrate

V1_CHANNELS = ("fast", "slow", "control", "message", "carrier", "gate")
V1State = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
V1Ablation = Literal[
    "none",
    "gate_disabled",
    "gate_frozen",
    "message_disabled",
    "carrier_disabled",
    "slow_disabled",
    "control_disabled",
]


@dataclass(frozen=True)
class V1RouteDefinition:
    id: str
    sources: tuple[str, ...]
    target: str
    operation: Literal["additive", "multiplicative"]
    injection_point: Literal[
        "target_state",
        "gate_context_preactivation",
        "fast_preactivation",
        "multiplicative_modulator",
    ]


V1_ROUTE_CATALOG = (
    V1RouteDefinition("gate_persistence", ("gate",), "gate", "additive", "target_state"),
    V1RouteDefinition("gate_integrated_update", ("gate", "fast", "slow", "control", "message", "carrier"), "gate", "additive", "target_state"),
    V1RouteDefinition("fast_to_gate", ("fast",), "gate", "additive", "gate_context_preactivation"),
    V1RouteDefinition("slow_to_gate", ("slow",), "gate", "additive", "gate_context_preactivation"),
    V1RouteDefinition("control_to_gate", ("control",), "gate", "additive", "gate_context_preactivation"),
    V1RouteDefinition("message_to_gate", ("message",), "gate", "additive", "gate_context_preactivation"),
    V1RouteDefinition("carrier_to_gate", ("carrier",), "gate", "additive", "gate_context_preactivation"),
    V1RouteDefinition("bias_to_gate", ("bias",), "gate", "additive", "gate_context_preactivation"),
    V1RouteDefinition("message_persistence", ("message",), "message", "additive", "target_state"),
    V1RouteDefinition("message_candidate", ("message",), "message", "additive", "target_state"),
    V1RouteDefinition("fast_to_message", ("fast",), "message", "additive", "target_state"),
    V1RouteDefinition("carrier_persistence", ("carrier",), "carrier", "additive", "target_state"),
    V1RouteDefinition("carrier_candidate", ("carrier",), "carrier", "additive", "target_state"),
    V1RouteDefinition("message_to_carrier", ("message",), "carrier", "additive", "target_state"),
    V1RouteDefinition("gate_modulates_message_to_carrier", ("gate",), "carrier", "multiplicative", "multiplicative_modulator"),
    V1RouteDefinition("slow_persistence", ("slow",), "slow", "additive", "target_state"),
    V1RouteDefinition("slow_candidate", ("slow",), "slow", "additive", "target_state"),
    V1RouteDefinition("fast_to_slow", ("fast",), "slow", "additive", "target_state"),
    V1RouteDefinition("carrier_to_slow", ("carrier",), "slow", "additive", "target_state"),
    V1RouteDefinition("gate_modulates_carrier_to_slow", ("gate",), "slow", "multiplicative", "multiplicative_modulator"),
    V1RouteDefinition("control_persistence", ("control",), "control", "additive", "target_state"),
    V1RouteDefinition("control_candidate", ("control",), "control", "additive", "target_state"),
    V1RouteDefinition("fast_slow_to_control", ("fast", "slow"), "control", "additive", "target_state"),
    V1RouteDefinition("fast_retention", ("fast",), "fast", "additive", "target_state"),
    V1RouteDefinition("fast_integrated_update", ("fast", "control", "message", "carrier", "gate"), "fast", "additive", "target_state"),
    V1RouteDefinition("fast_candidate", ("fast",), "fast", "additive", "fast_preactivation"),
    V1RouteDefinition("control_to_fast", ("control",), "fast", "additive", "fast_preactivation"),
    V1RouteDefinition("message_to_fast", ("message",), "fast", "additive", "fast_preactivation"),
    V1RouteDefinition("carrier_to_fast", ("carrier",), "fast", "additive", "fast_preactivation"),
    V1RouteDefinition("gate_modulates_surface_routes", ("gate",), "fast", "multiplicative", "multiplicative_modulator"),
)


@dataclass(frozen=True)
class V1RouteTrace:
    step: int
    catalog: dict[str, V1RouteDefinition]
    values: dict[str, torch.Tensor]


@dataclass(frozen=True)
class V1ResumeResult:
    """Summary of a short continuation comparison."""

    final_gap: float
    mean_step_gap: float


@dataclass(frozen=True)
class V1RuntimeSnapshot:
    state: V1State
    step_index: int
    frozen_gate: torch.Tensor | None


class DemianV1GateState(DemianNativeV9Substrate):
    """Six-channel substrate with explicit gate-state propagation.

    The pathway is:

    fast -> message -> carrier -> slow
    control/message/carrier/slow -> gate
    gate -> route modulation

    The gate is continuous and graded. It does not inject an additive release
    vector; it changes the strength of existing message, carrier, slow, and
    surface routes.
    """

    def __init__(
        self,
        hidden_size: int,
        message_decay: float = 0.90,
        carrier_decay: float = 0.97,
        gate_decay: float = 0.985,
        fast_to_message_scale: float = 0.28,
        message_to_carrier_scale: float = 0.42,
        carrier_to_slow_scale: float = 0.32,
        message_to_fast_scale: float = 0.12,
        carrier_to_fast_scale: float = 0.14,
        gate_to_message_carrier_scale: float = 0.85,
        gate_to_carrier_slow_scale: float = 0.75,
        gate_to_surface_scale: float = 0.55,
        gate_change_threshold: float = 0.015,
        initial_message_scale: float | None = None,
        initial_carrier_scale: float | None = None,
        initial_gate_scale: float | None = None,
        binding_start_step: int = 1,
        gate_disabled: bool = False,
        gate_frozen: bool = False,
        trace_routes: bool = False,
        **kwargs: float,
    ):
        super().__init__(hidden_size, **kwargs)
        self.message_decay = message_decay
        self.carrier_decay = carrier_decay
        self.gate_decay = gate_decay
        self.fast_to_message_scale = fast_to_message_scale
        self.message_to_carrier_scale = message_to_carrier_scale
        self.carrier_to_slow_scale = carrier_to_slow_scale
        self.message_to_fast_scale = message_to_fast_scale
        self.carrier_to_fast_scale = carrier_to_fast_scale
        self.gate_to_message_carrier_scale = gate_to_message_carrier_scale
        self.gate_to_carrier_slow_scale = gate_to_carrier_slow_scale
        self.gate_to_surface_scale = gate_to_surface_scale
        self.gate_change_threshold = gate_change_threshold
        self.initial_message_scale = self.init_scale if initial_message_scale is None else initial_message_scale
        self.initial_carrier_scale = self.init_scale if initial_carrier_scale is None else initial_carrier_scale
        self.initial_gate_scale = self.init_scale if initial_gate_scale is None else initial_gate_scale
        self.binding_start_step = binding_start_step
        self.gate_disabled = gate_disabled
        self.gate_frozen = gate_frozen
        if trace_routes and (gate_disabled or gate_frozen):
            raise ValueError("route tracing currently supports only the active gate update rule")
        self.trace_routes = trace_routes
        self._step_index = 0
        self._frozen_gate: torch.Tensor | None = None
        self._route_trace: V1RouteTrace | None = None

        self.message_gate = nn.Linear(hidden_size, hidden_size)
        self.message_mix = nn.Linear(hidden_size, hidden_size)
        self.carrier_gate = nn.Linear(hidden_size, hidden_size)
        self.carrier_mix = nn.Linear(hidden_size, hidden_size)
        self.gate_gate = nn.Linear(hidden_size, hidden_size)
        self.gate_mix = nn.Linear(hidden_size, hidden_size)
        self.gate_input = nn.Linear(hidden_size * 4 + self.control_dim, hidden_size)

        self.fast_to_message = nn.Linear(hidden_size, hidden_size, bias=False)
        self.message_to_carrier = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_to_slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.message_to_fast = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_to_fast = nn.Linear(hidden_size, hidden_size, bias=False)
        self.message_readout = nn.Linear(hidden_size, hidden_size, bias=False)
        self.carrier_readout = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_readout = nn.Linear(hidden_size, hidden_size, bias=False)

    def initial_state(self, batch_size: int, device: torch.device) -> V1State:
        fast, slow, control = super().initial_state(batch_size, device)
        self._step_index = 0
        self._frozen_gate = None
        self._route_trace = None
        message = torch.randn(batch_size, self.hidden_size, device=device) * self.initial_message_scale
        carrier = torch.randn(batch_size, self.hidden_size, device=device) * self.initial_carrier_scale
        gate = torch.randn(batch_size, self.hidden_size, device=device) * self.initial_gate_scale
        return fast, slow, control, message, carrier, gate

    def state_components(self, state: V1State) -> dict[str, torch.Tensor]:
        fast, slow, control, message, carrier, gate = state
        return {
            "fast": fast,
            "slow": slow,
            "control": control,
            "message": message,
            "carrier": carrier,
            "gate": gate,
        }

    def state_vector(self, state: V1State) -> torch.Tensor:
        fast, _, _, message, carrier, gate = state
        return (
            fast
            + self.message_to_fast_scale * torch.tanh(self.message_readout(message))
            + self.carrier_to_fast_scale * torch.tanh(self.carrier_readout(carrier))
            + 0.05 * torch.tanh(self.gate_readout(gate))
        )

    def inject_coupling_message(self, state: V1State, message: torch.Tensor, strength: float) -> V1State:
        fast, slow, control, msg_state, carrier, gate = state
        delta = strength * message
        if delta.shape != fast.shape:
            delta = delta.view(fast.shape)
        return fast + delta, slow, control, msg_state, carrier, gate

    def step(self, state: V1State) -> V1State:
        self._step_index += 1
        self._route_trace = None
        fast, slow, control, message, carrier, gate = state

        if self._step_index < self.binding_start_step:
            new_fast, new_slow, new_control = DemianNativeV9Substrate.step(self, (fast, slow, control))
            self._step_aux.update(self._inactive_gate_metrics(gate, gate))
            return new_fast, new_slow, new_control, message, carrier, gate

        raw_gate = self._next_gate(fast, slow, control, message, carrier, gate)
        if self.gate_frozen:
            if self._frozen_gate is None:
                self._frozen_gate = gate.detach().clone()
            new_gate = self._frozen_gate.to(device=gate.device, dtype=gate.dtype)
        elif self.gate_disabled:
            new_gate = torch.zeros_like(gate)
        else:
            new_gate = raw_gate

        gate_pressure = torch.sigmoid(new_gate)
        gate_centered = 2.0 * gate_pressure - 1.0
        message_carrier_mod = 1.0 + self.gate_to_message_carrier_scale * gate_centered
        carrier_slow_mod = 1.0 + self.gate_to_carrier_slow_scale * gate_centered
        surface_mod = 1.0 + self.gate_to_surface_scale * gate_centered

        message_write = torch.sigmoid(self.message_gate(message))
        message_candidate = torch.tanh(self.message_mix(message))
        fast_to_message = self.fast_to_message_scale * torch.tanh(self.fast_to_message(fast))
        new_message = self.message_decay * message + message_write * (message_candidate + fast_to_message)

        carrier_write = torch.sigmoid(self.carrier_gate(carrier))
        carrier_candidate = torch.tanh(self.carrier_mix(carrier))
        message_to_carrier = self.message_to_carrier_scale * torch.tanh(
            self.message_to_carrier(new_message)
        )
        gated_message_to_carrier = message_carrier_mod * message_to_carrier
        new_carrier = self.carrier_decay * carrier + carrier_write * (
            carrier_candidate + gated_message_to_carrier
        )

        slow_gate = torch.sigmoid(self.slow_gate(slow))
        slow_candidate = torch.tanh(self.slow_mix(slow))
        fast_to_slow_bias = self.slow_readout_scale * torch.tanh(self.fast_to_slow(fast))
        carrier_to_slow = self.carrier_to_slow_scale * torch.tanh(self.carrier_to_slow(new_carrier))
        gated_carrier_to_slow = carrier_slow_mod * carrier_to_slow
        new_slow = self.slow_decay * slow + slow_gate * (
            slow_candidate + fast_to_slow_bias + gated_carrier_to_slow
        )

        control_gate = torch.sigmoid(self.control_gate(control))
        control_candidate = torch.tanh(self.control_mix(control))
        fast_slow_bias = torch.tanh(self.fast_slow_to_control(torch.cat([fast, slow], dim=-1)))
        new_control = self.control_decay * control + control_gate * (control_candidate + fast_slow_bias)

        fast_gate = torch.sigmoid(self.fast_gate(fast))
        fast_candidate = torch.tanh(self.fast_mix(fast))
        control_bias = self.control_to_fast_scale * torch.tanh(self.control_readout(new_control))
        message_fast_bias = surface_mod * self.message_to_fast_scale * torch.tanh(
            self.message_to_fast(new_message)
        )
        carrier_fast_bias = surface_mod * self.carrier_to_fast_scale * torch.tanh(
            self.carrier_to_fast(new_carrier)
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(
            fast_candidate + control_bias + message_fast_bias + carrier_fast_bias
        )

        if self.trace_routes:
            self._route_trace = self._capture_route_trace(
                fast=fast,
                slow=slow,
                control=control,
                message=message,
                carrier=carrier,
                gate=gate,
                new_gate=new_gate,
                gate_pressure=gate_pressure,
                message_carrier_mod=message_carrier_mod,
                carrier_slow_mod=carrier_slow_mod,
                surface_mod=surface_mod,
                message_write=message_write,
                message_candidate=message_candidate,
                fast_to_message=fast_to_message,
                carrier_write=carrier_write,
                carrier_candidate=carrier_candidate,
                gated_message_to_carrier=gated_message_to_carrier,
                slow_gate=slow_gate,
                slow_candidate=slow_candidate,
                fast_to_slow_bias=fast_to_slow_bias,
                gated_carrier_to_slow=gated_carrier_to_slow,
                control_gate=control_gate,
                control_candidate=control_candidate,
                fast_slow_bias=fast_slow_bias,
                fast_gate=fast_gate,
                fast_candidate=fast_candidate,
                control_bias=control_bias,
                message_fast_bias=message_fast_bias,
                carrier_fast_bias=carrier_fast_bias,
            )

        gate_delta = new_gate - gate
        gate_change = (torch.abs(gate_delta) > self.gate_change_threshold).float()
        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "control_write_mean": float(control_gate.mean().item()),
            "message_write_mean": float(message_write.mean().item()),
            "carrier_write_mean": float(carrier_write.mean().item()),
            "gate_write_mean": float(torch.sigmoid(self.gate_gate(gate)).mean().item()),
            "fast_to_slow_bias_norm": float(torch.norm(fast_to_slow_bias).item()),
            "fast_slow_bias_norm": float(torch.norm(fast_slow_bias).item()),
            "control_bias_norm": float(torch.norm(control_bias).item()),
            "fast_to_message_norm": float(torch.norm(fast_to_message).item()),
            "message_to_carrier_norm": float(torch.norm(gated_message_to_carrier).item()),
            "carrier_to_slow_norm": float(torch.norm(gated_carrier_to_slow).item()),
            "message_to_fast_norm": float(torch.norm(message_fast_bias).item()),
            "carrier_to_fast_norm": float(torch.norm(carrier_fast_bias).item()),
            "gate_state_norm": float(torch.norm(new_gate).item()),
            "gate_state_delta": float(torch.norm(gate_delta).item()),
            "gate_change_duty": float(gate_change.mean().item()),
            "gate_pressure_mean": float(gate_pressure.mean().item()),
            "gate_modulation_mean": float(surface_mod.mean().item()),
            "gate_to_message_carrier_scale": float(message_carrier_mod.mean().item()),
            "gate_to_carrier_slow_scale": float(carrier_slow_mod.mean().item()),
            "gate_to_surface_scale": float(surface_mod.mean().item()),
            "gate_disabled": float(self.gate_disabled),
            "gate_frozen": float(self.gate_frozen),
        }
        return new_fast, new_slow, new_control, new_message, new_carrier, new_gate

    def route_trace(self) -> V1RouteTrace | None:
        return self._route_trace

    def _capture_route_trace(
        self,
        *,
        fast: torch.Tensor,
        slow: torch.Tensor,
        control: torch.Tensor,
        message: torch.Tensor,
        carrier: torch.Tensor,
        gate: torch.Tensor,
        new_gate: torch.Tensor,
        gate_pressure: torch.Tensor,
        message_carrier_mod: torch.Tensor,
        carrier_slow_mod: torch.Tensor,
        surface_mod: torch.Tensor,
        message_write: torch.Tensor,
        message_candidate: torch.Tensor,
        fast_to_message: torch.Tensor,
        carrier_write: torch.Tensor,
        carrier_candidate: torch.Tensor,
        gated_message_to_carrier: torch.Tensor,
        slow_gate: torch.Tensor,
        slow_candidate: torch.Tensor,
        fast_to_slow_bias: torch.Tensor,
        gated_carrier_to_slow: torch.Tensor,
        control_gate: torch.Tensor,
        control_candidate: torch.Tensor,
        fast_slow_bias: torch.Tensor,
        fast_gate: torch.Tensor,
        fast_candidate: torch.Tensor,
        control_bias: torch.Tensor,
        message_fast_bias: torch.Tensor,
        carrier_fast_bias: torch.Tensor,
    ) -> V1RouteTrace:
        widths = (
            self.hidden_size,
            self.hidden_size,
            self.control_dim,
            self.hidden_size,
            self.hidden_size,
        )
        sources = (fast, slow, control, message, carrier)
        source_names = ("fast", "slow", "control", "message", "carrier")
        offset = 0
        gate_context_terms: dict[str, torch.Tensor] = {}
        for name, source, width in zip(source_names, sources, widths, strict=True):
            weight = self.gate_input.weight[:, offset : offset + width]
            gate_context_terms[f"{name}_to_gate"] = torch.nn.functional.linear(source, weight)
            offset += width
        gate_context_terms["bias_to_gate"] = self.gate_input.bias.unsqueeze(0).expand_as(gate)

        gate_write = torch.sigmoid(self.gate_gate(gate))
        gate_candidate = torch.tanh(self.gate_mix(gate))
        values = {
            "gate_persistence": self.gate_decay * gate,
            "gate_integrated_update": new_gate - self.gate_decay * gate,
            **gate_context_terms,
            "message_persistence": self.message_decay * message,
            "message_candidate": message_write * message_candidate,
            "fast_to_message": message_write * fast_to_message,
            "carrier_persistence": self.carrier_decay * carrier,
            "carrier_candidate": carrier_write * carrier_candidate,
            "message_to_carrier": carrier_write * gated_message_to_carrier,
            "gate_modulates_message_to_carrier": message_carrier_mod,
            "slow_persistence": self.slow_decay * slow,
            "slow_candidate": slow_gate * slow_candidate,
            "fast_to_slow": slow_gate * fast_to_slow_bias,
            "carrier_to_slow": slow_gate * gated_carrier_to_slow,
            "gate_modulates_carrier_to_slow": carrier_slow_mod,
            "control_persistence": self.control_decay * control,
            "control_candidate": control_gate * control_candidate,
            "fast_slow_to_control": control_gate * fast_slow_bias,
            "fast_retention": (1.0 - fast_gate) * fast,
            "fast_integrated_update": fast_gate
            * self.state_gain
            * torch.tanh(fast_candidate + control_bias + message_fast_bias + carrier_fast_bias),
            "fast_candidate": fast_candidate,
            "control_to_fast": control_bias,
            "message_to_fast": message_fast_bias,
            "carrier_to_fast": carrier_fast_bias,
            "gate_modulates_surface_routes": surface_mod,
        }
        catalog = {route.id: route for route in V1_ROUTE_CATALOG}
        return V1RouteTrace(
            step=self._step_index,
            catalog=catalog,
            values={name: value.detach().clone() for name, value in values.items()},
        )

    def _next_gate(
        self,
        fast: torch.Tensor,
        slow: torch.Tensor,
        control: torch.Tensor,
        message: torch.Tensor,
        carrier: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        gate_write = torch.sigmoid(self.gate_gate(gate))
        gate_candidate = torch.tanh(self.gate_mix(gate))
        gate_context = torch.tanh(
            self.gate_input(torch.cat([fast, slow, control, message, carrier], dim=-1))
        )
        return self.gate_decay * gate + gate_write * (gate_candidate + gate_context)

    def _inactive_gate_metrics(self, old_gate: torch.Tensor, new_gate: torch.Tensor) -> dict[str, float]:
        gate_delta = new_gate - old_gate
        gate_pressure = torch.sigmoid(new_gate)
        return {
            "message_write_mean": 0.0,
            "carrier_write_mean": 0.0,
            "gate_write_mean": 0.0,
            "fast_to_message_norm": 0.0,
            "message_to_carrier_norm": 0.0,
            "carrier_to_slow_norm": 0.0,
            "message_to_fast_norm": 0.0,
            "carrier_to_fast_norm": 0.0,
            "gate_state_norm": float(torch.norm(new_gate).item()),
            "gate_state_delta": float(torch.norm(gate_delta).item()),
            "gate_change_duty": 0.0,
            "gate_pressure_mean": float(gate_pressure.mean().item()),
            "gate_modulation_mean": 1.0,
            "gate_to_message_carrier_scale": 1.0,
            "gate_to_carrier_slow_scale": 1.0,
            "gate_to_surface_scale": 1.0,
            "gate_disabled": float(self.gate_disabled),
            "gate_frozen": float(self.gate_frozen),
        }


def clone_state(state: V1State) -> V1State:
    """Clone all internal channels for capsule-style continuation."""

    return tuple(component.detach().clone() for component in state)  # type: ignore[return-value]


def match_v1_surface(
    model: DemianV1GateState,
    state: V1State,
    target_surface: torch.Tensor,
) -> V1State:
    """Set fast so the projected surface matches while other channels remain fixed."""

    _, slow, control, message, carrier, gate = state
    contribution = (
        model.message_to_fast_scale * torch.tanh(model.message_readout(message))
        + model.carrier_to_fast_scale * torch.tanh(model.carrier_readout(carrier))
        + 0.05 * torch.tanh(model.gate_readout(gate))
    )
    fast = target_surface.reshape_as(state[0]) - contribution
    return fast, slow, control, message, carrier, gate


def capture_v1_runtime(model: DemianV1GateState, state: V1State) -> V1RuntimeSnapshot:
    """Capture recurrent state plus model-local continuation metadata."""

    return V1RuntimeSnapshot(
        state=clone_state(state),
        step_index=int(model._step_index),
        frozen_gate=None if model._frozen_gate is None else model._frozen_gate.detach().clone(),
    )


def restore_v1_runtime(model: DemianV1GateState, snapshot: V1RuntimeSnapshot) -> V1State:
    """Restore continuation metadata and return a cloned six-channel state."""

    if len(snapshot.state) != len(V1_CHANNELS):
        raise ValueError("v1 runtime snapshot requires six channels")
    batch_size = snapshot.state[0].shape[0]
    expected_widths = (
        model.hidden_size,
        model.hidden_size,
        model.control_dim,
        model.hidden_size,
        model.hidden_size,
        model.hidden_size,
    )
    if any(
        component.ndim != 2
        or component.shape[0] != batch_size
        or component.shape[1] != width
        for component, width in zip(snapshot.state, expected_widths, strict=True)
    ):
        raise ValueError("v1 runtime snapshot channel shapes do not match the model")
    if snapshot.frozen_gate is not None and snapshot.frozen_gate.shape != snapshot.state[-1].shape:
        raise ValueError("frozen gate shape must match state channels")
    model._step_index = int(snapshot.step_index)
    model._frozen_gate = (
        None if snapshot.frozen_gate is None else snapshot.frozen_gate.detach().clone()
    )
    return clone_state(snapshot.state)


def clamp_v1_channel(state: V1State, channel: str) -> V1State:
    """Return a state with one internal channel clamped to zero."""

    if channel not in V1_CHANNELS:
        raise ValueError(f"unknown v1 channel: {channel}")
    parts = list(state)
    parts[V1_CHANNELS.index(channel)] = torch.zeros_like(parts[V1_CHANNELS.index(channel)])
    return tuple(parts)  # type: ignore[return-value]


def surface_only_resume_state(model: DemianV1GateState, state: V1State) -> V1State:
    """Reconstruct only the exposed surface and zero all internal channels."""

    surface = model.state_vector(state).detach()
    fast, slow, control, message, carrier, gate = state
    return (
        surface.clone(),
        torch.zeros_like(slow),
        torch.zeros_like(control),
        torch.zeros_like(message),
        torch.zeros_like(carrier),
        torch.zeros_like(gate),
    )


def run_v1_trace(
    model: DemianV1GateState,
    *,
    steps: int,
    seed: int,
    perturb_step: int | None = None,
    perturb_scale: float = 0.0,
    ablation: V1Ablation = "none",
) -> tuple[list[torch.Tensor], V1State, list[dict[str, float]]]:
    """Run a deterministic CPU trace and collect exposed surfaces plus metrics."""

    torch.manual_seed(seed)
    state = model.initial_state(1, torch.device("cpu"))
    frozen_gate = state[-1].detach().clone()
    surfaces: list[torch.Tensor] = []
    metrics: list[dict[str, float]] = []
    for step in range(steps):
        if perturb_step is not None and step == perturb_step:
            fast, slow, control, message, carrier, gate = state
            fast = fast + perturb_scale * torch.randn_like(fast)
            state = fast, slow, control, message, carrier, gate
        state = model.step(state)
        if ablation == "gate_disabled":
            state = clamp_v1_channel(state, "gate")
        elif ablation == "gate_frozen":
            fast, slow, control, message, carrier, _ = state
            state = fast, slow, control, message, carrier, frozen_gate
        elif ablation == "message_disabled":
            state = clamp_v1_channel(state, "message")
        elif ablation == "carrier_disabled":
            state = clamp_v1_channel(state, "carrier")
        elif ablation == "slow_disabled":
            state = clamp_v1_channel(state, "slow")
        elif ablation == "control_disabled":
            state = clamp_v1_channel(state, "control")
        surfaces.append(model.state_vector(state).view(-1).detach().clone())
        metrics.append(dict(model.step_aux()))
    return surfaces, state, metrics


def compare_v1_resume(
    model: DemianV1GateState,
    *,
    seed: int,
    pause_steps: int,
    resume_steps: int,
    surface_only: bool,
) -> V1ResumeResult:
    """Compare full-capsule or surface-only resume against uninterrupted run."""

    torch.manual_seed(seed)
    source = model
    state = source.initial_state(1, torch.device("cpu"))
    for _ in range(pause_steps):
        state = source.step(state)
    paused_model = copy.deepcopy(source)
    paused_state = clone_state(state)

    uninterrupted = copy.deepcopy(paused_model)
    uninterrupted_state = clone_state(paused_state)
    target_surfaces = []
    for _ in range(resume_steps):
        uninterrupted_state = uninterrupted.step(uninterrupted_state)
        target_surfaces.append(uninterrupted.state_vector(uninterrupted_state).view(-1).detach().clone())

    resumed = copy.deepcopy(paused_model)
    resumed_state = surface_only_resume_state(resumed, paused_state) if surface_only else clone_state(paused_state)
    resumed_surfaces = []
    for _ in range(resume_steps):
        resumed_state = resumed.step(resumed_state)
        resumed_surfaces.append(resumed.state_vector(resumed_state).view(-1).detach().clone())

    gaps = [
        float(torch.norm(actual - expected).item() / math.sqrt(max(actual.numel(), 1)))
        for actual, expected in zip(resumed_surfaces, target_surfaces, strict=True)
    ]
    return V1ResumeResult(final_gap=gaps[-1], mean_step_gap=sum(gaps) / len(gaps))
