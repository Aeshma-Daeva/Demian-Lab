"""Common untrained closed-loop adapters for small recurrent comparators."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import torch
from torch import nn

from development.closed_loop_demian import FrozenObservationEncoder
from development.closed_loop_world import ActionAcknowledgement, ActionProposal, WorldObservation
from development.demian_v1_gate_state import DemianV1GateState, V1State


Architecture = Literal["mlp", "rnn", "gru", "demian", "demian_route_ablation"]


@dataclass(frozen=True)
class AdapterStep:
    surface: list[float]
    proposal: ActionProposal


@dataclass(frozen=True)
class ModelRuntimeSnapshot:
    state: tuple[torch.Tensor, ...]
    configuration_fingerprint: str


class ClosedLoopModelAdapter:
    """One declared state owner with a shared input encoder and action head."""

    def __init__(self, *, architecture: Architecture, seed: int, hidden_size: int, symbol_count: int = 3) -> None:
        self.architecture = architecture
        self.seed = seed
        self.hidden_size = hidden_size
        self.symbol_count = symbol_count
        torch.manual_seed(seed)
        self.encoder = FrozenObservationEncoder(hidden_size=hidden_size, symbol_count=symbol_count)
        self.model: nn.Module
        self.state: tuple[torch.Tensor, ...]
        if architecture == "mlp":
            self.model = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size))
            self.state = ()
        elif architecture == "rnn":
            self.model = nn.RNNCell(hidden_size, hidden_size, nonlinearity="tanh")
            self.state = (torch.zeros((1, hidden_size)),)
        elif architecture == "gru":
            self.model = nn.GRUCell(hidden_size, hidden_size)
            self.state = (torch.zeros((1, hidden_size)),)
        else:
            gate_disabled = architecture == "demian_route_ablation"
            self.model = DemianV1GateState(hidden_size, gate_disabled=gate_disabled, trace_routes=False)
            self.state = tuple(self.model.initial_state(1, torch.device("cpu")))
        self.action_head = nn.Linear(hidden_size, 4 + symbol_count)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters()) + sum(
            parameter.numel() for parameter in self.action_head.parameters()
        )

    @property
    def state_tensor_count(self) -> int:
        return len(self.state)

    @property
    def state_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.state)

    def advance(self, observation: WorldObservation, acknowledgement: ActionAcknowledgement | None) -> AdapterStep:
        encoded = self.encoder.encode(observation, acknowledgement)
        if self.architecture == "mlp":
            surface = self.model(encoded)
        elif self.architecture in {"rnn", "gru"}:
            next_state = self.model(encoded, self.state[0])
            self.state = (next_state,)
            surface = next_state
        else:
            assert isinstance(self.model, DemianV1GateState)
            state: V1State = self.state  # type: ignore[assignment]
            state = self.encoder.inject(state, encoded)
            state = self.model.step(state)
            self.state = tuple(state)
            surface = self.model.state_vector(state)
        return AdapterStep(
            surface=surface.reshape(-1).detach().cpu().tolist(),
            proposal=self._decode(surface),
        )

    def capture_runtime(self) -> ModelRuntimeSnapshot:
        return ModelRuntimeSnapshot(
            state=tuple(value.detach().clone() for value in self.state),
            configuration_fingerprint=self._configuration_fingerprint(),
        )

    def restore_runtime(self, snapshot: ModelRuntimeSnapshot) -> None:
        if snapshot.configuration_fingerprint != self._configuration_fingerprint():
            raise ValueError("comparator configuration mismatch")
        self.state = tuple(value.detach().clone() for value in snapshot.state)

    def _decode(self, surface: torch.Tensor) -> ActionProposal:
        logits = self.action_head(surface).reshape(-1)
        operation = ("noop", "read", "write", "answer")[int(torch.argmax(logits[:4]).item())]
        if operation in {"write", "answer"}:
            operand = int(torch.argmax(logits[4:]).item()) % self.symbol_count
            return ActionProposal(operation=operation, operand=operand)  # type: ignore[arg-type]
        return ActionProposal(operation=operation)  # type: ignore[arg-type]

    def _configuration_fingerprint(self) -> str:
        configuration = {
            "architecture": self.architecture,
            "seed": self.seed,
            "hidden_size": self.hidden_size,
            "symbol_count": self.symbol_count,
            "encoder_scale": self.encoder.scale,
        }
        digest = sha256(json.dumps(configuration, sort_keys=True).encode("utf-8"))
        for name, value in list(self.model.state_dict().items()) + list(self.action_head.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(repr(value.detach().cpu().tolist()).encode("utf-8"))
        return digest.hexdigest()
