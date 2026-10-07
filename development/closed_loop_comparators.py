"""Common untrained closed-loop adapters for small recurrent comparators."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Literal

import torch
from torch import nn

from development.closed_loop_demian import FrozenObservationEncoder, InputFrame
from development.closed_loop_world import (
    ActionAcceptance,
    ActionAcknowledgement,
    ActionProposal,
    CueDelayQueryWorld,
    WorldConnector,
    WorldObservation,
    WorldRuntimeSnapshot,
    capture_world_runtime,
    restore_world_runtime,
)
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


@dataclass(frozen=True)
class ComparatorRecord:
    observation: WorldObservation
    executed_acknowledgement: ActionAcknowledgement
    input_frame: InputFrame
    proposal: ActionProposal
    acceptance: ActionAcceptance
    surface: list[float]


@dataclass(frozen=True)
class ComparatorRuntimeSnapshot:
    adapter: ModelRuntimeSnapshot
    world: WorldRuntimeSnapshot
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
        return self._advance_encoded(encoded)

    def advance_autonomous(self) -> AdapterStep:
        return self._advance_encoded(torch.zeros((1, self.hidden_size), dtype=torch.float32))

    def _advance_encoded(self, encoded: torch.Tensor) -> AdapterStep:
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


class ComparatorRunner:
    """Common world ordering for a declared comparator adapter."""

    def __init__(
        self,
        *,
        architecture: Architecture,
        seed: int,
        hidden_size: int,
        cue_symbol: int,
        delay_steps: int,
    ) -> None:
        self.adapter = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
        self.world = CueDelayQueryWorld(symbol_count=3, cue_symbol=cue_symbol, delay_steps=delay_steps)
        self.connector = WorldConnector(symbol_count=3)

    def step(self) -> ComparatorRecord:
        if self.world.completed:
            raise RuntimeError("closed-loop episode is completed")
        acknowledgement = self.connector.advance(self.world)
        observation = self.world.observe()
        input_frame = InputFrame(observation=observation, acknowledgement=acknowledgement)
        step = self.adapter.advance(observation, acknowledgement)
        acceptance = self.connector.submit(step.proposal)
        self.world.advance_time()
        return ComparatorRecord(
            observation=observation,
            executed_acknowledgement=acknowledgement,
            input_frame=input_frame,
            proposal=step.proposal,
            acceptance=acceptance,
            surface=step.surface,
        )


def capture_comparator_runtime(runner: ComparatorRunner) -> ComparatorRuntimeSnapshot:
    return ComparatorRuntimeSnapshot(
        adapter=runner.adapter.capture_runtime(),
        world=capture_world_runtime(runner.world, runner.connector),
        configuration_fingerprint=_runner_fingerprint(runner),
    )


def restore_comparator_runtime(runner: ComparatorRunner, snapshot: ComparatorRuntimeSnapshot) -> None:
    if snapshot.configuration_fingerprint != _runner_fingerprint(runner):
        raise ValueError("comparator runner configuration mismatch")
    runner.adapter.restore_runtime(snapshot.adapter)
    restore_world_runtime(runner.world, runner.connector, snapshot.world)


def replay_comparator_frames(
    *, architecture: Architecture, seed: int, hidden_size: int, frames: list[InputFrame]
) -> list[tuple[list[float], ActionProposal]]:
    adapter = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
    return [
        (step.surface, step.proposal)
        for frame in frames
        for step in [adapter.advance(frame.observation, frame.acknowledgement)]
    ]


def _runner_fingerprint(runner: ComparatorRunner) -> str:
    payload = {
        "adapter": runner.adapter._configuration_fingerprint(),
        "world": (runner.world.symbol_count, runner.world.cue_symbol, runner.world.delay_steps),
        "connector": (runner.connector.symbol_count, runner.connector.storage_enabled, runner.connector.storage_read_only),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def matched_perturbation_ticks(*, seed: int, steps: int, rate: float, salt: int) -> tuple[int, ...]:
    if steps < 1 or not 0 < rate <= 1:
        raise ValueError("steps and rate must be positive")
    count = max(1, round(steps * rate))
    return tuple(sorted(random.Random(f"{seed}:{salt}").sample(range(steps), count)))


def apply_relative_pulse(adapter: ClosedLoopModelAdapter, *, amplitude: float, seed: int, tick: int) -> None:
    if not adapter.state or amplitude == 0.0:
        return
    state_norm = sum(value.square().sum().item() for value in adapter.state) ** 0.5
    if state_norm == 0.0:
        return
    directions = []
    for index, value in enumerate(adapter.state):
        generator = torch.Generator(device=value.device).manual_seed(seed * 100_000 + tick * 10 + index)
        directions.append(torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype))
    direction_norm = sum(value.square().sum().item() for value in directions) ** 0.5
    adapter.state = tuple(
        state + direction * (state_norm * amplitude / direction_norm)
        for state, direction in zip(adapter.state, directions)
    )
