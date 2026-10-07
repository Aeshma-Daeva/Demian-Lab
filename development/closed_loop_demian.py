"""Declared input/action boundary for deterministic closed-loop Demian tests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

import torch

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
from development.demian_v1_gate_state import (
    V1RuntimeSnapshot,
    V1State,
    DemianV1GateState,
    capture_v1_runtime,
    restore_v1_runtime,
)
from development.demian_v1_measurement import V1RouteStepRecord


@dataclass(frozen=True)
class InputFrame:
    observation: WorldObservation
    acknowledgement: ActionAcknowledgement


@dataclass(frozen=True)
class ClosedLoopRecord:
    observation: WorldObservation
    executed_acknowledgement: ActionAcknowledgement
    input_frame: InputFrame
    proposal: ActionProposal
    acceptance: ActionAcceptance
    surface: list[float]
    route_step: V1RouteStepRecord | None


@dataclass(frozen=True)
class ReplayRecord:
    surface: list[float]
    proposal: ActionProposal


@dataclass(frozen=True)
class ClosedLoopRuntimeSnapshot:
    demian: V1RuntimeSnapshot
    world: WorldRuntimeSnapshot
    configuration_fingerprint: str


class FrozenObservationEncoder:
    """Stateless, parameter-free mapping from declared world fields to fast state."""

    def __init__(self, *, hidden_size: int, symbol_count: int, scale: float = 0.25) -> None:
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if symbol_count < 2:
            raise ValueError("symbol_count must be at least 2")
        self.hidden_size = hidden_size
        self.symbol_count = symbol_count
        self.scale = scale

    def encode(
        self,
        observation: WorldObservation,
        acknowledgement: ActionAcknowledgement | None,
    ) -> torch.Tensor:
        encoded = torch.zeros((1, self.hidden_size), dtype=torch.float32)

        def add(slot: int, value: float) -> None:
            encoded[0, slot % self.hidden_size] += value

        phase_index = {"cue": 0, "delay": 1, "query": 2, "complete": 3}[observation.phase]
        add(phase_index, 1.0)
        if observation.cue_symbol is not None:
            add(4 + observation.cue_symbol, 1.0)
        if observation.query:
            add(4 + self.symbol_count, 1.0)
        if acknowledgement is not None:
            operation_index = {"noop": 0, "read": 1, "write": 2, "answer": 3}[
                acknowledgement.executed_operation
            ]
            add(5 + self.symbol_count + operation_index, 1.0)
            if acknowledgement.read_value is not None:
                add(9 + self.symbol_count + acknowledgement.read_value, 1.0)
            if acknowledgement.answer_correct is not None:
                add(9 + 2 * self.symbol_count + int(acknowledgement.answer_correct), 1.0)
        return encoded * self.scale

    def inject(self, state: V1State, encoded: torch.Tensor) -> V1State:
        fast, slow, control, message, carrier, gate = state
        return fast + encoded.to(device=fast.device, dtype=fast.dtype), slow, control, message, carrier, gate


class FrozenActionDecoder:
    """Stateless declared decoder from six-channel state to bounded actions."""

    def __init__(self, *, symbol_count: int) -> None:
        self.symbol_count = symbol_count

    def decode(self, components: dict[str, torch.Tensor]) -> ActionProposal:
        control = components["control"].reshape(-1)
        operation = ("noop", "read", "write", "answer")[int(torch.argmax(control[:4]).item())]
        if operation in {"write", "answer"}:
            operand = int(torch.argmax(components["fast"].reshape(-1)).item()) % self.symbol_count
            return ActionProposal(operation=operation, operand=operand)  # type: ignore[arg-type]
        return ActionProposal(operation=operation)  # type: ignore[arg-type]


class ClosedLoopDemianRunner:
    """One deterministic Demian/world loop with explicit connector ownership."""

    def __init__(self, *, seed: int, hidden_size: int, cue_symbol: int, delay_steps: int) -> None:
        self.seed = seed
        self.hidden_size = hidden_size
        torch.manual_seed(seed)
        self.model = DemianV1GateState(hidden_size, trace_routes=True)
        self.state = self.model.initial_state(1, torch.device("cpu"))
        self.world = CueDelayQueryWorld(symbol_count=3, cue_symbol=cue_symbol, delay_steps=delay_steps)
        self.connector = WorldConnector(symbol_count=3)
        self.encoder = FrozenObservationEncoder(hidden_size=hidden_size, symbol_count=3)
        self.decoder = FrozenActionDecoder(symbol_count=3)

    def step(self) -> ClosedLoopRecord:
        if self.world.completed:
            raise RuntimeError("closed-loop episode is completed")
        acknowledgement = self.connector.advance(self.world)
        observation = self.world.observe()
        frame = InputFrame(observation=observation, acknowledgement=acknowledgement)
        encoded = self.encoder.encode(observation, acknowledgement)
        self.state = self.encoder.inject(self.state, encoded)
        self.state = self.model.step(self.state)
        proposal = self.decoder.decode(self.model.state_components(self.state))
        acceptance = self.connector.submit(proposal)
        route_trace = self.model.route_trace()
        route_step = (
            None
            if route_trace is None
            else V1RouteStepRecord(
                step=route_trace.step,
                values={
                    route_id: value.reshape(-1).detach().cpu().tolist()
                    for route_id, value in route_trace.values.items()
                },
            )
        )
        surface = self.model.state_vector(self.state).reshape(-1).detach().cpu().tolist()
        self.world.advance_time()
        return ClosedLoopRecord(
            observation=observation,
            executed_acknowledgement=acknowledgement,
            input_frame=frame,
            proposal=proposal,
            acceptance=acceptance,
            surface=surface,
            route_step=route_step,
        )


def capture_closed_loop_runtime(runner: ClosedLoopDemianRunner) -> ClosedLoopRuntimeSnapshot:
    return ClosedLoopRuntimeSnapshot(
        demian=capture_v1_runtime(runner.model, runner.state),
        world=capture_world_runtime(runner.world, runner.connector),
        configuration_fingerprint=_configuration_fingerprint(runner),
    )


def restore_closed_loop_runtime(
    runner: ClosedLoopDemianRunner,
    snapshot: ClosedLoopRuntimeSnapshot,
) -> None:
    if _configuration_fingerprint(runner) != snapshot.configuration_fingerprint:
        raise ValueError("closed-loop configuration mismatch")
    runner.state = restore_v1_runtime(runner.model, snapshot.demian)
    restore_world_runtime(runner.world, runner.connector, snapshot.world)


def _configuration_fingerprint(runner: ClosedLoopDemianRunner) -> str:
    """Bind a runtime snapshot to immutable model, interface, and world configuration."""
    configuration = {
        "model": {
            "hidden_size": runner.model.hidden_size,
            "trace_routes": runner.model.trace_routes,
        },
        "encoder": {
            "hidden_size": runner.encoder.hidden_size,
            "symbol_count": runner.encoder.symbol_count,
            "scale": runner.encoder.scale,
        },
        "decoder": {"symbol_count": runner.decoder.symbol_count},
        "world": {
            "symbol_count": runner.world.symbol_count,
            "cue_symbol": runner.world.cue_symbol,
            "delay_steps": runner.world.delay_steps,
        },
        "connector": {
            "symbol_count": runner.connector.symbol_count,
            "storage_enabled": runner.connector.storage_enabled,
            "storage_read_only": runner.connector.storage_read_only,
        },
    }
    digest = sha256(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    for name, value in runner.model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(repr(value.detach().cpu().tolist()).encode("utf-8"))
    return digest.hexdigest()


def replay_observation_frames(
    *,
    seed: int,
    hidden_size: int,
    frames: list[InputFrame],
) -> list[ReplayRecord]:
    torch.manual_seed(seed)
    model = DemianV1GateState(hidden_size, trace_routes=True)
    state = model.initial_state(1, torch.device("cpu"))
    encoder = FrozenObservationEncoder(hidden_size=hidden_size, symbol_count=3)
    decoder = FrozenActionDecoder(symbol_count=3)
    records: list[ReplayRecord] = []
    for frame in frames:
        state = encoder.inject(state, encoder.encode(frame.observation, frame.acknowledgement))
        state = model.step(state)
        records.append(
            ReplayRecord(
                surface=model.state_vector(state).reshape(-1).detach().cpu().tolist(),
                proposal=decoder.decode(model.state_components(state)),
            )
        )
    return records
