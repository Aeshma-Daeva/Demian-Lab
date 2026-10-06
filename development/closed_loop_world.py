"""Deterministic bounded world and capability-scoped action connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Operation = Literal["noop", "read", "write", "answer"]
Phase = Literal["cue", "delay", "query", "complete"]


@dataclass(frozen=True)
class ActionProposal:
    operation: Operation
    operand: int | None = None


@dataclass(frozen=True)
class ActionAcceptance:
    accepted: bool
    reason: str

    def __bool__(self) -> bool:
        return self.accepted


@dataclass(frozen=True)
class ActionAcknowledgement:
    executed_operation: Operation
    register_value: int | None
    read_value: int | None
    answer_correct: bool | None


@dataclass(frozen=True)
class WorldObservation:
    tick: int
    phase: Phase
    cue_symbol: int | None
    query: bool
    score: None = None


@dataclass(frozen=True)
class WorldRuntimeSnapshot:
    tick: int
    register: int | None
    score: int
    completed: bool
    pending: ActionProposal | None
    last_acknowledgement: ActionAcknowledgement | None


class CueDelayQueryWorld:
    """A deterministic cue-delay-query world with one bounded register."""

    def __init__(self, *, symbol_count: int, cue_symbol: int, delay_steps: int) -> None:
        if symbol_count < 2:
            raise ValueError("symbol_count must be at least 2")
        if not 0 <= cue_symbol < symbol_count:
            raise ValueError("cue_symbol must fall within symbol_count")
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self.symbol_count = symbol_count
        self.cue_symbol = cue_symbol
        self.delay_steps = delay_steps
        self.tick = 0
        self.register: int | None = None
        self.score = 0
        self.completed = False

    @property
    def phase(self) -> Phase:
        if self.completed:
            return "complete"
        if self.tick == 0:
            return "cue"
        if self.tick <= self.delay_steps:
            return "delay"
        return "query"

    def observe(self) -> WorldObservation:
        phase = self.phase
        return WorldObservation(
            tick=self.tick,
            phase=phase,
            cue_symbol=self.cue_symbol if phase == "cue" else None,
            query=phase == "query",
        )

    def advance_time(self) -> None:
        if not self.completed:
            self.tick += 1

    def reset_storage(self) -> None:
        self.register = None

    def apply(self, action: ActionProposal) -> ActionAcknowledgement:
        if action.operation == "noop":
            return self._acknowledgement("noop")
        if action.operation == "write":
            assert action.operand is not None
            self.register = action.operand
            return self._acknowledgement("write")
        if action.operation == "read":
            return self._acknowledgement("read", read_value=self.register)
        assert action.operation == "answer"
        assert action.operand is not None
        if self.phase != "query":
            return self._acknowledgement("answer")
        correct = action.operand == self.cue_symbol
        self.score += int(correct)
        self.completed = True
        return self._acknowledgement("answer", answer_correct=correct)

    def _acknowledgement(
        self,
        operation: Operation,
        *,
        read_value: int | None = None,
        answer_correct: bool | None = None,
    ) -> ActionAcknowledgement:
        return ActionAcknowledgement(
            executed_operation=operation,
            register_value=self.register,
            read_value=read_value,
            answer_correct=answer_correct,
        )


class WorldConnector:
    """One-action queue with validation and explicit acknowledgement state."""

    def __init__(
        self,
        *,
        symbol_count: int,
        storage_enabled: bool = True,
        storage_read_only: bool = False,
    ) -> None:
        self.symbol_count = symbol_count
        self.storage_enabled = storage_enabled
        self.storage_read_only = storage_read_only
        self.pending: ActionProposal | None = None
        self.last_acknowledgement: ActionAcknowledgement | None = None

    def submit(self, proposal: ActionProposal) -> ActionAcceptance:
        if self.pending is not None:
            return ActionAcceptance(False, "connector_busy")
        reason = self._validation_error(proposal)
        if reason is not None:
            return ActionAcceptance(False, reason)
        self.pending = proposal
        return ActionAcceptance(True, "queued")

    def advance(self, world: CueDelayQueryWorld) -> ActionAcknowledgement:
        proposal = self.pending or ActionProposal("noop")
        self.pending = None
        acknowledgement = world.apply(proposal)
        self.last_acknowledgement = acknowledgement
        return acknowledgement

    def _validation_error(self, proposal: ActionProposal) -> str | None:
        if proposal.operation not in {"noop", "read", "write", "answer"}:
            return "unknown_operation"
        if proposal.operation in {"read", "write"} and not self.storage_enabled:
            return "storage_disabled"
        if proposal.operation == "write" and self.storage_read_only:
            return "storage_read_only"
        if proposal.operation in {"write", "answer"}:
            if proposal.operand is None:
                return "missing_operand"
            if not 0 <= proposal.operand < self.symbol_count:
                return "operand_out_of_range"
        elif proposal.operand is not None:
            return "unexpected_operand"
        return None


def capture_world_runtime(
    world: CueDelayQueryWorld,
    connector: WorldConnector,
) -> WorldRuntimeSnapshot:
    return WorldRuntimeSnapshot(
        tick=world.tick,
        register=world.register,
        score=world.score,
        completed=world.completed,
        pending=connector.pending,
        last_acknowledgement=connector.last_acknowledgement,
    )


def restore_world_runtime(
    world: CueDelayQueryWorld,
    connector: WorldConnector,
    snapshot: WorldRuntimeSnapshot,
) -> None:
    world.tick = snapshot.tick
    world.register = snapshot.register
    world.score = snapshot.score
    world.completed = snapshot.completed
    connector.pending = snapshot.pending
    connector.last_acknowledgement = snapshot.last_acknowledgement
