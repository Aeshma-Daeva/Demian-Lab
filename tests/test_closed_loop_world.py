"""Deterministic world and connector contracts for closed-loop Demian."""

from __future__ import annotations

import pytest

from development.closed_loop_world import (
    ActionProposal,
    CueDelayQueryWorld,
    WorldConnector,
    capture_world_runtime,
    restore_world_runtime,
)


def test_world_exposes_cue_delay_and_query_without_score_leakage() -> None:
    world = CueDelayQueryWorld(symbol_count=3, cue_symbol=2, delay_steps=2)

    assert world.observe().phase == "cue"
    assert world.observe().cue_symbol == 2
    assert world.observe().score is None

    world.advance_time()
    assert world.observe().phase == "delay"
    assert world.observe().cue_symbol is None
    world.advance_time()
    assert world.observe().phase == "delay"
    world.advance_time()
    assert world.observe().phase == "query"
    assert world.observe().query is True
    assert world.observe().score is None


def test_connector_queues_one_action_and_returns_acknowledged_register_effect() -> None:
    world = CueDelayQueryWorld(symbol_count=3, cue_symbol=1, delay_steps=1)
    connector = WorldConnector(symbol_count=3)

    queued = connector.submit(ActionProposal(operation="write", operand=2))
    assert queued.accepted is True
    assert world.register is None
    assert connector.pending is not None

    acknowledgement = connector.advance(world)
    assert acknowledgement.executed_operation == "write"
    assert acknowledgement.register_value == 2
    assert world.register == 2
    assert connector.pending is None

    assert connector.submit(ActionProposal(operation="read"))
    acknowledgement = connector.advance(world)
    assert acknowledgement.executed_operation == "read"
    assert acknowledgement.read_value == 2


def test_connector_rejects_invalid_or_competing_action_without_mutating_world() -> None:
    world = CueDelayQueryWorld(symbol_count=3, cue_symbol=1, delay_steps=1)
    connector = WorldConnector(symbol_count=3)

    invalid = connector.submit(ActionProposal(operation="write", operand=9))
    assert invalid.accepted is False
    assert invalid.reason == "operand_out_of_range"
    assert world.register is None

    assert connector.submit(ActionProposal(operation="noop"))
    competing = connector.submit(ActionProposal(operation="read"))
    assert competing.accepted is False
    assert competing.reason == "connector_busy"
    assert connector.advance(world).executed_operation == "noop"
    assert world.register is None


def test_storage_capabilities_and_reset_are_explicit_controls() -> None:
    world = CueDelayQueryWorld(symbol_count=3, cue_symbol=1, delay_steps=1)
    disabled = WorldConnector(symbol_count=3, storage_enabled=False)
    read_only = WorldConnector(symbol_count=3, storage_read_only=True)

    assert disabled.submit(ActionProposal(operation="read")).reason == "storage_disabled"
    assert disabled.submit(ActionProposal(operation="write", operand=1)).reason == "storage_disabled"
    assert read_only.submit(ActionProposal(operation="write", operand=1)).reason == "storage_read_only"
    assert read_only.submit(ActionProposal(operation="read")).accepted is True

    world.register = 2
    world.reset_storage()
    assert world.register is None

def test_world_and_connector_snapshot_restore_replays_exactly() -> None:
    world = CueDelayQueryWorld(symbol_count=3, cue_symbol=1, delay_steps=1)
    connector = WorldConnector(symbol_count=3)
    assert connector.submit(ActionProposal(operation="write", operand=1))
    world.advance_time()
    snapshot = capture_world_runtime(world, connector)

    observed_first = []
    for proposal in [ActionProposal(operation="read"), ActionProposal(operation="answer", operand=1)]:
        observed_first.append(connector.advance(world))
        connector.submit(proposal)
        world.advance_time()

    restore_world_runtime(world, connector, snapshot)
    observed_second = []
    for proposal in [ActionProposal(operation="read"), ActionProposal(operation="answer", operand=1)]:
        observed_second.append(connector.advance(world))
        connector.submit(proposal)
        world.advance_time()

    assert observed_first == observed_second
    assert world.register == 1


def test_answer_is_only_scored_at_query() -> None:
    world = CueDelayQueryWorld(symbol_count=3, cue_symbol=1, delay_steps=1)
    connector = WorldConnector(symbol_count=3)
    assert connector.submit(ActionProposal(operation="answer", operand=1))

    premature = connector.advance(world)
    assert premature.executed_operation == "answer"
    assert premature.answer_correct is None

    world.advance_time()
    world.advance_time()
    assert world.observe().phase == "query"
    assert connector.submit(ActionProposal(operation="answer", operand=1))
    scored = connector.advance(world)
    assert scored.answer_correct is True
    assert world.score == 1


def test_world_rejects_invalid_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="symbol_count"):
        CueDelayQueryWorld(symbol_count=1, cue_symbol=0, delay_steps=1)
    with pytest.raises(ValueError, match="cue_symbol"):
        CueDelayQueryWorld(symbol_count=3, cue_symbol=3, delay_steps=1)
    with pytest.raises(ValueError, match="delay_steps"):
        CueDelayQueryWorld(symbol_count=3, cue_symbol=1, delay_steps=-1)
