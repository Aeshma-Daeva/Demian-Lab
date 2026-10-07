"""Closed-loop Demian interface contracts."""

from __future__ import annotations

import pytest
import torch

from development.closed_loop_demian import (
    ClosedLoopDemianRunner,
    FrozenActionDecoder,
    FrozenObservationEncoder,
    capture_closed_loop_runtime,
    replay_observation_frames,
    restore_closed_loop_runtime,
)


def test_frozen_observation_encoder_changes_only_fast_channel() -> None:
    runner = ClosedLoopDemianRunner(seed=94, hidden_size=8, cue_symbol=1, delay_steps=1)
    before = tuple(component.clone() for component in runner.state)
    encoded = FrozenObservationEncoder(hidden_size=8, symbol_count=3).encode(
        runner.world.observe(), None
    )

    after = FrozenObservationEncoder(hidden_size=8, symbol_count=3).inject(
        before, encoded
    )

    assert after[0].tolist() != before[0].tolist()
    assert all(after[index].tolist() == before[index].tolist() for index in range(1, 6))


def test_frozen_action_decoder_has_bounded_operation_and_operand() -> None:
    runner = ClosedLoopDemianRunner(seed=94, hidden_size=8, cue_symbol=1, delay_steps=1)
    decoder = FrozenActionDecoder(symbol_count=3)

    proposal = decoder.decode(runner.model.state_components(runner.state))

    assert proposal.operation in {"noop", "read", "write", "answer"}
    if proposal.operation in {"write", "answer"}:
        assert proposal.operand in {0, 1, 2}
    else:
        assert proposal.operand is None


def test_closed_loop_runner_logs_world_action_and_route_state() -> None:
    runner = ClosedLoopDemianRunner(seed=94, hidden_size=8, cue_symbol=1, delay_steps=1)

    record = runner.step()

    assert record.observation.phase == "cue"
    assert record.executed_acknowledgement.executed_operation == "noop"
    assert record.acceptance.accepted is True
    assert record.route_step is not None
    assert runner.world.tick == 1


def test_closed_loop_checkpoint_restores_exact_future_trace() -> None:
    runner = ClosedLoopDemianRunner(seed=95, hidden_size=8, cue_symbol=2, delay_steps=2)
    runner.step()
    snapshot = capture_closed_loop_runtime(runner)

    first = [runner.step(), runner.step()]
    restore_closed_loop_runtime(runner, snapshot)
    second = [runner.step(), runner.step()]

    assert first == second


def test_closed_loop_checkpoint_restores_into_fresh_matching_runner() -> None:
    runner = ClosedLoopDemianRunner(seed=95, hidden_size=8, cue_symbol=2, delay_steps=2)
    runner.step()
    snapshot = capture_closed_loop_runtime(runner)
    first = [runner.step(), runner.step()]

    restored = ClosedLoopDemianRunner(seed=95, hidden_size=8, cue_symbol=2, delay_steps=2)
    restore_closed_loop_runtime(restored, snapshot)

    assert [restored.step(), restored.step()] == first


@pytest.mark.parametrize("mismatch", ["weight", "encoder", "world", "capability"])
def test_closed_loop_checkpoint_rejects_configuration_mismatch(mismatch: str) -> None:
    source = ClosedLoopDemianRunner(seed=95, hidden_size=8, cue_symbol=2, delay_steps=2)
    snapshot = capture_closed_loop_runtime(source)
    target = ClosedLoopDemianRunner(seed=95, hidden_size=8, cue_symbol=2, delay_steps=2)

    if mismatch == "weight":
        with torch.no_grad():
            next(target.model.parameters()).add_(1.0)
    elif mismatch == "encoder":
        target.encoder.scale = 0.5
    elif mismatch == "world":
        target.world.delay_steps = 3
    else:
        target.connector.storage_enabled = False

    with pytest.raises(ValueError, match="configuration mismatch"):
        restore_closed_loop_runtime(target, snapshot)


def test_fixed_observation_replay_is_identical_to_recorded_closed_loop_inputs() -> None:
    runner = ClosedLoopDemianRunner(seed=96, hidden_size=8, cue_symbol=0, delay_steps=2)
    recorded = [runner.step() for _ in range(4)]

    replayed = replay_observation_frames(
        seed=96,
        hidden_size=8,
        frames=[record.input_frame for record in recorded],
    )

    assert [record.surface for record in replayed] == [record.surface for record in recorded]
    assert [record.proposal for record in replayed] == [record.proposal for record in recorded]


def test_terminal_episode_does_not_emit_repeated_complete_ticks() -> None:
    runner = ClosedLoopDemianRunner(seed=94, hidden_size=8, cue_symbol=1, delay_steps=1)

    while not runner.world.completed:
        runner.step()

    with pytest.raises(RuntimeError, match="completed"):
        runner.step()
