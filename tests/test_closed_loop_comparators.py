"""Common closed-loop comparator adapter contracts."""

from __future__ import annotations

import pytest

from development.closed_loop_comparators import (
    ClosedLoopModelAdapter,
    ComparatorRunner,
    capture_comparator_runtime,
    replay_comparator_frames,
    restore_comparator_runtime,
    apply_relative_pulse,
    build_budget_tracks,
    matched_perturbation_ticks,
    run_comparator_condition,
)
from development.closed_loop_world import WorldObservation


def test_memoryless_mlp_retains_no_runtime_state() -> None:
    adapter = ClosedLoopModelAdapter(architecture="mlp", seed=94, hidden_size=8)
    observation = WorldObservation(tick=0, phase="cue", cue_symbol=1, query=False)

    first = adapter.advance(observation, None)
    second = adapter.advance(observation, None)

    assert adapter.state_bytes == 0
    assert first.surface == second.surface
    assert first.proposal == second.proposal


@pytest.mark.parametrize("architecture", ["mlp", "rnn", "gru", "demian", "demian_route_ablation"])
def test_adapters_declare_complete_state_and_parameter_budget(architecture: str) -> None:
    adapter = ClosedLoopModelAdapter(architecture=architecture, seed=94, hidden_size=8)

    assert adapter.parameter_count > 0
    assert adapter.state_bytes == sum(
        value.numel() * value.element_size() for value in adapter.capture_runtime().state
    )
    if architecture == "mlp":
        assert adapter.state_bytes == 0
    else:
        assert adapter.state_bytes > 0


def test_checkpoint_restore_rejects_different_configuration() -> None:
    source = ClosedLoopModelAdapter(architecture="gru", seed=94, hidden_size=8)
    snapshot = source.capture_runtime()
    mismatch = ClosedLoopModelAdapter(architecture="gru", seed=95, hidden_size=8)

    with pytest.raises(ValueError, match="configuration mismatch"):
        mismatch.restore_runtime(snapshot)


def test_comparator_runner_preserves_world_action_order() -> None:
    runner = ComparatorRunner(architecture="gru", seed=94, hidden_size=8, cue_symbol=1, delay_steps=2)

    record = runner.step()

    assert record.observation.phase == "cue"
    assert record.executed_acknowledgement.executed_operation == "noop"
    assert record.acceptance.accepted is True
    assert runner.world.tick == 1


def test_comparator_runtime_restores_into_fresh_matching_runner() -> None:
    runner = ComparatorRunner(architecture="rnn", seed=95, hidden_size=8, cue_symbol=2, delay_steps=3)
    runner.step()
    snapshot = capture_comparator_runtime(runner)
    first = [runner.step(), runner.step()]

    restored = ComparatorRunner(architecture="rnn", seed=95, hidden_size=8, cue_symbol=2, delay_steps=3)
    restore_comparator_runtime(restored, snapshot)

    assert [restored.step(), restored.step()] == first


def test_fixed_observation_replay_matches_comparator_records() -> None:
    runner = ComparatorRunner(architecture="gru", seed=96, hidden_size=8, cue_symbol=0, delay_steps=3)
    records = [runner.step() for _ in range(4)]

    replayed = replay_comparator_frames(
        architecture="gru", seed=96, hidden_size=8, frames=[record.input_frame for record in records]
    )

    assert replayed == [(record.surface, record.proposal) for record in records]


def test_matched_perturbation_ticks_are_seed_deterministic() -> None:
    assert matched_perturbation_ticks(seed=94, steps=100, rate=0.01, salt=11) == matched_perturbation_ticks(
        seed=94, steps=100, rate=0.01, salt=11
    )


def test_relative_pulse_scales_to_complete_state_norm() -> None:
    adapter = ClosedLoopModelAdapter(architecture="gru", seed=94, hidden_size=8)
    before = adapter.capture_runtime().state
    apply_relative_pulse(adapter, amplitude=0.01, seed=94, tick=3)
    after = adapter.capture_runtime().state

    before_norm = sum(value.square().sum().item() for value in before) ** 0.5
    pulse_norm = sum((new - old).square().sum().item() for old, new in zip(before, after)) ** 0.5
    assert pulse_norm == pytest.approx(before_norm * 0.01)


def test_comparator_condition_records_action_and_storage_metrics() -> None:
    result = run_comparator_condition(
        architecture="rnn", condition="both", seeds=[94, 95], hidden_size=8, steps=16, delay_steps=15
    )

    assert result["condition"] == "both"
    assert result["metrics"]["trajectory_count"] == 2
    assert result["metrics"]["scheduled_internal_events"] == 2
    assert result["metrics"]["scheduled_environment_events"] == 2
    assert len(result["runs"]) == 2


def test_forced_read_exposure_marks_environment_disturbances_observed() -> None:
    result = run_comparator_condition(
        architecture="rnn",
        condition="environment",
        seeds=[94],
        hidden_size=8,
        steps=16,
        delay_steps=15,
        environment_exposure="forced_read",
    )

    event = result["runs"][0]["events"][0]
    assert event["observed"] is True
    assert event["read_tick"] == event["tick"]
    assert result["metrics"]["observed_environment_events"] == 1


def test_comparator_condition_samples_full_state_without_dropping_tick_metrics() -> None:
    result = run_comparator_condition(
        architecture="gru",
        condition="baseline",
        seeds=[94],
        hidden_size=8,
        steps=8,
        delay_steps=7,
        sample_every=4,
    )

    run = result["runs"][0]
    assert len(run["trace"]) == 8
    assert [sample["tick"] for sample in run["state_samples"]] == [0, 4, 7]
    assert all("full_state" not in step for step in run["trace"])
    assert all("full_state" in sample for sample in run["state_samples"])


def test_budget_tracks_are_deterministic_and_separate_state_from_parameters() -> None:
    first = build_budget_tracks(reference_hidden_size=8, seed=94)
    second = build_budget_tracks(reference_hidden_size=8, seed=94)

    assert first == second
    assert first["state"]["target"] == first["state"]["specifications"]["demian"]
    assert first["parameters"]["target"] == first["parameters"]["specifications"]["demian"]
    assert first["state"]["specifications"]["mlp"]["eligible"] is False
    assert first["state"]["specifications"]["gru"]["hidden_size"] != first["parameters"]["specifications"]["gru"]["hidden_size"]
