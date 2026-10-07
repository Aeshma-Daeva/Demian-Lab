"""Common closed-loop comparator adapter contracts."""

from __future__ import annotations

import pytest

from development.closed_loop_comparators import ClosedLoopModelAdapter
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
