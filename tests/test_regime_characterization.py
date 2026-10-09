"""Analytic and finite-difference checks for tangent instrumentation."""

import math

import pytest
import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter
from development.regime_characterization import StateMap, tangent_window, check_jvp


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 1.2])
def test_tangent_rate_retains_negative_zero_and_positive_growth(multiplier):
    result = tangent_window(
        lambda state, value: multiplier * state,
        lambda state: state,
        torch.ones(2, dtype=torch.float64),
        [torch.zeros(1, dtype=torch.float64)] * 16,
        direction_seed=1701,
    )
    assert result["rate"] == pytest.approx(math.log(multiplier), abs=1e-12)
    assert result["readout_rate"] == pytest.approx(math.log(multiplier), abs=1e-12)


def test_zero_initial_readout_visibility_is_unresolved():
    result = tangent_window(
        lambda state, value: state,
        lambda state: state[:1] * 0,
        torch.ones(2, dtype=torch.float64),
        [torch.zeros(1, dtype=torch.float64)] * 4,
        direction_seed=1701,
    )
    assert result["readout_rate"] is None
    assert result["readout_status"] == "initial_visibility_below_floor"


@pytest.mark.parametrize("architecture", ["rnn", "gru", "demian", "demian_route_ablation"])
def test_state_map_matches_adapter_and_jvp_without_mutating_runtime(architecture):
    adapter = ClosedLoopModelAdapter(architecture=architecture, seed=94, hidden_size=8)
    mapping = StateMap(adapter)
    state = mapping.initial_state
    encoded = torch.full((1, 8), 0.01, dtype=torch.float64)
    before = getattr(adapter.model, "_step_index", None)
    expected = adapter._advance_encoded(encoded)
    adapter.model._step_index = before if before is not None else 0
    computed = mapping.transition(state, encoded)
    assert torch.allclose(mapping.readout(computed), torch.tensor(expected.surface, dtype=torch.float64))
    assert torch.allclose(computed, torch.cat([value.reshape(-1) for value in adapter.state]))
    check = check_jvp(mapping.transition, state, encoded, direction_seed=1701)
    assert check["passed"]
    assert getattr(adapter.model, "_step_index", None) == before if before is not None else True
