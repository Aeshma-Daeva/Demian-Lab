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


@pytest.mark.parametrize("scale", [0.01, 25.0])
def test_gru_operation_diagnostics_reconstruct_fused_transition(scale):
    mapping = StateMap(ClosedLoopModelAdapter(architecture="gru", seed=94, hidden_size=8))
    state = torch.linspace(-scale, scale, 8, dtype=torch.float64)
    point = mapping.operating_point(state, torch.full((1, 8), scale, dtype=torch.float64), 0.001)
    assert set(point["operation_saturation"]) == {"gru_reset", "gru_update", "gru_candidate"}
    assert point["saturation_coverage"] == "complete_declared_operations"
    assert point["transition_reconstruction_max_error"] < 1e-12


def test_gru_saturation_uses_activation_slopes_not_hidden_magnitude():
    mapping = StateMap(ClosedLoopModelAdapter(architecture="gru", seed=94, hidden_size=8))
    with torch.no_grad():
        for parameter in mapping.adapter.model.parameters():
            parameter.zero_()
    point = mapping.operating_point(torch.full((8,), 100.0, dtype=torch.float64), torch.zeros(1, 8).double(), 0.001)
    assert point["operation_saturation"] == {"gru_reset": 0.0, "gru_update": 0.0, "gru_candidate": 0.0}
    with torch.no_grad():
        mapping.adapter.model.bias_ih.fill_(100)
    saturated = mapping.operating_point(mapping.initial_state, torch.zeros(1, 8).double(), 0.001)
    assert saturated["operation_saturation"] == {"gru_reset": 1.0, "gru_update": 1.0, "gru_candidate": 1.0}


@pytest.mark.parametrize("architecture", ["demian", "demian_route_ablation"])
def test_demian_diagnostics_cover_routes_pressure_compound_and_readout(architecture):
    mapping = StateMap(ClosedLoopModelAdapter(architecture=architecture, seed=94, hidden_size=8))
    with torch.no_grad():
        for parameter in mapping.adapter.model.parameters():
            parameter.zero_()
        mapping.adapter.model.fast_mix.bias.fill_(100)
    state = torch.full_like(mapping.initial_state, 100)
    point = mapping.operating_point(state, torch.zeros(1, 8).double(), 0.001)
    slopes = point["operation_saturation"]
    assert len(slopes) == 26
    assert slopes["fast_mix"] == 1.0
    assert slopes["fast_integrated"] == 0.0  # tanh(tanh(100)), not tanh(100).
    assert slopes["gate_pressure"] == (0.0 if architecture == "demian_route_ablation" else 1.0)
    assert {"fast_to_message", "carrier_to_slow", "message_readout", "gate_readout"} <= slopes.keys()
    assert point["transition_reconstruction_max_error"] < 1e-12
    assert mapping.adapter.model._step_index == 0
    assert not any(module._forward_hooks for module in mapping.adapter.model.modules())


def test_operating_point_callback_runs_only_at_sample_ticks():
    calls = []
    result = tangent_window(
        lambda state, value: state,
        lambda state: state,
        torch.ones(2).double(),
        [torch.zeros(1).double()] * 65,
        direction_seed=1701,
        operating_point=lambda z, x: calls.append(1) or {},
    )
    assert result["status"] == "ok"
    assert len(calls) == 3


@pytest.mark.parametrize("floor", [0.0, -1.0, float("nan")])
def test_operating_point_rejects_invalid_slope_floor(floor):
    mapping = StateMap(ClosedLoopModelAdapter(architecture="rnn", seed=94, hidden_size=8))
    with pytest.raises(ValueError, match="floor"):
        mapping.operating_point(mapping.initial_state, torch.zeros(1, 8).double(), floor)
