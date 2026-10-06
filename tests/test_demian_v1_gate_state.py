"""Tests for the Demian v1 explicit gate-state substrate."""

from __future__ import annotations

import torch
import pytest

from development.demian_v1_gate_state import (
    V1_ROUTE_CATALOG,
    V1_CHANNELS,
    DemianV1GateState,
    V1RuntimeSnapshot,
    capture_v1_runtime,
    clamp_v1_channel,
    compare_v1_resume,
    match_v1_surface,
    restore_v1_runtime,
    run_v1_trace,
    surface_only_resume_state,
)


def test_route_catalog_covers_six_channels_and_explicit_cross_channel_paths() -> None:
    route_ids = {route.id for route in V1_ROUTE_CATALOG}
    participating_channels = {
        channel
        for route in V1_ROUTE_CATALOG
        for channel in (*route.sources, route.target)
        if channel in V1_CHANNELS
    }

    assert participating_channels == set(V1_CHANNELS)
    assert {
        "fast_to_message",
        "message_to_carrier",
        "carrier_to_slow",
        "control_to_fast",
        "message_to_fast",
        "carrier_to_fast",
        "gate_modulates_message_to_carrier",
        "gate_modulates_carrier_to_slow",
        "gate_modulates_surface_routes",
    } <= route_ids


def test_route_tracing_does_not_change_trajectory() -> None:
    torch.manual_seed(90)
    untraced = DemianV1GateState(hidden_size=8, trace_routes=False)
    traced = DemianV1GateState(hidden_size=8, trace_routes=True)
    traced.load_state_dict(untraced.state_dict())
    torch.manual_seed(91)
    initial = untraced.initial_state(1, torch.device("cpu"))
    traced_initial = tuple(component.clone() for component in initial)

    untraced_next = untraced.step(initial)
    traced_next = traced.step(traced_initial)

    assert all(
        torch.equal(left, right)
        for left, right in zip(untraced_next, traced_next, strict=True)
    )
    assert untraced.route_trace() is None
    assert traced.route_trace() is not None


@pytest.mark.parametrize("override", ["gate_disabled", "gate_frozen"])
def test_route_tracing_rejects_discarded_gate_branch(override: str) -> None:
    kwargs = {override: True}

    with pytest.raises(ValueError, match="active gate update rule"):
        DemianV1GateState(hidden_size=8, trace_routes=True, **kwargs)


def test_recorded_integration_terms_reconstruct_all_six_targets() -> None:
    torch.manual_seed(92)
    model = DemianV1GateState(hidden_size=8, trace_routes=True)
    state = model.initial_state(1, torch.device("cpu"))

    next_state = model.step(state)
    trace = model.route_trace()

    assert trace is not None
    assert trace.step == 1
    for channel, expected in zip(V1_CHANNELS, next_state, strict=True):
        terms = [
            value
            for route_id, value in trace.values.items()
            if trace.catalog[route_id].target == channel
            and trace.catalog[route_id].injection_point == "target_state"
        ]
        assert terms
        reconstructed = torch.stack(terms).sum(dim=0)
        assert torch.allclose(reconstructed, expected, atol=1e-7, rtol=1e-7)


def test_surface_match_preserves_internal_channels() -> None:
    torch.manual_seed(91)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))
    target = model.state_vector(state) + 0.125

    matched = match_v1_surface(model, state, target)

    assert torch.allclose(model.state_vector(matched), target, atol=1e-6, rtol=1e-6)
    assert all(torch.equal(matched[i], state[i]) for i in range(1, 6))


def test_runtime_snapshot_restores_cloned_state_and_metadata() -> None:
    torch.manual_seed(92)
    model = DemianV1GateState(hidden_size=8, gate_frozen=True)
    state = model.initial_state(1, torch.device("cpu"))
    state = model.step(state)
    snapshot = capture_v1_runtime(model, state)
    model._step_index = 99
    model._frozen_gate = None

    restored = restore_v1_runtime(model, snapshot)

    assert isinstance(snapshot, V1RuntimeSnapshot)
    assert model._step_index == 1
    assert torch.equal(model._frozen_gate, snapshot.frozen_gate)
    assert all(torch.equal(left, right) for left, right in zip(restored, snapshot.state))
    assert all(left.data_ptr() != right.data_ptr() for left, right in zip(restored, snapshot.state))


def test_v1_state_has_explicit_gate_channel() -> None:
    torch.manual_seed(94)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))

    assert len(state) == 6
    assert tuple(model.state_components(state)) == V1_CHANNELS
    assert model.state_components(state)["gate"].shape == (1, 8)


def test_surface_only_resume_keeps_surface_and_zeros_internal_capsule() -> None:
    torch.manual_seed(94)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))
    resumed = surface_only_resume_state(model, state)

    assert torch.allclose(resumed[0], model.state_vector(state))
    for channel in resumed[1:]:
        assert torch.count_nonzero(channel) == 0


def test_gate_disabled_clamps_gate_after_trace_step() -> None:
    model = DemianV1GateState(hidden_size=8, gate_disabled=True)

    _, state, metrics = run_v1_trace(model, steps=4, seed=95)

    assert torch.count_nonzero(state[-1]) == 0
    assert metrics[-1]["gate_disabled"] == 1.0


def test_gate_modulates_routes_without_release_vector_metrics() -> None:
    torch.manual_seed(96)
    model = DemianV1GateState(
        hidden_size=8,
        gate_to_message_carrier_scale=0.9,
        gate_to_carrier_slow_scale=0.8,
        gate_to_surface_scale=0.7,
    )
    state = model.initial_state(1, torch.device("cpu"))

    _ = model.step(state)
    aux = model.step_aux()

    assert aux["gate_state_norm"] > 0.0
    assert aux["gate_change_duty"] >= 0.0
    assert aux["gate_to_message_carrier_scale"] != 1.0
    assert aux["gate_to_carrier_slow_scale"] != 1.0
    assert aux["gate_to_surface_scale"] != 1.0
    assert "release_bias_norm" not in aux
    assert "release_open_mean" not in aux
    assert "endogenous_release_norm" not in aux


def test_channel_clamp_validates_known_v1_channels() -> None:
    torch.manual_seed(97)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))

    clamped = clamp_v1_channel(state, "message")

    assert torch.count_nonzero(clamped[V1_CHANNELS.index("message")]) == 0


def test_full_capsule_resume_outperforms_surface_only_resume() -> None:
    model = DemianV1GateState(hidden_size=8)

    full = compare_v1_resume(model, seed=98, pause_steps=6, resume_steps=6, surface_only=False)
    surface = compare_v1_resume(model, seed=98, pause_steps=6, resume_steps=6, surface_only=True)

    assert full.mean_step_gap < 1e-7
    assert surface.mean_step_gap > full.mean_step_gap
