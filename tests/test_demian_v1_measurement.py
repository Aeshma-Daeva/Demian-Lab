"""Tests for typed six-channel Demian measurement traces."""

from __future__ import annotations

import pytest

from development.demian_v1_gate_state import V1_CHANNELS
from development.demian_v1_measurement import V1TraceConfig, run_v1_measurement


def test_measurement_records_complete_six_channel_trace() -> None:
    config = V1TraceConfig(seed=94, hidden_size=8, steps=4)

    result = run_v1_measurement(config)

    assert len(result.surfaces) == 4
    assert len(result.full_states) == 4
    assert set(result.channels) == set(V1_CHANNELS)
    assert all(len(result.channels[name]) == 4 for name in V1_CHANNELS)
    assert len(result.metrics) == 4
    assert result.config == config


@pytest.mark.parametrize("update_mode", ["gate_disabled", "gate_frozen"])
def test_update_intervention_differs_from_post_step_gate_clamp(update_mode: str) -> None:
    update = run_v1_measurement(
        V1TraceConfig(seed=95, hidden_size=8, steps=4, update_mode=update_mode)
    )
    clamped = run_v1_measurement(
        V1TraceConfig(seed=95, hidden_size=8, steps=4, post_step_clamp="gate")
    )

    assert update.config.update_mode != clamped.config.update_mode
    assert update.config.post_step_clamp != clamped.config.post_step_clamp
    assert update.full_states[-1] != clamped.full_states[-1]


def test_surface_perturbation_is_reconstructed_exactly() -> None:
    result = run_v1_measurement(
        V1TraceConfig(
            seed=96,
            hidden_size=8,
            steps=4,
            history_condition="surface_perturbed",
            perturb_step=2,
            perturb_scale=0.05,
        )
    )

    assert result.metrics[1]["surface_reconstruction_error"] < 1e-6


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 2},
        {"parameter_provenance": ""},
        {"update_mode": "unknown"},
        {"post_step_clamp": "unknown"},
        {"perturb_step": 5},
    ],
)
def test_measurement_rejects_invalid_configuration(changes: dict[str, object]) -> None:
    values: dict[str, object] = {"seed": 1, "hidden_size": 8, "steps": 4}
    values.update(changes)

    with pytest.raises(ValueError):
        run_v1_measurement(V1TraceConfig(**values))
