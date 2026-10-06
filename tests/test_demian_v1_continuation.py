"""Tests for six-channel continuation controls."""

from __future__ import annotations

import pytest

from development.demian_v1_gate_state import V1_CHANNELS
from development.demian_v1_measurement import V1TraceConfig
from development.demian_v1_measurement import run_v1_measurement
from development.probe_demian_v1_continuation import run_v1_continuation_probe


def test_full_and_sham_resume_are_exact_and_aligned() -> None:
    result = run_v1_continuation_probe(
        V1TraceConfig(seed=94, hidden_size=8, steps=12),
        pause_steps=6,
        resume_steps=6,
    )

    assert result["control_steps"] == [7, 12]
    for name in ("full_checkpoint", "sham"):
        assert result["arms"][name]["mean_step_gap_vs_uninterrupted"] < 1e-7
        assert result["arms"][name]["step_index_start"] == 6
        assert result["arms"][name]["update_mode"] == "active"


def test_surface_random_and_channel_controls_are_explicit() -> None:
    result = run_v1_continuation_probe(
        V1TraceConfig(seed=95, hidden_size=8, steps=12),
        pause_steps=6,
        resume_steps=6,
    )

    assert result["arms"]["body_surface"]["surface_reconstruction_error"] < 1e-6
    assert result["arms"]["random_norm_matched"]["max_norm_mismatch"] < 1e-5
    assert all(f"{name}_only" in result["arms"] for name in V1_CHANNELS)


def test_reachable_surface_match_reports_provenance_and_residual() -> None:
    result = run_v1_continuation_probe(
        V1TraceConfig(seed=95, hidden_size=8, steps=12),
        pause_steps=6,
        resume_steps=6,
    )

    reachable = result["arms"]["reachable_surface_match"]
    assert 0 <= reachable["source_step"] < result["pause_steps"]
    assert reachable["surface_match_error"] >= 0.0
    assert reachable["surface_match_within_tolerance"] is False
    assert reachable["surface_match_tolerance"] == 1e-7
    assert "surface_match_exact" not in reachable
    assert reachable["step_index_start"] == result["pause_steps"]
    assert result["arms"]["full_checkpoint"]["mean_step_gap_vs_uninterrupted"] < 1e-7
    assert result["arms"]["sham"]["mean_step_gap_vs_uninterrupted"] < 1e-7


def test_perturbed_history_is_replayed_in_continuation_reference() -> None:
    config = V1TraceConfig(
        seed=96,
        hidden_size=8,
        steps=12,
        history_condition="surface_perturbed",
        perturb_step=3,
        perturb_scale=0.05,
    )
    trace = run_v1_measurement(config)

    result = run_v1_continuation_probe(config, pause_steps=6, resume_steps=6)

    assert result["reference_final_surface"] == pytest.approx(trace.surfaces[-1], abs=1e-7)


@pytest.mark.parametrize("pause,resume", [(0, 6), (6, 2)])
def test_continuation_rejects_invalid_intervals(pause: int, resume: int) -> None:
    with pytest.raises(ValueError):
        run_v1_continuation_probe(
            V1TraceConfig(seed=1, hidden_size=8, steps=12),
            pause_steps=pause,
            resume_steps=resume,
        )
