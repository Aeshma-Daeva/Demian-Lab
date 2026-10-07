"""Tests for native-v9 AFP causal state surgery."""

from __future__ import annotations

from development.afp_v9_causal_followup import CausalConfig, run_experiment


def test_small_causal_probe_preserves_surface_and_full_clone() -> None:
    config = CausalConfig(
        hidden_size=8,
        checkpoint_step=20,
        stale_lags=(2, 4, 6, 8),
        future_steps=4,
        seeds=(104,),
        candidate_seeds=(104,),
        impulse_scale=0.05,
    )
    payload = run_experiment(config)

    rows = payload["rows"]
    assert rows

    for row in rows:
        assert row["initial_surface_gap"] <= 1e-10
        if row["intervention"] == "full_clone":
            assert row["mean_gap"] <= 1e-10
            assert row["max_gap"] <= 1e-10

    interventions = {row["intervention"] for row in rows}
    assert interventions == {
        "full_clone",
        "surface_only",
        "slow_reset",
        "control_reset",
        "stale_2",
        "stale_4",
        "stale_6",
        "stale_8",
    }
    assert {row["future_condition"] for row in rows} == {
        "autonomous",
        "shared_impulse",
    }
