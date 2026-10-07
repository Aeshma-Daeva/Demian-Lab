"""Smoke tests for the Demian AFP lineage probe."""

from __future__ import annotations

import math

from development.afp_lineage_probe import LineageConfig, run_experiment


def test_small_lineage_probe_is_complete_and_finite() -> None:
    config = LineageConfig(
        hidden_size=8,
        steps=10,
        tail_steps=4,
        seeds=(94,),
        perturb_step=5,
        perturb_scale=0.1,
    )
    result = run_experiment(config)

    expected = {
        "native_v2",
        "native_v3",
        "native_v8",
        "native_v9",
        "v9_5ch_default",
        "v9_5ch_accumulator",
        "v9_5ch_accumulator_zero_init",
        "demian_v1",
    }
    assert set(result["variants"]) == expected

    for variant in result["variants"].values():
        for arm in ("clean", "perturbed"):
            rows = variant[arm]["rows"]
            assert len(rows) == 1
            row = rows[0]
            assert row["surface_dim"] > 0
            assert row["latent_dim"] >= row["surface_dim"]
            assert math.isfinite(row["surface_rel_velocity"])
            assert math.isfinite(row["latent_rel_velocity"])
            assert 0.0 <= variant[arm]["aggregate"]["strict_afp_fraction"] <= 1.0
            assert variant[arm]["relative_sensitivity"]
            assert variant[arm]["absolute_sensitivity"]
            assert math.isfinite(row["absolute_latent_surface_delta_ratio"])
            assert math.isfinite(row["tail_latent_norm_slope"])
