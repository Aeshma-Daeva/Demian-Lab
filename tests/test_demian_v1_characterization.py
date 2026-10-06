"""Tests for the six-channel Demian characterization campaign."""

from __future__ import annotations

import math

import pytest

from development.afp_v2_characterization import AFPV2Config, classify_trace_result
from development.demian_v1_gate_state import V1_CHANNELS
from development.demian_v1_measurement import V1TraceConfig, V1TraceResult
from development.run_demian_v1_characterization import run_demian_v1_campaign


def test_trace_classification_separates_dynamics_causality_and_utility() -> None:
    surface = [[1.0, -1.0] for _ in range(32)]
    circle = [[math.cos(i * math.pi / 8), math.sin(i * math.pi / 8)] for i in range(32)]
    channels = {name: circle for name in V1_CHANNELS}
    trace = V1TraceResult(
        config=V1TraceConfig(seed=1, hidden_size=2, steps=32),
        surfaces=surface,
        full_states=[row * len(V1_CHANNELS) for row in circle],
        channels=channels,
        metrics=[{} for _ in range(32)],
    )

    result = classify_trace_result(
        trace,
        continuation_gap=0.1,
        config=AFPV2Config(hidden_delta_min=0.1, continuation_gap_min=0.01),
    )

    assert result["full_state"]["classification"] == "afp_v2_supported"
    assert set(result["channels"]) == set(V1_CHANNELS)
    assert result["functional_relevance"]["continuation_causal"] is True
    assert result["functional_relevance"]["task_utility"] is None


def test_internal_regime_is_independent_of_moving_surface() -> None:
    surface = [[float(i), 0.0] for i in range(32)]
    fixed = [[0.25, -0.25] for _ in range(32)]
    trace = V1TraceResult(
        config=V1TraceConfig(seed=1, hidden_size=2, steps=32),
        surfaces=surface,
        full_states=[row * len(V1_CHANNELS) for row in fixed],
        channels={name: fixed for name in V1_CHANNELS},
        metrics=[{} for _ in range(32)],
    )

    result = classify_trace_result(trace, continuation_gap=None)

    assert result["full_state"]["classification"] == "surface_nonconvergent"
    assert result["internal_regimes"]["full_state"]["classification"] == "internal_fixed"
    assert result["internal_regimes"]["channels"]["gate"]["classification"] == "internal_fixed"


def test_small_campaign_keeps_factors_separate() -> None:
    summary = run_demian_v1_campaign(
        seeds=[1, 2],
        hidden_size=4,
        steps=12,
        perturb_step=6,
        update_modes=("active",),
        history_conditions=("uninterrupted",),
        robustness_tails=(4, 6, 8),
        threshold_multipliers=(1.0,),
    )

    assert {row["parameter_provenance"] for row in summary["runs"]} == {
        "unselected_random",
        "historical_control",
    }
    assert all("history_condition" in row for row in summary["runs"])
    assert all("update_mode" in row and "post_step_clamp" in row for row in summary["runs"])
    assert any(row["system"] == "v9_five_channel" for row in summary["runs"])
    six = next(row for row in summary["runs"] if row["system"] == "demian_v1_gate_state")
    long_slice = six["robustness"]["tail=8:threshold=1"]
    assert long_slice["continuation_evidence"] == "not_run_for_tail"
    assert "numerical_drift_multiplier" in long_slice["varied_thresholds"]
    assert long_slice["effective_config"]["tail_steps"] == 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"seeds": []},
        {"steps": 8, "robustness_tails": (4, 8, 12)},
        {"update_modes": ("unknown",)},
        {"update_modes": ()},
        {"history_conditions": ()},
        {"threshold_multipliers": ()},
        {"threshold_multipliers": (float("nan"),)},
        {"threshold_multipliers": (float("inf"),)},
        {"robustness_tails": (1, 4, 8)},
        {"steps": 12, "perturb_step": 13, "robustness_tails": (4, 6, 8)},
    ],
)
def test_campaign_rejects_invalid_configuration(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "seeds": [1],
        "hidden_size": 4,
        "steps": 12,
        "perturb_step": 6,
        "robustness_tails": (4, 6, 8),
        "threshold_multipliers": (1.0,),
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        run_demian_v1_campaign(**values)
