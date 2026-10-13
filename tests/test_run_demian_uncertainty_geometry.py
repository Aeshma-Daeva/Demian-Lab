from development import run_demian_uncertainty_geometry


def test_geometry_protocol_reports_paired_full_state_and_channel_metrics() -> None:
    payload = run_demian_uncertainty_geometry.run_protocol({
        "hidden_size": 16,
        "model_seeds": [94],
        "base_delay_steps": 2,
        "train_reliability_schedules": [[0.5, 0.75]],
        "heldout_reliability_schedules": [[0.75, 0.5]],
        "allocation_seed": 31,
    })

    assert payload["integrity"]["all_finite"] is True
    assert {row["comparison"] for row in payload["pair_metrics"]} == {"extra_delay", "neutral_suffix"}
    assert {row["channel"] for row in payload["channel_metrics"]} >= {"full_state", "fast", "slow", "carrier"}
