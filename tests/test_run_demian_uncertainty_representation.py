from development import run_demian_uncertainty_representation


def test_representation_rows_capture_only_pre_resolution_state_and_evaluator_labels() -> None:
    rows = run_demian_uncertainty_representation.collect_pre_resolution_rows({
        "hidden_size": 16,
        "model_seeds": [94],
        "episode_seeds": [11, 12],
        "delay_steps": [2],
        "reliability_schedules": [[0.75, 0.5]],
    })

    assert len(rows) == 4
    assert {row["checkpoint_phase"] for row in rows} == {"delay"}
    assert {row["target"] for row in rows} == {"log_odds", "entropy"}
    assert all(
        len(row["features"]["full_state"])
        == sum(len(row["features"][channel]) for channel in ("fast", "slow", "control", "message", "carrier", "gate"))
        for row in rows
    )


def test_representation_protocol_scores_heldout_sequences_per_model_seed() -> None:
    payload = run_demian_uncertainty_representation.run_protocol({
        "hidden_size": 16,
        "model_seeds": [94],
        "episode_seeds": list(range(11, 21)),
        "delay_steps": [2],
        "reliability_schedules": [[0.75, 0.5], [0.95, 0.75]],
        "heldout_fraction": 0.3,
        "ridge_alpha": 0.001,
    })

    assert payload["integrity"]["all_finite"] is True
    assert {row["feature_set"] for row in payload["scores"]} >= {"full_state", "current_observation", "latest_observation"}
    assert {row["model_seed"] for row in payload["scores"]} == {94}
