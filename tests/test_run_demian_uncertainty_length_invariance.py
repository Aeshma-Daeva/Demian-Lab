from development import run_demian_uncertainty_length_invariance


def test_neutral_cue_corpus_preserves_public_log_odds_and_has_paired_conditions() -> None:
    corpus = run_demian_uncertainty_length_invariance.build_neutral_cue_corpus({
        "train_reliability_schedules": [[0.5, 0.75]],
        "heldout_reliability_schedules": [[0.75, 0.5]],
        "allocation_seed": 31,
        "base_delay_steps": 2,
    })

    assert {row["condition"] for row in corpus} == {"base", "delay_plus_two", "neutral_cue_plus_two"}
    grouped = {}
    for row in corpus:
        grouped.setdefault(row["episode_id"], []).append(row)
    assert {len(rows) for rows in grouped.values()} == {1, 6}
    assert {len(rows) for rows in grouped.values() if rows[0]["family"] == "train"} == {1}
    for rows in grouped.values():
        assert {row["target_value"] for row in rows}
        assert len({row["target_value"] for row in rows}) == 1
        if rows[0]["family"] == "heldout":
            assert sorted(len(row["evidence"]) for row in rows) == [2, 2, 4, 4, 4, 4]


def test_length_invariance_protocol_scores_all_paired_conditions(tmp_path) -> None:
    protocol = {
        "hidden_size": 16,
        "model_seeds": [94],
        "base_delay_steps": 2,
        "train_reliability_schedules": [[0.5, 0.75]],
        "heldout_reliability_schedules": [[0.75, 0.5]],
        "allocation_seed": 31,
        "split_seed": 9,
        "heldout_fraction": 0.5,
        "projection_dimensions": [2],
        "projection_seeds": [17],
        "ridge_alpha": 1.0,
    }
    payload = run_demian_uncertainty_length_invariance.run_protocol(protocol)

    assert payload["integrity"]["all_finite"] is True
    assert {row["evaluation_condition"] for row in payload["scores"]} == {"base", "delay_plus_two", "neutral_cue_plus_two"}
    assert {row["feature_set"] for row in payload["scores"]} >= {
        "projected_full_state", "projected_full_state_l2_normalized", "projected_full_state_condition_normalized", "current_observation", "latest_observation", "schedule_only", "full_state_conditional_shuffle"
    }
    out = tmp_path / "length"
    run_demian_uncertainty_length_invariance.write_artifacts(out, {"protocol_id": "test"}, payload)
    assert (out / "summary.json").is_file()
