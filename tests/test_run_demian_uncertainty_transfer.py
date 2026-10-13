from development import run_demian_uncertainty_transfer


def test_transfer_corpus_uses_disjoint_train_and_test_order_families() -> None:
    protocol = {
        "train_reliability_schedules": [[0.5, 0.75]],
        "unseen_order_schedules": [[0.75, 0.5]],
        "unseen_length_schedules": [[0.5, 0.75, 0.5]],
        "allocation_seed": 31,
        "unseen_length_history_count": 4,
    }

    corpus = run_demian_uncertainty_transfer.build_transfer_corpus(protocol)

    assert {row["family"] for row in corpus} == {"train", "unseen_order", "unseen_length"}
    assert len([row for row in corpus if row["family"] == "train"]) == 4
    assert len([row for row in corpus if row["family"] == "unseen_order"]) == 4
    assert len([row for row in corpus if row["family"] == "unseen_length"]) == 4
    assert set(tuple(row["reliability_schedule"]) for row in corpus if row["family"] == "train").isdisjoint(
        {tuple(row["reliability_schedule"]) for row in corpus if row["family"] == "unseen_order"}
    )


def test_transfer_protocol_scores_frozen_low_capacity_features_on_both_test_families() -> None:
    payload = run_demian_uncertainty_transfer.run_protocol({
        "hidden_size": 16,
        "model_seeds": [94],
        "delay_steps": [2],
        "train_reliability_schedules": [[0.5, 0.75]],
        "unseen_order_schedules": [[0.75, 0.5]],
        "unseen_length_schedules": [[0.5, 0.75, 0.5]],
        "allocation_seed": 31,
        "unseen_length_history_count": 4,
        "projection_dimensions": [2],
        "projection_seeds": [17],
        "ridge_alpha": 1.0,
    })

    assert payload["integrity"]["all_finite"] is True
    assert {row["evaluation_family"] for row in payload["scores"]} == {"unseen_order", "unseen_length"}
    assert {row["feature_set"] for row in payload["scores"]} >= {
        "projected_full_state", "current_observation", "latest_observation", "schedule_only", "full_state_conditional_shuffle"
    }
    assert all(row["projection_dimension"] == 2 for row in payload["scores"] if row["feature_set"] == "projected_full_state")
