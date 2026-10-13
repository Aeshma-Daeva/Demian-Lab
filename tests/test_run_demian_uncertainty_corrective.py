from development import run_demian_uncertainty_corrective
from development.demian_uncertainty_world import EvidenceObservation, reference_belief


def test_balanced_corpus_equalizes_log_odds_support_across_schedules() -> None:
    specs = run_demian_uncertainty_corrective.build_balanced_corpus({
        "reliability_schedules": [[0.5, 0.75], [0.75, 0.5]],
        "allocation_seed": 31,
    })

    assert len(specs) == 8
    by_schedule = {}
    for spec in specs:
        by_schedule.setdefault(spec["schedule_id"], []).append(spec)
    assert {len(items) for items in by_schedule.values()} == {4}
    assert all({item["target"] for item in items} == {0, 1} for items in by_schedule.values())
    supports = []
    for items in by_schedule.values():
        supports.append(sorted(reference_belief([EvidenceObservation(index, "evidence", symbol, reliability) for index, (symbol, reliability) in enumerate(item["evidence"])])["log_odds"] for item in items))
    assert supports[0] == supports[1]


def test_stratified_split_keeps_each_schedule_target_cell_in_both_arms() -> None:
    rows = [
        {"episode_id": index, "schedule_id": schedule, "target": target}
        for schedule in range(2)
        for target in range(2)
        for index in range(schedule * 20 + target * 10, schedule * 20 + target * 10 + 10)
    ]
    train, heldout = run_demian_uncertainty_corrective.stratified_episode_split(rows, split_seed=9, heldout_fraction=0.3)

    assert {row["episode_id"] for row in train}.isdisjoint({row["episode_id"] for row in heldout})
    for schedule in range(2):
        for target in range(2):
            assert any(row["schedule_id"] == schedule and row["target"] == target for row in train)
            assert any(row["schedule_id"] == schedule and row["target"] == target for row in heldout)


def test_corrective_protocol_includes_schedule_only_control() -> None:
    payload = run_demian_uncertainty_corrective.run_protocol({
        "hidden_size": 16,
        "model_seeds": [94],
        "delay_steps": [2],
        "reliability_schedules": [[0.5, 0.75], [0.75, 0.5]],
        "allocation_seed": 31,
        "split_seeds": [3],
        "heldout_fraction": 0.5,
        "ridge_alpha": 0.001,
    })

    assert payload["integrity"]["all_finite"] is True
    assert {row["feature_set"] for row in payload["scores"]} >= {"full_state", "current_observation", "latest_observation", "schedule_only", "full_state_shuffled_label"}
