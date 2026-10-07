"""Contracts for the first bounded closed-loop perturbation campaign."""

from __future__ import annotations

from development.closed_loop_perturbation import build_four_group_conditions, run_perturbation_campaign


def test_four_group_conditions_are_disjoint_and_deterministic() -> None:
    first = build_four_group_conditions(list(range(94, 102)), steps=100)
    second = build_four_group_conditions(list(range(94, 102)), steps=100)

    assert first == second
    assert [condition.name for condition in first] == ["baseline", "internal", "environment", "both"]
    assert [len(condition.seeds) for condition in first] == [2, 2, 2, 2]
    assert first[0].internal_amplitude == 0.0
    assert first[1].internal_amplitude == 0.01
    assert first[2].environment_mode == "mixed"
    assert first[3].environment_mode == "mixed"
    assert all(not ticks for ticks in first[0].internal_ticks.values())
    assert all(len(ticks) == 1 for ticks in first[1].internal_ticks.values())
    assert all(len(ticks) == 1 for ticks in first[2].environment_ticks.values())


def test_campaign_records_scheduled_internal_and_environment_disturbances() -> None:
    result = run_perturbation_campaign(
        seeds=list(range(94, 102)), hidden_size=8, steps=16, delay_steps=15, sample_every=8
    )

    assert result["replication_scope"]["trajectory_count"] == 8
    assert [group["name"] for group in result["groups"]] == [
        "baseline",
        "internal",
        "environment",
        "both",
    ]
    assert result["groups"][0]["scheduled_internal_events"] == 0
    assert result["groups"][1]["scheduled_internal_events"] == 2
    assert result["groups"][2]["scheduled_environment_events"] == 2
    assert result["groups"][3]["scheduled_internal_events"] == 2
    assert result["groups"][3]["scheduled_environment_events"] == 2
