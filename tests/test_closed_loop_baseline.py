"""Unselected closed-loop baseline contract."""

from __future__ import annotations

from development.closed_loop_baseline import run_closed_loop_baseline


def test_unselected_closed_loop_baseline_is_deterministic_and_explicitly_unevaluable() -> None:
    first = run_closed_loop_baseline(seeds=[94, 95], hidden_size=8, steps=5)
    second = run_closed_loop_baseline(seeds=[94, 95], hidden_size=8, steps=5)

    assert first == second
    assert first["experiment"] == "demian-v1-unselected-closed-loop-baseline"
    assert first["replication_scope"]["parameter_seed_count"] == 2
    assert first["replication_scope"]["trajectory_count"] == 2
    assert first["replication_scope"]["independent_unit"] == "parameter_seed"
    assert len(first["runs"]) == 2
    assert 1 <= len(first["runs"][0]["trace"]) <= 5
    assert first["replication_scope"]["dependent_tick_count"] == sum(
        len(run["trace"]) for run in first["runs"]
    )
    assert len(first["runs"][0]["trace"][0]["route_l2"]) == 30
    assert first["evidence_status"]["hypothesis_status"] == "untested"


def test_baseline_sampling_bounds_trace_storage_without_changing_tick_summary() -> None:
    result = run_closed_loop_baseline(
        seeds=[94], hidden_size=8, steps=8, delay_steps=7, sample_every=3
    )

    run = result["runs"][0]
    assert result["replication_scope"]["executed_tick_count"] == 8
    assert result["replication_scope"]["sampled_tick_count"] == 4
    assert [record["tick"] for record in run["trace"]] == [0, 3, 6, 7]
    assert len(run["route_l2_mean"]) == 30
