"""Finite-window and paired-sampling tests for the basin-horizon pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import pyarrow.parquet as pq
import torch

from development.basin_horizon_pilot import (
    classify_checkpoint,
    perturb_reference_state,
    run_basin_horizon_pilot,
    write_pilot_artifacts,
)
from development.demian_v1_gate_state import DemianV1GateState
from development.lab_schemas import BasinHorizonPilotSpec, FiniteWindowProtocol


SPEC_PATH = Path("docs/superpowers/specs/2026-10-05-demian-basin-horizon-pilot.json")


def _protocol(**changes) -> FiniteWindowProtocol:
    values = {
        "surface_delta_tolerance": 0.01,
        "surface_drift_tolerance": 0.0001,
        "surface_escape_tolerance": 0.02,
        "settling_dwell_steps": 3,
        "persistence_steps": 5,
        "checkpoints": [8, 10],
        "censoring": "right_censored_when_checkpoint_ends_before_required_window",
        "terminology": "unsettled_is_not_asymptotic_nonconvergence",
    }
    values.update(changes)
    return FiniteWindowProtocol.model_validate(values)


def _circle(steps: int) -> list[list[float]]:
    return [[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)] for i in range(steps)]


def test_checkpoint_requires_complete_post_settling_window() -> None:
    surface = [[0.0] for _ in range(10)]
    hidden = _circle(len(surface))

    censored = classify_checkpoint(surface, hidden, checkpoint=8, protocol=_protocol())
    complete = classify_checkpoint(surface, hidden, checkpoint=10, protocol=_protocol())

    assert censored["surface_status"] == "operationally_converged"
    assert censored["settlement_step"] == 4
    assert censored["censoring"] == "right_censored"
    assert censored["afp_status"] == "not_evaluable"
    assert complete["censoring"] == "complete"
    assert complete["internal_regime"] == "internal_structured_persistent"
    assert complete["afp_status"] == "afp_v2_candidate"


def test_checkpoint_reports_unsettled_and_surface_escape_separately() -> None:
    unsettled_surface = [[float(i)] for i in range(8)]
    escaped_surface = [[0.0], [0.0], [0.0], [0.0], [0.009], [0.1], [0.2], [0.3]]
    hidden = _circle(8)

    unsettled = classify_checkpoint(unsettled_surface, hidden, 8, _protocol())
    escaped = classify_checkpoint(escaped_surface, hidden, 8, _protocol())

    assert unsettled["surface_status"] == "unsettled_at_checkpoint"
    assert unsettled["settlement_step"] is None
    assert escaped["surface_status"] == "escaped_after_settlement"
    assert escaped["escape_step"] == 6
    assert escaped["afp_status"] == "not_afp_v2"


def test_constant_small_surface_drift_never_settles() -> None:
    surface = [[0.005 * step] for step in range(64)]
    hidden = _circle(64)

    result = classify_checkpoint(
        surface,
        hidden,
        checkpoint=64,
        protocol=_protocol(checkpoints=[64]),
    )

    assert result["surface_status"] == "unsettled_at_checkpoint"
    assert result["afp_status"] == "not_evaluable"


def test_escape_is_distance_from_settled_region_not_only_a_large_step() -> None:
    surface = [[0.0], [0.0], [0.0], [0.0]] + [[0.005 * step] for step in range(1, 9)]
    hidden = _circle(len(surface))

    result = classify_checkpoint(
        surface,
        hidden,
        checkpoint=len(surface),
        protocol=_protocol(checkpoints=[len(surface)]),
    )

    assert result["surface_status"] == "escaped_after_settlement"
    assert result["escape_step"] == 9


def test_internal_classifier_uses_complete_persistence_window() -> None:
    surface = [[0.0] for _ in range(60)]
    hidden = [[0.0, 0.0] for _ in range(28)] + _circle(32)
    protocol = _protocol(persistence_steps=48, checkpoints=[60])

    result = classify_checkpoint(surface, hidden, checkpoint=60, protocol=protocol)

    assert result["censoring"] == "complete"
    assert result["internal_regime"] == "internal_transient"
    assert result["internal_classifier_config"]["tail_steps"] == 49
    assert result["afp_status"] == "not_afp_v2"


def test_channel_normalized_sphere_perturbations_are_paired_and_finite() -> None:
    state = (
        torch.tensor([[1.0, -1.0]]),
        torch.zeros(1, 2),
        torch.tensor([[0.5, 0.25]]),
        torch.ones(1, 2),
        torch.tensor([[2.0, -2.0]]),
        torch.tensor([[0.1, -0.1]]),
    )

    reference = perturb_reference_state(state, 0.0, None, 1e-6)
    small = perturb_reference_state(state, 0.01, 7001, 1e-6)
    large = perturb_reference_state(state, 0.1, 7001, 1e-6)

    assert all(torch.equal(left, right) for left, right in zip(reference, state, strict=True))
    for original, small_channel, large_channel in zip(state, small, large, strict=True):
        small_delta = small_channel - original
        large_delta = large_channel - original
        assert torch.isfinite(small_channel).all()
        assert torch.isfinite(large_channel).all()
        torch.testing.assert_close(large_delta, small_delta * 10.0, rtol=1e-4, atol=1e-6)


def _small_spec() -> BasinHorizonPilotSpec:
    payload = json.loads(SPEC_PATH.read_text())
    payload["system"]["hidden_size"] = 4
    payload["replication"]["model_seeds"] = [1]
    payload["selection_control"]["pilot_seeds"] = [1]
    payload["selection_control"]["held_out_confirmation_seeds"] = []
    payload["conditions"]["architecture_modes"] = ["active"]
    payload["basin_probe"].update(
        reference_steps=3,
        radii=[0.0, 0.1],
        direction_seeds=[7],
    )
    payload["finite_window_protocol"].update(
        checkpoints=[8, 10],
        settling_dwell_steps=2,
        persistence_steps=5,
    )
    return BasinHorizonPilotSpec.model_validate(payload)


def test_pilot_runs_each_trajectory_once_and_preserves_dependence(monkeypatch) -> None:
    calls = 0
    original = DemianV1GateState.step

    def counted(self, state):
        nonlocal calls
        calls += 1
        return original(self, state)

    monkeypatch.setattr(DemianV1GateState, "step", counted)

    payload = run_basin_horizon_pilot(_small_spec())

    assert calls == 3 + 2 * 10
    assert len(payload["runs"]) == 2
    zero = next(row for row in payload["runs"] if row["radius"] == 0.0)
    perturbed = next(row for row in payload["runs"] if row["radius"] > 0.0)
    assert zero["direction_seed"] is None
    assert perturbed["direction_seed"] == 7
    assert perturbed["dependency_group_id"].endswith("direction=7")
    assert [row["checkpoint"] for row in perturbed["checkpoints"]] == [8, 10]
    assert perturbed["steps_run"] == 10


def test_artifact_writer_emits_valid_json_and_one_parquet_row_per_checkpoint(tmp_path) -> None:
    payload = run_basin_horizon_pilot(_small_spec())

    artifacts = write_pilot_artifacts(payload, tmp_path)

    summary_text = artifacts["summary"].read_text()
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    saved = json.loads(summary_text)
    assert saved["run_count"] == 2
    assert saved["aggregates"]["regime_frequency"]
    table = pq.read_table(artifacts["checkpoints"])
    assert table.num_rows == 4
    assert {
        "trajectory_id",
        "dependency_group_id",
        "radius",
        "checkpoint",
        "surface_status",
        "internal_regime",
        "afp_status",
    }.issubset(table.column_names)
