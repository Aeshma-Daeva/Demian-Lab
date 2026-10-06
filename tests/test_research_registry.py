"""Validation contracts for the concise Demian research registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from development.research_registry import (
    ResearchRegistry,
    export_registry_snapshot,
    load_registry,
    validate_registry,
)


REGISTRY_PATH = Path("data/RESEARCH_REGISTRY.json")


def _payload() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def test_checked_in_registry_links_hypotheses_to_evidence() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert validate_registry(REGISTRY_PATH) == registry
    assert {item.id for item in registry.hypotheses} == {
        "H-AFP-002",
        "H-GATE-001",
        "H-CONT-001",
        "H-ROUTE-001",
        "H-ROUTE-002",
        "H-ROUTE-003",
    }
    afp = next(item for item in registry.hypotheses if item.id == "H-AFP-002")
    afp_result = next(item for item in registry.results if item.id in afp.result_ids)
    assert afp.evidence_status == "inconclusive"
    assert afp_result.evaluable is False
    assert afp_result.outcome == "unevaluable"
    assert afp_result.missing_evaluation_conditions == ["operational_surface_convergence"]


def test_registry_rejects_null_as_hypothesis_status() -> None:
    payload = _payload()
    payload["hypotheses"][0]["evidence_status"] = "null"

    with pytest.raises(ValidationError):
        ResearchRegistry.model_validate(payload)


def test_registry_rejects_unresolved_result_reference() -> None:
    payload = _payload()
    payload["hypotheses"][0]["result_ids"].append("R-MISSING")

    with pytest.raises(ValidationError, match="unknown result"):
        ResearchRegistry.model_validate(payload)


def test_registry_rejects_unevaluable_null_result() -> None:
    payload = _payload()
    result = next(item for item in payload["results"] if item["id"] == "R-AFP-BASIN-001")
    result["outcome"] = "null"

    with pytest.raises(ValidationError, match="unevaluable"):
        ResearchRegistry.model_validate(payload)


def test_registry_requires_manual_status_basis() -> None:
    payload = _payload()
    payload["hypotheses"][0]["status_basis"]["method"] = "automatic_metric_rule"

    with pytest.raises(ValidationError):
        ResearchRegistry.model_validate(payload)


def test_registry_marks_checkpoint_rows_as_dependent_observations() -> None:
    payload = _payload()
    experiment = next(item for item in payload["experiments"] if item["id"] == "E-BASIN-001")
    experiment["replication_scope"]["dependent_observations"] = []

    with pytest.raises(ValidationError, match="dependent checkpoint"):
        ResearchRegistry.model_validate(payload)


def test_snapshot_export_is_valid_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    export_registry_snapshot(REGISTRY_PATH, first)
    export_registry_snapshot(REGISTRY_PATH, second)

    assert first.read_bytes() == second.read_bytes()
    snapshot = json.loads(first.read_text())
    assert snapshot["metadata"]["authority"] == "Demian"
    assert snapshot["metadata"]["status_policy"] == "manual_review_only"
    assert len(snapshot["hypotheses"]) == 6


def test_invalid_registry_is_not_exported(tmp_path: Path) -> None:
    payload = _payload()
    payload["hypotheses"][0]["result_ids"].append("R-MISSING")
    source = tmp_path / "invalid.json"
    destination = tmp_path / "snapshot.json"
    source.write_text(json.dumps(payload))

    with pytest.raises(ValidationError, match="unknown result"):
        export_registry_snapshot(source, destination)

    assert not destination.exists()


def test_results_separate_observation_from_interpretation() -> None:
    registry = load_registry(REGISTRY_PATH)

    for result in registry.results:
        assert result.observation
        assert result.interpretation


def test_registry_rejects_duplicate_ids_within_a_collection() -> None:
    payload = _payload()
    payload["results"].append(payload["results"][0].copy())

    with pytest.raises(ValidationError, match="globally unique"):
        ResearchRegistry.model_validate(payload)


def test_status_evidence_must_be_considered_and_linked() -> None:
    payload = _payload()
    hypothesis = next(item for item in payload["hypotheses"] if item["id"] == "H-AFP-002")
    hypothesis["status_basis"]["supporting_result_ids"] = ["R-GATE-REGIME-001"]

    with pytest.raises(ValidationError, match="supporting and conflicting"):
        ResearchRegistry.model_validate(payload)


def test_hypothesis_and_result_links_must_be_reciprocal() -> None:
    payload = _payload()
    result = next(item for item in payload["results"] if item["id"] == "R-AFP-BASIN-001")
    result["hypothesis_ids"] = []

    with pytest.raises(ValidationError, match="reciprocal"):
        ResearchRegistry.model_validate(payload)


def test_continuation_result_has_result_specific_scope() -> None:
    registry = load_registry(REGISTRY_PATH)
    result = next(item for item in registry.results if item.id == "R-CONT-RECON-001")

    assert result.analysis_scope.independent_unit == "parameter_seed"
    assert result.analysis_scope.independent_unit_count == 2
    assert result.analysis_scope.comparison_count == 6
    assert result.analysis_scope.comparison_unit == "reference controls"
    assert "gate modes share parameter seeds" in result.analysis_scope.dependent_observations


def test_route_pilot_remains_unevaluable_for_route_hypotheses() -> None:
    registry = load_registry(REGISTRY_PATH)
    result = next(item for item in registry.results if item.id == "R-ROUTE-PILOT-001")
    route_hypotheses = [
        item for item in registry.hypotheses if item.id.startswith("H-ROUTE-")
    ]

    assert result.evaluable is False
    assert result.outcome == "unevaluable"
    assert result.analysis_scope.independent_unit_count == 3
    assert result.analysis_scope.comparison_count == 96
    assert set(result.missing_evaluation_conditions) == {
        "matched_route_intervention",
        "surface_matched_comparison",
        "removal_rescue_test",
    }
    assert all(item.evidence_status == "untested" for item in route_hypotheses)
    assert all(item.result_ids == [result.id] for item in route_hypotheses)
