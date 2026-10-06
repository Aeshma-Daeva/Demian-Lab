#!/usr/bin/env python3
"""Validated research records for concise Demian evidence review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


EvidenceStatus = Literal[
    "untested",
    "inconclusive",
    "preliminary_support",
    "supported",
    "conflicting",
    "contradicted",
]
Lifecycle = Literal["active", "superseded", "retired"]
ResultOutcome = Literal["observed", "null", "mixed", "unevaluable"]


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StatusBasis(RegistryModel):
    method: Literal["manual_review"]
    rule: str
    considered_result_ids: list[str]
    supporting_result_ids: list[str]
    conflicting_result_ids: list[str]
    reviewer: str
    reviewed_on: str


class StatusHistoryEntry(RegistryModel):
    evidence_status: EvidenceStatus
    lifecycle: Lifecycle
    changed_on: str
    reviewer: str
    basis: str


class HypothesisRecord(RegistryModel):
    id: str = Field(pattern=r"^H-[A-Z]+-[0-9]{3}$")
    statement: str
    operational_prediction: str
    evaluation_conditions: list[str]
    evidence_status: EvidenceStatus
    lifecycle: Lifecycle
    status_basis: StatusBasis
    status_history: list[StatusHistoryEntry]
    result_ids: list[str]
    alternative_explanations: list[str]
    falsifiers: list[str]
    next_test: str


class ReplicationScope(RegistryModel):
    parameter_seed_count: PositiveInt
    initial_state_count: PositiveInt
    input_seed_count: int = Field(ge=0)
    history_condition_count: PositiveInt
    trajectory_count: PositiveInt
    checkpoint_row_count: PositiveInt
    independent_unit: str
    dependent_observations: list[str]

    @model_validator(mode="after")
    def record_dependent_checkpoints(self) -> ReplicationScope:
        if self.checkpoint_row_count > self.trajectory_count and not any(
            "checkpoint" in item for item in self.dependent_observations
        ):
            raise ValueError("dependent checkpoint observations must be explicit")
        return self


class ExperimentRecord(RegistryModel):
    id: str = Field(pattern=r"^E-[A-Z]+-[0-9]{3}$")
    name: str
    system: str
    protocol_version: str
    replication_scope: ReplicationScope
    artifact_ids: list[str]


class ControlRecord(RegistryModel):
    id: str = Field(pattern=r"^C-[A-Z-]+-[0-9]{3}$")
    name: str
    semantics: str
    limitations: list[str]
    artifact_ids: list[str]


class AnalysisScope(RegistryModel):
    independent_unit: str
    independent_unit_count: PositiveInt
    comparison_count: PositiveInt
    comparison_unit: str
    dependent_observations: list[str]
    scope_note: str


class ResultRecord(RegistryModel):
    id: str = Field(pattern=r"^R-[A-Z-]+-[0-9]{3}$")
    hypothesis_ids: list[str]
    experiment_id: str
    evaluable: bool
    outcome: ResultOutcome
    observation: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    analysis_scope: AnalysisScope
    evaluation_conditions_met: list[str]
    missing_evaluation_conditions: list[str]
    effect_summary: dict[str, Any]
    uncertainty: list[str]
    exclusions: list[str]
    control_ids: list[str]
    artifact_ids: list[str]

    @model_validator(mode="after")
    def keep_evaluability_distinct_from_null(self) -> ResultRecord:
        if not self.evaluable:
            if self.outcome != "unevaluable" or not self.missing_evaluation_conditions:
                raise ValueError("unevaluable results require outcome=unevaluable and missing conditions")
        elif self.outcome == "unevaluable":
            raise ValueError("evaluable results cannot use the unevaluable outcome")
        return self


class ArtifactRecord(RegistryModel):
    id: str = Field(pattern=r"^A-[A-Z-]+-[0-9]{3}$")
    repository: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    path: str
    media_type: str
    url: str


class RegistryMetadata(RegistryModel):
    schema_version: PositiveInt
    registry_id: str
    protocol_version: str
    generated_on: str
    authority: Literal["Demian"]
    source_repository: str
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    status_policy: Literal["manual_review_only"]


class ResearchRegistry(RegistryModel):
    metadata: RegistryMetadata
    hypotheses: list[HypothesisRecord]
    experiments: list[ExperimentRecord]
    controls: list[ControlRecord]
    results: list[ResultRecord]
    artifacts: list[ArtifactRecord]

    @model_validator(mode="after")
    def resolve_references(self) -> ResearchRegistry:
        records = {
            "hypothesis": self.hypotheses,
            "experiment": self.experiments,
            "control": self.controls,
            "result": self.results,
            "artifact": self.artifacts,
        }
        all_ids = [item.id for values in records.values() for item in values]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("registry IDs must be globally unique")
        collections = {kind: {item.id for item in values} for kind, values in records.items()}
        result_by_id = {item.id: item for item in self.results}
        hypothesis_by_id = {item.id: item for item in self.hypotheses}

        def require(kind: str, values: list[str], owner: str) -> None:
            missing = sorted(set(values) - collections[kind])
            if missing:
                raise ValueError(f"{owner} references unknown {kind}: {', '.join(missing)}")

        for hypothesis in self.hypotheses:
            require("result", hypothesis.result_ids, hypothesis.id)
            require("result", hypothesis.status_basis.considered_result_ids, hypothesis.id)
            require("result", hypothesis.status_basis.supporting_result_ids, hypothesis.id)
            require("result", hypothesis.status_basis.conflicting_result_ids, hypothesis.id)
            considered = set(hypothesis.status_basis.considered_result_ids)
            classified = set(hypothesis.status_basis.supporting_result_ids) | set(
                hypothesis.status_basis.conflicting_result_ids
            )
            if not classified.issubset(considered):
                raise ValueError(f"{hypothesis.id} supporting and conflicting results must be considered")
            if not considered.issubset(hypothesis.result_ids):
                raise ValueError(f"{hypothesis.id} status basis must use linked results")
            for result_id in hypothesis.result_ids:
                if hypothesis.id not in result_by_id[result_id].hypothesis_ids:
                    raise ValueError(f"{hypothesis.id} and {result_id} links must be reciprocal")
        for experiment in self.experiments:
            require("artifact", experiment.artifact_ids, experiment.id)
        for control in self.controls:
            require("artifact", control.artifact_ids, control.id)
        for result in self.results:
            require("hypothesis", result.hypothesis_ids, result.id)
            require("experiment", [result.experiment_id], result.id)
            require("control", result.control_ids, result.id)
            require("artifact", result.artifact_ids, result.id)
            for hypothesis_id in result.hypothesis_ids:
                if result.id not in hypothesis_by_id[hypothesis_id].result_ids:
                    raise ValueError(f"{result.id} and {hypothesis_id} links must be reciprocal")
        return self


def load_registry(path: Path) -> ResearchRegistry:
    return ResearchRegistry.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_registry(path: Path) -> ResearchRegistry:
    return load_registry(path)


def export_registry_snapshot(source: Path, destination: Path) -> Path:
    """Validate and write a deterministic public registry snapshot."""
    registry = load_registry(source)
    payload = json.dumps(registry.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return destination
