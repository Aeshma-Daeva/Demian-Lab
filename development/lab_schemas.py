"""Pydantic contracts for Demian research-lab artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

ArtifactKind = Literal["candidate", "evolution_config", "evolution_summary"]


class FlexibleArtifact(BaseModel):
    """Base model that validates known fields while preserving older artifact extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class Metrics(FlexibleArtifact):
    """Shared metric surface for candidate and summary rows."""

    internal_richness: float | None = None
    channel_separation: float | None = None
    release_duty_cycle: NonNegativeFloat | None = None
    release_geometric_event: float | None = None
    release_causal_divergence: float | None = None
    release_gain_zero_release_causal_divergence: float | None = None
    release_timing_score: float | None = None
    phase_transition_score: float | None = None
    mathematical_curiosity: float | None = None
    geometric_coherence: float | None = None
    regime_bonus: float | None = None
    regimes: dict[str, int] = Field(default_factory=dict)

    @field_validator("release_duty_cycle")
    @classmethod
    def duty_is_fraction(cls, value: float | None) -> float | None:
        if value is not None and value > 1.0:
            raise ValueError("release_duty_cycle must be a fraction in [0, 1]")
        return value


class CandidateArtifact(FlexibleArtifact):
    """Single evolved candidate JSON artifact."""

    id: str
    generation: NonNegativeInt
    index: NonNegativeInt | None = None
    genome: dict[str, Any]
    metrics: Metrics
    reproduction_kind: str
    parent_ids: list[str] = Field(default_factory=list)
    ancestor_ids: list[str] = Field(default_factory=list)
    rank_score: float | None = None
    rank_components: dict[str, float] = Field(default_factory=dict)
    runs: list[dict[str, Any]] = Field(default_factory=list)
    paired_runs: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def candidate_id_matches_generation(self) -> CandidateArtifact:
        if self.id.startswith("gen"):
            prefix = f"gen{int(self.generation):03d}_"
            if not self.id.startswith(prefix):
                raise ValueError(f"candidate id {self.id!r} does not match generation {self.generation}")
        return self


class EvolutionConfigArtifact(FlexibleArtifact):
    """Configuration emitted by evolution scripts."""

    population: NonNegativeInt
    generations: NonNegativeInt
    seed: int
    eval_seeds: list[int]
    hidden_size: NonNegativeInt
    steps: NonNegativeInt
    perturb_step: NonNegativeInt
    perturb_scales: list[float]
    rank: NonNegativeInt
    experiment_version: str

    @model_validator(mode="after")
    def perturb_step_within_run(self) -> EvolutionConfigArtifact:
        if self.perturb_step > self.steps:
            raise ValueError("perturb_step must be <= steps")
        return self


class EvolutionSummaryArtifact(FlexibleArtifact):
    """Cross-island or single-run summary JSON."""

    experiment: str
    candidate_count: NonNegativeInt
    generation_count: NonNegativeInt
    final_generation: NonNegativeInt | None = None
    best_overall: dict[str, Any] | None = None
    final_best: dict[str, Any] | None = None
    per_generation: list[dict[str, Any]] = Field(default_factory=list)


class FiniteWindowProtocol(FlexibleArtifact):
    """Finite-horizon convergence and persistence rules."""

    surface_delta_tolerance: PositiveFloat
    surface_drift_tolerance: PositiveFloat
    surface_escape_tolerance: PositiveFloat
    settling_dwell_steps: PositiveInt
    persistence_steps: PositiveInt
    checkpoints: list[PositiveInt]
    censoring: str
    terminology: str

    @model_validator(mode="after")
    def validate_windows(self) -> FiniteWindowProtocol:
        if self.checkpoints != sorted(set(self.checkpoints)):
            raise ValueError("checkpoints must be strictly increasing and unique")
        if self.surface_escape_tolerance < self.surface_delta_tolerance:
            raise ValueError("surface escape tolerance must be at least the settling tolerance")
        if self.persistence_steps < 5:
            raise ValueError("persistence window must contain at least five transitions")
        return self


class BasinProbeSpec(FlexibleArtifact):
    """Local state-space sampling protocol around saved reference states."""

    reference_steps: PositiveInt
    radii: list[NonNegativeFloat]
    geometry: Literal["sphere"]
    perturbation_norm: Literal["per_channel_rms_normalized_l2"]
    channel_scale_floor: PositiveFloat
    direction_seeds: list[int]
    reuse_directions_across_radii: bool
    reported_quantity: str

    @model_validator(mode="after")
    def validate_sampling(self) -> BasinProbeSpec:
        if not self.radii or self.radii[0] != 0.0:
            raise ValueError("radii must include zero as the first reference condition")
        if self.radii != sorted(set(self.radii)):
            raise ValueError("radii must be increasing and unique")
        if not self.direction_seeds or len(self.direction_seeds) != len(set(self.direction_seeds)):
            raise ValueError("direction seeds must be non-empty and unique")
        return self


class ReplicationSpec(FlexibleArtifact):
    model_seeds: list[int]
    reference_states_per_model: PositiveInt
    exact_replay_role: str
    claim_limit: str


class SelectionControlSpec(FlexibleArtifact):
    pilot_seeds: list[int]
    held_out_confirmation_seeds: list[int]
    rule: str

    @model_validator(mode="after")
    def disjoint_seed_sets(self) -> SelectionControlSpec:
        if set(self.pilot_seeds) & set(self.held_out_confirmation_seeds):
            raise ValueError("pilot and held-out seed sets must be disjoint")
        return self


class BasinHorizonPilotSpec(FlexibleArtifact):
    """Validated machine contract for the basin-horizon pilot."""

    schema_version: PositiveInt
    spec_id: str
    status: Literal["pilot"]
    system: dict[str, Any]
    research_target: str
    hypothesis: dict[str, Any]
    finite_window_protocol: FiniteWindowProtocol
    basin_probe: BasinProbeSpec
    replication: ReplicationSpec
    conditions: dict[str, Any]
    evidence_separation: dict[str, str]
    selection_control: SelectionControlSpec
    outputs: list[str]
    claim_boundary: dict[str, str]

    @model_validator(mode="after")
    def validate_seed_partition(self) -> BasinHorizonPilotSpec:
        declared = set(self.replication.model_seeds)
        selected = set(self.selection_control.pilot_seeds) | set(
            self.selection_control.held_out_confirmation_seeds
        )
        if declared != selected:
            raise ValueError("replication model seeds must equal pilot plus held-out seeds")
        return self


class BasinHorizonRunRecord(FlexibleArtifact):
    """One maximum-horizon trajectory and its dependent checkpoint analyses."""

    run_id: str
    spec_id: str
    model_seed: int
    update_mode: Literal["active", "gate_disabled", "gate_frozen"]
    reference_id: str
    trajectory_id: str
    dependency_group_id: str
    forcing_protocol: Literal["autonomous_after_reference_state"]
    radius: NonNegativeFloat
    direction_seed: int | None
    checkpoints: list[dict[str, Any]]
    continuation: dict[str, Any] | None = None
    predictive_information: None = None
    task_utility: None = None

    @model_validator(mode="after")
    def validate_direction_metadata(self) -> BasinHorizonRunRecord:
        if self.radius == 0.0 and self.direction_seed is not None:
            raise ValueError("radius-zero reference must not carry a direction seed")
        if self.radius > 0.0 and self.direction_seed is None:
            raise ValueError("positive-radius trajectories require a direction seed")
        return self


def load_basin_horizon_spec(path: Path) -> BasinHorizonPilotSpec:
    return BasinHorizonPilotSpec.model_validate(load_json(path))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def infer_artifact_kind(path: Path, payload: Any) -> ArtifactKind:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object artifact")
    if "genome" in payload and "metrics" in payload and "id" in payload:
        return "candidate"
    if "population" in payload and "generations" in payload and "eval_seeds" in payload:
        return "evolution_config"
    if "experiment" in payload and "candidate_count" in payload and "generation_count" in payload:
        return "evolution_summary"
    raise ValueError(f"cannot infer artifact kind for {path}")


def validate_artifact(path: Path, *, kind: ArtifactKind | None = None) -> FlexibleArtifact:
    payload = load_json(path)
    artifact_kind = kind or infer_artifact_kind(path, payload)
    if artifact_kind == "candidate":
        return CandidateArtifact.model_validate(payload)
    if artifact_kind == "evolution_config":
        return EvolutionConfigArtifact.model_validate(payload)
    if artifact_kind == "evolution_summary":
        return EvolutionSummaryArtifact.model_validate(payload)
    raise ValueError(f"unknown artifact kind: {artifact_kind}")
