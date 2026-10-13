import math

import pytest

from development import demian_uncertainty_world
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, EvidenceObservation, reference_belief


def test_sampled_evidence_is_seed_reproducible_and_respects_public_schedule() -> None:
    schedule = [0.95, 0.75, 0.5, 0.95]
    left = demian_uncertainty_world.sample_evidence(hidden_target=1, reliabilities=schedule, seed=17)
    right = demian_uncertainty_world.sample_evidence(hidden_target=1, reliabilities=schedule, seed=17)

    assert left == right
    assert [reliability for _, reliability in left] == schedule


def test_sampled_evidence_rejects_invalid_generation_inputs() -> None:
    with pytest.raises(ValueError, match="hidden target"):
        demian_uncertainty_world.sample_evidence(hidden_target=2, reliabilities=[0.75], seed=1)
    with pytest.raises(ValueError, match="reliability"):
        demian_uncertainty_world.sample_evidence(hidden_target=1, reliabilities=[0.6], seed=1)


def test_reference_belief_uses_public_evidence_only() -> None:
    assert reference_belief([])["probability"] == pytest.approx(0.5)
    positive = [EvidenceObservation(0, "evidence", 1, 0.75)]
    assert reference_belief(positive)["probability"] == pytest.approx(0.75)
    assert reference_belief(positive + [EvidenceObservation(1, "evidence", 0, 0.75)])["probability"] == pytest.approx(0.5)
    assert reference_belief(positive * 2)["probability"] == pytest.approx(0.9)
    assert reference_belief([EvidenceObservation(0, "evidence", 1, 0.5)])["log_odds"] == 0.0


def test_identical_public_prefix_ignores_hidden_target_and_snapshot_replays() -> None:
    events = [(1, 0.75), (0, 0.75)]
    left = AmbiguousEvidenceWorld(hidden_target=0, evidence=events, delay_steps=2)
    right = AmbiguousEvidenceWorld(hidden_target=1, evidence=events, delay_steps=2)
    assert left.observe() == right.observe()
    snapshot = left.capture()
    first = [left.step() for _ in range(3)]
    left.restore(snapshot)
    assert [left.step() for _ in range(3)] == first


def test_world_rejects_malformed_evidence() -> None:
    with pytest.raises(ValueError):
        AmbiguousEvidenceWorld(hidden_target=1, evidence=[(2, 0.75)], delay_steps=1)
    with pytest.raises(ValueError):
        AmbiguousEvidenceWorld(hidden_target=1, evidence=[(1, 0.7)], delay_steps=1)
