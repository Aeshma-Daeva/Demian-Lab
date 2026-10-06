"""Operational tests for AFP-v2 surface/internal classification."""

import math
import random

from development.afp_v2_characterization import (
    AFPV2Config,
    build_campaign_summary,
    characterize_model,
    classify_afp_v2,
    null_control_results,
    surface_matched_regimes,
    tail_control_window,
)
import torch


def _fixed_surface(steps: int) -> list[list[float]]:
    return [[1.0, -1.0] for _ in range(steps)]


def test_rejects_nonconverged_surface():
    surface = [[float(step), 0.0] for step in range(24)]
    hidden = [[float(step), (-1.0) ** step] for step in range(24)]

    result = classify_afp_v2(surface, hidden)

    assert result["classification"] == "surface_nonconvergent"
    assert result["surface_converged"] is False


def test_distinguishes_internal_fixed_and_numerical_drift():
    surface = _fixed_surface(24)
    fixed = [[0.2, -0.2] for _ in range(24)]
    drift = [[0.2 + step * 1e-12, -0.2] for step in range(24)]

    fixed_result = classify_afp_v2(surface, fixed)
    drift_result = classify_afp_v2(surface, drift)

    assert fixed_result["classification"] == "surface_fixed_internal_fixed"
    assert drift_result["classification"] == "surface_fixed_numerical_drift"


def test_rejects_transient_and_trivial_accumulation():
    surface = _fixed_surface(32)
    transient = [[0.2 * step, 0.0] if step < 16 else [3.0, 0.0] for step in range(32)]
    accumulation = [[0.02 * step, -0.01 * step] for step in range(32)]

    transient_result = classify_afp_v2(surface, transient)
    accumulation_result = classify_afp_v2(surface, accumulation)

    assert transient_result["classification"] == "surface_fixed_transient_internal"
    assert accumulation_result["classification"] == "surface_fixed_trivial_accumulation"


def test_structured_hidden_dynamics_need_fixed_body_continuation_control():
    surface = _fixed_surface(32)
    hidden = [
        [math.cos(step * math.pi / 8.0), math.sin(step * math.pi / 8.0)]
        for step in range(32)
    ]
    config = AFPV2Config(hidden_delta_min=0.1, continuation_gap_min=0.01)

    candidate = classify_afp_v2(surface, hidden, config=config)
    supported = classify_afp_v2(
        surface,
        hidden,
        fixed_body_continuation_gap=0.2,
        config=config,
    )

    assert candidate["classification"] == "afp_v2_candidate"
    assert candidate["continuation_relevant"] is None
    assert supported["classification"] == "afp_v2_supported"
    assert supported["continuation_relevant"] is True


def test_rejects_small_step_surface_drift():
    surface = [[0.005 * step, 0.0] for step in range(32)]
    hidden = [
        [math.cos(step * math.pi / 8.0), math.sin(step * math.pi / 8.0)]
        for step in range(32)
    ]

    result = classify_afp_v2(surface, hidden, fixed_body_continuation_gap=0.1)

    assert result["classification"] == "surface_small_step_drift"


def test_rejects_damped_hidden_transient():
    surface = _fixed_surface(128)
    hidden = [[((-1.0) ** step) * (0.98**step), 0.0] for step in range(128)]

    result = classify_afp_v2(surface, hidden, fixed_body_continuation_gap=0.1)

    assert result["classification"] == "surface_fixed_transient_internal"


def test_rejects_unstructured_random_walk():
    rng = random.Random(7)
    surface = _fixed_surface(128)
    hidden = [[0.0, 0.0]]
    for _ in range(127):
        hidden.append(
            [
                hidden[-1][0] + rng.gauss(0.0, 0.02),
                hidden[-1][1] + rng.gauss(0.0, 0.02),
            ]
        )

    result = classify_afp_v2(
        surface,
        hidden,
        fixed_body_continuation_gap=0.1,
        config=AFPV2Config(
            hidden_stationarity_min_ratio=0.0,
            hidden_stationarity_max_ratio=10.0,
        ),
    )

    assert result["classification"] == "surface_fixed_unstructured_change"


def test_rejects_rank_one_random_walk():
    rng = random.Random(3)
    surface = _fixed_surface(128)
    hidden = [[0.0]]
    for _ in range(127):
        hidden.append([hidden[-1][0] + rng.gauss(0.0, 0.02)])

    result = classify_afp_v2(surface, hidden, fixed_body_continuation_gap=0.1)

    assert result["classification"] == "surface_fixed_unstructured_change"


def test_reports_same_surface_class_with_different_internal_regimes():
    results = [
        {"run_id": "fixed", "surface_converged": True, "classification": "surface_fixed_internal_fixed"},
        {"run_id": "structured", "surface_converged": True, "classification": "afp_v2_supported"},
        {"run_id": "moving", "surface_converged": False, "classification": "surface_nonconvergent"},
    ]

    matched = surface_matched_regimes(results)

    assert matched == {
        "surface_converged": {
            "classifications": ["afp_v2_supported", "surface_fixed_internal_fixed"],
            "run_ids": ["fixed", "structured"],
        }
    }


class _ToyProjectedSystem(torch.nn.Module):
    def initial_state(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(batch_size, 1, device=device),
            torch.tensor([[1.0, 0.0]], device=device),
        )

    def step(self, state):
        surface, hidden = state
        angle = math.pi / 8.0
        return surface, torch.stack(
            (
                math.cos(angle) * hidden[:, 0] - math.sin(angle) * hidden[:, 1],
                math.sin(angle) * hidden[:, 0] + math.cos(angle) * hidden[:, 1],
            ),
            dim=1,
        )

    def state_vector(self, state):
        return state[0]

    def state_components(self, state):
        return {"surface": state[0], "hidden": state[1]}

    def write_surface_state(self, state, surface):
        return surface.view(1, -1), state[1]


def test_characterizes_full_state_and_explicit_channels():
    result = characterize_model(
        "toy",
        lambda _size: _ToyProjectedSystem(),
        hidden_size=2,
        seed=7,
        steps=32,
        fixed_body_continuation_gap=0.2,
        config=AFPV2Config(hidden_delta_min=0.1, continuation_gap_min=0.01),
    )

    assert result["run_id"] == "toy:seed=7:baseline"
    assert result["classification"] == "afp_v2_supported"
    assert result["channel_regimes"]["surface"]["classification"] == "surface_fixed_internal_fixed"
    assert result["channel_regimes"]["hidden"]["classification"] == "afp_v2_candidate"


def test_null_controls_cover_drift_transient_and_accumulation():
    controls = null_control_results(AFPV2Config(hidden_delta_min=0.01))

    assert controls["numerical_drift"]["classification"] == "surface_fixed_numerical_drift"
    assert controls["ordinary_transient"]["classification"] == "surface_fixed_transient_internal"
    assert controls["trivial_accumulation"]["classification"] == "surface_fixed_trivial_accumulation"
    assert controls["small_step_surface_drift"]["classification"] == "surface_small_step_drift"
    assert controls["stochastic_accumulation"]["classification"] in {
        "surface_fixed_transient_internal",
        "surface_fixed_unstructured_change",
    }


def test_campaign_summary_separates_evidence_statuses():
    runs = [
        {"run_id": "a", "surface_converged": True, "classification": "surface_fixed_internal_fixed"},
        {"run_id": "b", "surface_converged": True, "classification": "afp_v2_supported"},
    ]

    summary = build_campaign_summary(runs, AFPV2Config())

    assert summary["definition"]["surface_convergence"] == "delta_y_t -> 0"
    assert summary["evidence_status"]["observation"]
    assert summary["evidence_status"]["hypothesis"]
    assert summary["evidence_status"]["control"]
    assert summary["evidence_status"]["interpretation"]
    assert summary["evidence_status"]["interpretation"].startswith("1/2 baseline runs pass")
    assert summary["evidence_status"]["untested_speculation"]
    assert summary["surface_matched_regimes"]["surface_converged"]["run_ids"] == ["a", "b"]


def test_continuation_control_matches_classified_tail():
    assert tail_control_window(steps=128, tail_steps=24) == {
        "pause_steps": 104,
        "resume_steps": 24,
        "classified_steps": [105, 128],
        "control_steps": [105, 128],
    }
