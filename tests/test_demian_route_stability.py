"""Contracts for fixed-forcing Demian route-stability experiments."""

from development.closed_loop_comparators import ComparatorRunner
from development.demian_route_stability import run_route_stability_cell
from development.run_demian_route_stability import run_route_stability_protocol
import pytest


def test_route_stability_cell_replays_one_input_stream_across_controlled_arms() -> None:
    runner = ComparatorRunner(architecture="demian", seed=94, hidden_size=8, cue_symbol=1, delay_steps=15)
    frames = [runner.step().input_frame for _ in range(16)]

    result = run_route_stability_cell(
        seed=94,
        hidden_size=8,
        frames=frames,
        channel="carrier",
        route_id="carrier_to_slow",
        intervention_tick=4,
        amplitude=0.01,
        direction_seed=1701,
        random_direction_seed=2701,
        recovery_threshold=1e-6,
        recovery_dwell=2,
        route_multiplier=0.5,
    )

    assert set(result["arms"]) == {"baseline", "lesion_only", "targeted_lesion", "sham", "random_direction"}
    assert result["arms"]["targeted_lesion"]["route_multiplier"] == 0.5
    assert result["arms"]["sham"]["route_multiplier"] == 1.0
    assert len(result["arms"]["baseline"]["trajectory"]) == len(frames) - 4 + 1
    for name in ("lesion_only", "targeted_lesion", "sham", "random_direction"):
        arm = result["arms"][name]
        assert len(arm["trajectory"]) == len(frames) - 4 + 1
        assert all(point["input_replay"] is True for point in arm["trajectory"])
        assert all("finite_time_state_gain" in point for point in arm["trajectory"])
        assert all("channel_norms" in point for point in arm["trajectory"])
        assert "recovery" in arm
    for name in ("targeted_lesion", "sham", "random_direction"):
        assert result["arms"][name]["initial_pulse_norm"] > 0
    assert result["arms"]["lesion_only"]["initial_pulse_norm"] == 0.0
    assert result["arms"]["lesion_only"]["trajectory"][0]["state_separation"] == 0.0
    assert result["arms"]["lesion_only"]["trajectory"][0]["finite_time_state_gain"] is None
    direct_targeted = result["arms"]["targeted_lesion"]["trajectory"][0]
    direct_sham = result["arms"]["sham"]["trajectory"][0]
    assert direct_targeted["state_separation"] == pytest.approx(direct_sham["state_separation"])
    assert len(result["interaction_trajectory"]) == len(frames) - 4 + 1
    assert all("state_interaction_norm" in point for point in result["interaction_trajectory"])


def test_route_stability_protocol_keeps_development_and_heldout_seed_splits_explicit() -> None:
    protocol = {
        "protocol_id": "test-route-stability",
        "development_seeds": [94],
        "heldout_seeds": [194],
        "hidden_size": 8,
        "steps": 16,
        "channels": ["carrier"],
        "route_id": "carrier_to_slow",
        "route_multipliers": [1.0, 0.5, 0.0],
        "intervention_ticks": [4, 8],
        "amplitudes": [0.003, 0.01, 0.03],
        "direction_seeds": [1701],
        "random_direction_seed_offset": 1000,
        "include_random_route_interaction": True,
        "recovery_threshold": 1e-6,
        "recovery_dwell": 2,
    }

    payload = run_route_stability_protocol(protocol)

    assert len(payload["cells"]) == 2 * 2 * 3 * 3
    assert {cell["split"] for cell in payload["cells"]} == {"development", "heldout"}
    assert {cell["config"]["amplitude"] for cell in payload["cells"]} == {0.003, 0.01, 0.03}
    assert {cell["config"]["intervention_tick"] for cell in payload["cells"]} == {4, 8}
    assert {cell["config"]["route_multiplier"] for cell in payload["cells"]} == {1.0, 0.5, 0.0}
    assert all("mean_peak_random_state_interaction_norm" in group for group in payload["summary"].values())


def test_route_selectivity_cell_adds_matched_random_direction_gain_interaction() -> None:
    runner = ComparatorRunner(architecture="demian", seed=94, hidden_size=8, cue_symbol=1, delay_steps=15)
    frames = [runner.step().input_frame for _ in range(16)]

    result = run_route_stability_cell(
        seed=94,
        hidden_size=8,
        frames=frames,
        channel="carrier",
        route_id="carrier_to_slow",
        intervention_tick=4,
        amplitude=0.01,
        direction_seed=1701,
        random_direction_seed=2701,
        recovery_threshold=1e-6,
        recovery_dwell=2,
        route_multiplier=0.5,
        include_random_route_interaction=True,
    )

    assert "random_targeted_lesion" in result["arms"]
    assert result["arms"]["random_targeted_lesion"]["route_multiplier"] == 0.5
    assert len(result["random_interaction_trajectory"]) == len(frames) - 4 + 1
    assert all("state_interaction_norm" in point for point in result["random_interaction_trajectory"])


def test_route_stability_cell_records_pre_pulse_features_and_matches_full_state_pulse_norm() -> None:
    runner = ComparatorRunner(architecture="demian", seed=94, hidden_size=8, cue_symbol=1, delay_steps=15)
    frames = [runner.step().input_frame for _ in range(16)]

    result = run_route_stability_cell(
        seed=94,
        hidden_size=8,
        frames=frames,
        channel="carrier",
        route_id="carrier_to_slow",
        intervention_tick=4,
        amplitude=0.01,
        direction_seed=1701,
        random_direction_seed=2701,
        recovery_threshold=1e-6,
        recovery_dwell=2,
        route_multiplier=0.5,
        include_random_route_interaction=True,
        random_pulse_scope="full_state",
    )

    features = result["pre_pulse_features"]
    assert {"carrier_norm", "slow_norm", "gate_pressure_mean", "state_velocity", "carrier_slow_alignment"} <= set(features)
    assert result["config"]["random_pulse_scope"] == "full_state"
    assert result["arms"]["random_direction"]["initial_pulse_norm"] == pytest.approx(
        result["arms"]["sham"]["initial_pulse_norm"]
    )


def test_route_stability_protocol_crosses_declared_histories_and_random_pulse_scopes() -> None:
    protocol = {
        "protocol_id": "test-route-characterization",
        "development_seeds": [94],
        "heldout_seeds": [194],
        "hidden_size": 8,
        "steps": 16,
        "channels": ["carrier"],
        "route_id": "carrier_to_slow",
        "route_multipliers": [0.5],
        "intervention_ticks": [4],
        "amplitudes": [0.01],
        "direction_seeds": [1701],
        "random_direction_seed_offset": 1000,
        "random_pulse_scopes": ["channel", "full_state"],
        "history_cue_symbols": [0, 1],
        "response_window_transitions": 3,
        "include_random_route_interaction": True,
        "recovery_threshold": 1e-6,
        "recovery_dwell": 2,
    }

    payload = run_route_stability_protocol(protocol)

    assert len(payload["cells"]) == 2 * 2 * 2
    assert {cell["history"]["cue_symbol"] for cell in payload["cells"]} == {0, 1}
    assert {cell["config"]["random_pulse_scope"] for cell in payload["cells"]} == {"channel", "full_state"}
    assert {group["analysis_window_transitions"] for group in payload["summary"].values()} == {3}
