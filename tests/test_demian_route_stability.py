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
        "recovery_threshold": 1e-6,
        "recovery_dwell": 2,
    }

    payload = run_route_stability_protocol(protocol)

    assert len(payload["cells"]) == 2 * 2 * 3 * 3
    assert {cell["split"] for cell in payload["cells"]} == {"development", "heldout"}
    assert {cell["config"]["amplitude"] for cell in payload["cells"]} == {0.003, 0.01, 0.03}
    assert {cell["config"]["intervention_tick"] for cell in payload["cells"]} == {4, 8}
    assert {cell["config"]["route_multiplier"] for cell in payload["cells"]} == {1.0, 0.5, 0.0}
