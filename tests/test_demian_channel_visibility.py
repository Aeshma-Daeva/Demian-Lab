"""Channel-local visibility under fixed-input replay."""

import pytest
import torch

from development.closed_loop_comparators import ComparatorRunner
from development.demian_v1_gate_state import V1_CHANNELS
from development.demian_channel_visibility import channel_pulse, first_visibility_tick, run_channel_visibility_cell
from development.run_demian_channel_visibility import run_visibility_protocol, write_visibility_artifacts


def test_channel_pulse_changes_only_selected_channel_at_declared_relative_norm():
    state = tuple(torch.full((1, 3), float(index + 1)) for index in range(len(V1_CHANNELS)))

    pulsed = channel_pulse(state, channel="slow", amplitude=0.2, direction_seed=17, sign=1)

    selected = V1_CHANNELS.index("slow")
    delta = (pulsed[selected] - state[selected]).norm()
    assert delta == pytest.approx(state[selected].norm().item() * 0.2)
    assert all(
        torch.equal(before, after) for index, (before, after) in enumerate(zip(state, pulsed)) if index != selected
    )


def test_first_visibility_tick_distinguishes_immediate_and_unresolved():
    assert first_visibility_tick([0.0, 0.2, 0.1], threshold=0.1) == 1
    assert first_visibility_tick([0.0, 0.09, 0.05], threshold=0.1) is None


def test_visibility_cell_retains_opposite_paired_arms_under_identical_inputs():
    runner = ComparatorRunner(architecture="demian", seed=94, hidden_size=8, cue_symbol=1, delay_steps=15)
    frames = [runner.step().input_frame for _ in range(16)]

    result = run_channel_visibility_cell(
        seed=94,
        hidden_size=8,
        frames=frames,
        channel="carrier",
        intervention_tick=4,
        amplitude=0.05,
        direction_seed=1701,
        visibility_threshold=1e-12,
    )

    assert result["visibility_tick"] is not None
    assert result["visibility_tick"] >= 4
    assert len(result["trajectory"]) == len(frames)
    assert all(point["input_replay"] is True for point in result["trajectory"])
    assert result["trajectory"][4]["plus_surface_separation"] > 0
    assert result["trajectory"][4]["minus_surface_separation"] > 0
    assert result["trajectory"][4]["paired_surface_separation"] > 0


def test_visibility_protocol_covers_each_declared_seed_channel_and_direction(tmp_path):
    protocol = {
        "schema_version": 1,
        "protocol_id": "test",
        "seeds": [94, 95],
        "channels": list(V1_CHANNELS),
        "direction_seeds": [1701, 1702],
        "hidden_size": 8,
        "steps": 16,
        "intervention_tick": 4,
        "amplitude": 0.05,
        "visibility_threshold": 1e-12,
    }

    payload = run_visibility_protocol(protocol)
    write_visibility_artifacts(tmp_path / "run", protocol, payload)

    assert len(payload["cells"]) == 2 * len(V1_CHANNELS) * 2
    assert {(cell["seed"], cell["channel"], cell["direction_seed"]) for cell in payload["cells"]} == {
        (seed, channel, direction)
        for seed in protocol["seeds"]
        for channel in V1_CHANNELS
        for direction in protocol["direction_seeds"]
    }


def test_visibility_artifact_writer_rejects_existing_directory(tmp_path):
    protocol = {"protocol_id": "test"}
    payload = {"cells": [], "channel_summary": {}}
    out = tmp_path / "run"

    write_visibility_artifacts(out, protocol, payload)
    with pytest.raises(FileExistsError):
        write_visibility_artifacts(out, protocol, payload)
