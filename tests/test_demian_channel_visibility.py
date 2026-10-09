"""Channel-local visibility under fixed-input replay."""

import pytest
import torch
import json

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


def test_opposite_channel_pulses_have_exactly_opposite_selected_deltas():
    state = tuple(torch.full((1, 3), float(index + 1)) for index in range(len(V1_CHANNELS)))
    selected = V1_CHANNELS.index("carrier")

    plus = channel_pulse(state, channel="carrier", amplitude=0.2, direction_seed=17, sign=1)
    minus = channel_pulse(state, channel="carrier", amplitude=0.2, direction_seed=17, sign=-1)

    assert torch.equal(plus[selected] - state[selected], -(minus[selected] - state[selected]))


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
    assert len(result["trajectory"]) == len(frames) + 1
    assert all(point["input_replay"] is True for point in result["trajectory"])
    first_transition = next(point for point in result["trajectory"] if point["transitions_after_pulse"] == 1)
    assert first_transition["plus_surface_separation"] > 0
    assert first_transition["minus_surface_separation"] > 0
    assert first_transition["paired_surface_separation"] > 0


def test_visibility_cell_counts_direct_readout_as_zero_and_first_transition_as_one():
    runner = ComparatorRunner(architecture="demian", seed=94, hidden_size=8, cue_symbol=1, delay_steps=15)
    frames = [runner.step().input_frame for _ in range(16)]

    result = run_channel_visibility_cell(
        seed=94,
        hidden_size=8,
        frames=frames,
        channel="slow",
        intervention_tick=4,
        amplitude=0.05,
        direction_seed=1701,
        visibility_threshold=1e-6,
    )

    direct = next(point for point in result["trajectory"] if point["transitions_after_pulse"] == 0)
    assert direct["paired_surface_separation"] == 0.0
    assert direct["paired_state_separation"] > 0.0
    assert result["visibility_delay"] == 1
    assert result["visibility_transition"] == 1


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
    assert all(cell["trajectory"] for cell in payload["cells"])
    assert all("paired_state_separation" in point for cell in payload["cells"] for point in cell["trajectory"])
    persisted = [json.loads(line) for line in (tmp_path / "run" / "cells.jsonl").read_text().splitlines()]
    assert all(cell["trajectory"] for cell in persisted)
    assert all("paired_state_separation" in point for cell in persisted for point in cell["trajectory"])


def test_visibility_artifact_writer_rejects_existing_directory(tmp_path):
    protocol = {"protocol_id": "test"}
    payload = {"cells": [], "channel_summary": {}}
    out = tmp_path / "run"

    write_visibility_artifacts(out, protocol, payload)
    with pytest.raises(FileExistsError):
        write_visibility_artifacts(out, protocol, payload)
