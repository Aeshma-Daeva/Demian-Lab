"""Schema checks for the immutable six-channel characterization artifact."""

from __future__ import annotations

import json
from pathlib import Path

from development.demian_v1_gate_state import V1_CHANNELS


ARTIFACT = Path("data/diagnostics/demian_v1_characterization_20261005/summary.json")


def test_six_channel_artifact_is_complete_and_task_neutral() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["campaign"]["seeds"] == [94, 95, 96]
    assert payload["afp_v2_config"]
    assert payload["null_controls"]
    assert set(payload["evidence_status"]) == {
        "observation",
        "hypothesis",
        "control",
        "interpretation",
        "untested_speculation",
    }
    six_channel = [row for row in payload["runs"] if row["system"] == "demian_v1_gate_state"]
    assert six_channel
    assert all(set(row["classification"]["channels"]) == set(V1_CHANNELS) for row in six_channel)
    assert all(row["robustness"] for row in six_channel)
    uninterrupted = [row for row in six_channel if row["history_condition"] == "uninterrupted"]
    assert all(row["continuation_control"]["control_steps"] for row in uninterrupted)
    assert all(
        row["classification"]["functional_relevance"]["task_utility"] is None
        for row in six_channel
    )
