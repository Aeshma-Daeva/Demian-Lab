"""Contracts for the declared closed-loop experimental boundary."""

from __future__ import annotations

from pathlib import Path

from development.closed_loop_protocol import load_closed_loop_protocol


def test_checked_in_protocol_declares_external_connector_and_checkpoint_ownership() -> None:
    protocol = load_closed_loop_protocol(Path("data/CLOSED_LOOP_PROTOCOL.json"))

    assert protocol.schema_version == 1
    assert protocol.demian_state_channels == [
        "fast", "slow", "control", "message", "carrier", "gate"
    ]
    assert protocol.connector_is_external is True
    assert set(protocol.checkpoint_owners) == {
        "demian_runtime",
        "world_register",
        "connector_queue",
        "interface_configuration",
    }
    assert protocol.action_operations == ["noop", "read", "write", "answer"]
    assert "fixed_observation_replay" in protocol.controls
    assert "storage_reset" in protocol.controls
