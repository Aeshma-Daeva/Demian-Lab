"""Tests for route-trajectory protocol and unselected pilot."""

from __future__ import annotations

import json
from pathlib import Path

from development.demian_v1_gate_state import V1_ROUTE_CATALOG
from development.route_trajectory import (
    RouteTrajectoryProtocol,
    load_route_protocol,
    run_route_trajectory_pilot,
    write_route_pilot_artifacts,
)


PROTOCOL_PATH = Path("data/ROUTE_TRAJECTORY_PROTOCOL.json")


def test_checked_in_protocol_matches_implemented_route_catalog() -> None:
    protocol = load_route_protocol(PROTOCOL_PATH)

    assert protocol.schema_version == 1
    assert protocol.route_catalog_version == "demian-v1-routes-1.0.0"
    assert {route.id for route in protocol.route_catalog} == {
        route.id for route in V1_ROUTE_CATALOG
    }
    assert {item.id for item in protocol.hypotheses} == {
        "H-ROUTE-001",
        "H-ROUTE-002",
        "H-ROUTE-003",
    }
    assert all(item.evidence_status == "untested" for item in protocol.hypotheses)


def test_protocol_rejects_route_catalog_drift() -> None:
    payload = json.loads(PROTOCOL_PATH.read_text())
    payload["route_catalog"] = payload["route_catalog"][:-1]

    try:
        RouteTrajectoryProtocol.model_validate(payload)
    except ValueError as exc:
        assert "implemented route catalog" in str(exc)
    else:
        raise AssertionError("catalog drift was accepted")


def test_unselected_pilot_emits_compact_summary_and_timestamped_records() -> None:
    payload = run_route_trajectory_pilot(
        seeds=[94, 95],
        hidden_size=8,
        steps=6,
        protocol_path=PROTOCOL_PATH,
    )

    assert payload["experiment"] == "demian-v1-route-trajectory-unselected-pilot"
    assert payload["replication_scope"]["parameter_seed_count"] == 2
    assert payload["replication_scope"]["trajectory_count"] == 2
    assert payload["replication_scope"]["route_step_count"] == 12
    assert len(payload["runs"]) == 2
    assert len(payload["runs"][0]["route_steps"]) == 6
    assert payload["runs"][0]["route_steps"][0]["step"] == 1
    assert "fast_to_message" in payload["runs"][0]["route_summary"]
    route = payload["runs"][0]["route_summary"]["fast_to_message"]
    assert route["norm_basis"] == "contribution_vector"
    assert route["mean_target_update_ratio"] >= 0.0
    assert route["mean_declared_scale_ratio"] >= 0.0
    modulator = payload["runs"][0]["route_summary"]["gate_modulates_surface_routes"]
    assert modulator["norm_basis"] == "deviation_from_neutral"
    assert "mean_target_update_ratio" not in modulator
    assert "mean_declared_scale_ratio" not in modulator
    assert "fast" in payload["runs"][0]["target_summary"]
    assert payload["evidence_status"]["hypothesis_status"] == "untested"


def test_route_cancellation_is_bounded_and_pilot_is_deterministic() -> None:
    first = run_route_trajectory_pilot(
        seeds=[94], hidden_size=8, steps=6, protocol_path=PROTOCOL_PATH
    )
    second = run_route_trajectory_pilot(
        seeds=[94], hidden_size=8, steps=6, protocol_path=PROTOCOL_PATH
    )

    assert first == second
    for target in first["runs"][0]["target_summary"].values():
        assert 0.0 <= target["mean_cancellation"] <= 1.0


def test_artifact_writer_separates_raw_route_steps_from_summary(tmp_path: Path) -> None:
    payload = run_route_trajectory_pilot(
        seeds=[94], hidden_size=8, steps=6, protocol_path=PROTOCOL_PATH
    )
    summary_path = tmp_path / "summary.json"

    parquet_path = write_route_pilot_artifacts(payload, summary_path)
    written = json.loads(summary_path.read_text())

    assert parquet_path == tmp_path / "route_steps.parquet"
    assert parquet_path.exists()
    assert written["runs"][0]["route_steps"] == {
        "artifact": "route_steps.parquet",
        "record_count": 180,
    }
    assert summary_path.stat().st_size < 100_000
