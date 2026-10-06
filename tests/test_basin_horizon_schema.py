"""Contracts for the machine-readable basin-horizon pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from development.lab_schemas import (
    BasinHorizonPilotSpec,
    BasinHorizonRunRecord,
    load_basin_horizon_spec,
)


SPEC_PATH = Path("docs/superpowers/specs/2026-10-05-demian-basin-horizon-pilot.json")


def _spec_payload() -> dict:
    return json.loads(SPEC_PATH.read_text())


def test_checked_in_basin_horizon_spec_is_valid() -> None:
    spec = load_basin_horizon_spec(SPEC_PATH)

    assert spec.spec_id == "demian-basin-horizon-afpv2-pilot-v1"
    assert spec.basin_probe.radii[0] == 0.0
    assert spec.finite_window_protocol.checkpoints == [64, 128, 256]
    assert spec.selection_control.pilot_seeds == [94]
    assert spec.selection_control.held_out_confirmation_seeds == [95]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["basin_probe"].update(radii=[0.001, 0.01]),
        lambda payload: payload["finite_window_protocol"].update(checkpoints=[128, 64]),
        lambda payload: payload["finite_window_protocol"].update(
            surface_delta_tolerance=0.02,
            surface_escape_tolerance=0.01,
        ),
        lambda payload: payload["finite_window_protocol"].update(persistence_steps=4),
        lambda payload: payload["selection_control"].update(
            pilot_seeds=[94],
            held_out_confirmation_seeds=[94],
        ),
    ],
)
def test_spec_rejects_designs_that_break_pilot_inference(mutate) -> None:
    payload = _spec_payload()
    mutate(payload)

    with pytest.raises(ValidationError):
        BasinHorizonPilotSpec.model_validate(payload)


def test_run_record_requires_dependency_and_direction_metadata() -> None:
    common = {
        "run_id": "seed94:active:r=0.01:direction=7001",
        "spec_id": "demian-basin-horizon-afpv2-pilot-v1",
        "model_seed": 94,
        "update_mode": "active",
        "reference_id": "seed94:active:reference0",
        "trajectory_id": "seed94:active:r=0.01:direction=7001",
        "dependency_group_id": "seed94:active:direction=7001",
        "forcing_protocol": "autonomous_after_reference_state",
        "radius": 0.01,
        "direction_seed": 7001,
        "checkpoints": [],
    }

    record = BasinHorizonRunRecord.model_validate(common)
    assert record.dependency_group_id == "seed94:active:direction=7001"

    with pytest.raises(ValidationError):
        BasinHorizonRunRecord.model_validate({key: value for key, value in common.items() if key != "dependency_group_id"})

    with pytest.raises(ValidationError):
        BasinHorizonRunRecord.model_validate({**common, "direction_seed": None})

    zero = BasinHorizonRunRecord.model_validate(
        {
            **common,
            "run_id": "seed94:active:r=0",
            "trajectory_id": "seed94:active:r=0",
            "dependency_group_id": "seed94:active:reference0",
            "radius": 0.0,
            "direction_seed": None,
        }
    )
    assert zero.direction_seed is None
