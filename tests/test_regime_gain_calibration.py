"""Recurrent-only scaling and conservative development selection."""

import math
import copy
import json
import sys

import pytest
import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter
from development.regime_characterization import StateMap, tangent_window


@pytest.mark.parametrize("gain", [0.5, 1.2])
def test_scaled_rnn_tangent_matches_zero_orbit_derivative(gain):
    adapter = ClosedLoopModelAdapter(architecture="rnn", seed=94, hidden_size=1)
    with torch.no_grad():
        adapter.model.weight_ih.zero_()
        adapter.model.weight_hh.fill_(1)
        adapter.model.bias_ih.zero_()
        adapter.model.bias_hh.zero_()
    mapping = StateMap(adapter, recurrent_multiplier=gain)
    result = tangent_window(
        mapping.transition,
        mapping.readout,
        mapping.initial_state,
        [torch.zeros(1, 1).double()] * 16,
        direction_seed=1701,
    )
    assert result["rate"] == pytest.approx(math.log(gain), abs=1e-12)


def test_multiplier_changes_only_recurrent_weight():
    reference = ClosedLoopModelAdapter(architecture="rnn", seed=94, hidden_size=8)
    scaled = ClosedLoopModelAdapter(architecture="rnn", seed=94, hidden_size=8)
    StateMap(scaled, recurrent_multiplier=1.5)
    for name, value in reference.model.state_dict().items():
        expected = value.double() * (1.5 if name == "weight_hh" else 1)
        assert torch.equal(scaled.model.state_dict()[name], expected)


@pytest.mark.parametrize("fault", [None, "missing", "failed", "distant", "direction_sensitive"])
def test_selection_requires_complete_valid_nearby_cells(fault):
    from development.regime_gain_calibration import select_gain

    protocol = {
        "grid": [0.5, 1.0],
        "seeds": [94],
        "directions": [1701, 1702],
        "horizons": [512, 2048],
        "matching_tolerance": 0.01,
        "direction_tolerance": 0.01,
        "horizon_tolerance": 0.01,
    }
    records = [
        {
            "multiplier": gain,
            "seed": 94,
            "direction_seed": direction,
            "horizon": horizon,
            "rate": (-0.3 if gain == 0.5 else 0.039),
            "passed": True,
        }
        for gain in protocol["grid"]
        for direction in protocol["directions"]
        for horizon in protocol["horizons"]
    ]
    if fault == "missing":
        records.pop()
    if fault == "failed":
        records[-1]["passed"] = False
    if fault == "distant":
        for record in records:
            if record["multiplier"] == 1.0:
                record["rate"] = -0.1
    if fault == "direction_sensitive":
        records[-1]["rate"] = 0.07
    result = select_gain(records, {512: 0.04, 2048: 0.04}, protocol)
    assert result["selected_multiplier"] == (1.0 if fault is None else None)
    assert result["status"] == ("development_candidate" if fault is None else "no_valid_match")


def test_gain_runner_requires_a_completed_phase0_audit(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "forbidden"
    result = subprocess.run(
        [sys.executable, "-m", "development.run_regime_gain_calibration", "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not out.exists()


@pytest.fixture
def gain_cli(tmp_path, monkeypatch):
    from development import run_regime_gain_calibration as runner

    protocol = json.loads((runner.ROOT / "docs/REGIME_GAIN_PROTOCOL.json").read_text())
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    base = {
        "development_seeds": protocol["seeds"],
        "direction_seeds": protocol["directions"],
        "horizons": protocol["horizons"],
        "rnn_weight_hh_multiplier_grid": protocol["grid"],
        "state_budget_elements": protocol["state_elements"],
        "burn_in": protocol["burn_in"],
        "reference_hidden_size": protocol["reference_hidden_size"],
        "direction_spread_tolerance_per_tick": 0.01,
        "horizon_difference_tolerance_per_tick": 0.01,
    }
    (campaign / "protocol.json").write_text(json.dumps(base))
    fresh = {
        "approved": True,
        "audit": [],
        "authorized_stage": "phase0b_development_calibration",
        "campaign": str(campaign),
        "confirmed_scientific_claims": [],
        "evidence_hashes": {"records.jsonl": "frozen-records"},
        "implementation_hashes": {"development/closed_loop_comparators.py": "frozen-source"},
        "original_decision_sha256": "frozen-decision",
        "protocol_sha256": "frozen-protocol",
        "reasons": [],
        "records_checked": 2400,
        "run_dirs": [],
        "schema_version": 1,
        "test_verification": {"returncode": 0},
    }
    # Isolate the CLI boundary from the expensive historical 2400-cell audit.
    monkeypatch.setattr(runner, "audit_campaign", lambda path: copy.deepcopy(fresh))
    audit_path, protocol_path, out = (tmp_path / name for name in ("audit.json", "protocol.json", "out"))
    monkeypatch.setattr(
        sys, "argv", ["calibration", "--audit", str(audit_path), "--protocol", str(protocol_path), "--out", str(out)]
    )
    return runner, copy.deepcopy(fresh), protocol, audit_path, protocol_path, out


@pytest.mark.parametrize("fault", ["empty", "altered", "missing"])
def test_gain_cli_rejects_unbound_source_hashes(gain_cli, capsys, fault):
    runner, audit, protocol, audit_path, protocol_path, out = gain_cli
    if fault == "missing":
        del audit["implementation_hashes"]
    else:
        audit["implementation_hashes"] = (
            {} if fault == "empty" else {"development/closed_loop_comparators.py": "altered"}
        )
    audit_path.write_text(json.dumps(audit))
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "audit is invalid or stale" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("field", ["matching_tolerance", "direction_tolerance", "horizon_tolerance"])
def test_gain_cli_rejects_relaxed_selection_tolerances(gain_cli, capsys, field):
    runner, audit, protocol, audit_path, protocol_path, out = gain_cli
    protocol[field] = 1.0
    audit_path.write_text(json.dumps(audit))
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "calibration changes" in capsys.readouterr().err
    assert not out.exists()
