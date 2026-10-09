"""Fail-closed provenance and development gates, with hand-declared rate cells."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from development import regime_validation as validation


def _audit(dirs, raw, hashes):
    return validation.audit_development(
        dirs,
        raw,
        hashes,
        test_verification={
            "returncode": 0,
            "implementation_hashes": hashes,
            "command": ["pytest", "tests/test_regime_characterization.py"],
        },
    )


def _panel(tmp_path):
    protocol = {
        "architectures": ["rnn"],
        "development_seeds": [94],
        "direction_seeds": [1701, 1702],
        "reference_hidden_size": 8,
        "burn_in": 2,
        "horizons": [8, 12],
        "confirmation_seed_range": [194, 293],
        "finite_difference_steps": [1e-4, 1e-5, 1e-6],
        "finite_difference_relative_error_tolerance": 1e-5,
        "classification_tolerance_per_tick": 0.01,
        "direction_spread_tolerance_per_tick": 0.01,
        "horizon_difference_tolerance_per_tick": 0.01,
        "reconstruction_absolute_tolerance": 1e-10,
        "sample_every": 32,
        "state_budget_elements": 8,
    }
    raw = json.dumps(protocol).encode()
    digest = hashlib.sha256(raw).hexdigest()
    source = b"frozen measurement source\n"
    hashes = {"measurement.py": hashlib.sha256(source).hexdigest()}
    dirs = []
    for horizon in protocol["horizons"]:
        path = tmp_path / str(horizon)
        path.mkdir()
        (path / "sources").mkdir()
        (path / "sources/measurement.py").write_bytes(source)
        manifest = {
            "protocol": protocol,
            "protocol_sha256": digest,
            "implementation_hashes": hashes,
            "torch_version": "test",
            "threads": 1,
            "actual_config": {"reference_hidden_size": 8, "burn_in": 2, "steps": horizon, "seeds": [94]},
        }
        (path / "manifest.json").write_text(json.dumps(manifest))
        check = {
            "passed": True,
            "checks": [{"epsilon": e, "relative_error": 1e-8} for e in protocol["finite_difference_steps"]],
        }
        records = []
        for direction in protocol["direction_seeds"]:
            records.append(
                {
                    "architecture": "rnn",
                    "seed": 94,
                    "protocol_sha256": digest,
                    "state_elements": 8,
                    "hidden_size": 8,
                    "budget_reference_dtype": "float32",
                    "numerical_state_bytes": 64,
                    "measurement_dtype": "float64",
                    "jvp_validation": check,
                    "endpoint_jvp_validation": check,
                    "result": {
                        "status": "ok",
                        "direction_seed": direction,
                        "rate": -0.5,
                        "measured_ticks": horizon - 2,
                        "samples": [
                            {
                                "tick": t,
                                "state_norm": 1.0,
                                "operating_point": {
                                    "saturation_coverage": "complete_declared_operations",
                                    "operation_saturation": {"rnn_tanh": 0.0},
                                    "transition_reconstruction_max_error": 0.0,
                                    "component_norms": [1.0],
                                    "component_increment_norms": [0.0],
                                },
                            }
                            for t in [0, horizon - 3]
                        ],
                    },
                }
            )
        (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (path / "summary.json").write_text(
            json.dumps({"all_passed": True, "records": 2, "precision_and_finiteness_passed": 2})
        )
        dirs.append(path)
    return raw, hashes, dirs


def _rewrite(path, mutate):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_complete_development_panel_approves_only_measurement_confirmation(tmp_path):
    raw, hashes, dirs = _panel(tmp_path)
    decision = _audit(dirs, raw, hashes)
    assert decision["approved"]
    assert decision["confirmed_scientific_claims"] == []
    assert decision["records_checked"] == 4
    validation.validate_confirmation(raw, decision, hashes, steps=12, width=8, burn_in=2, seeds=list(range(194, 294)))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "precision",
        "coverage",
        "parity",
        "direction",
        "horizon",
        "protocol",
        "snapshot",
        "summary",
        "nonfinite",
        "budget",
        "operation_names",
    ],
)
def test_faults_prevent_confirmation(tmp_path, fault):
    raw, hashes, dirs = _panel(tmp_path)
    path = dirs[-1] / "records.jsonl"
    if fault == "snapshot":
        (dirs[-1] / "sources/measurement.py").write_text("modified")
    elif fault == "summary":
        (dirs[-1] / "summary.json").unlink()
    else:

        def mutate(records):
            if fault == "missing":
                records.pop()
            if fault == "duplicate":
                records.append(copy.deepcopy(records[0]))
            if fault == "precision":
                records[0]["jvp_validation"]["checks"][0]["relative_error"] = 0.1
            if fault == "coverage":
                records[0]["result"]["samples"][0]["operating_point"]["saturation_coverage"] = "partial"
            if fault == "parity":
                records[0]["result"]["samples"][0]["operating_point"]["transition_reconstruction_max_error"] = 0.01
            if fault == "direction":
                records[0]["result"]["rate"] = -0.4
            if fault == "horizon":
                for record in records:
                    record["result"]["rate"] = -0.4
            if fault == "protocol":
                records[0]["protocol_sha256"] = "changed"
            if fault == "nonfinite":
                records[0]["result"]["rate"] = float("nan")
            if fault == "budget":
                records[0]["state_elements"] = 9
                records[0]["numerical_state_bytes"] = 72
            if fault == "operation_names":
                records[0]["result"]["samples"][0]["operating_point"]["operation_saturation"] = {"unrelated": 0.0}

        _rewrite(path, mutate)
    decision = _audit(dirs, raw, hashes)
    assert not decision["approved"]
    assert decision["reasons"]


@pytest.mark.parametrize("fault", ["unapproved", "source", "protocol", "width", "burn", "seeds", "horizon"])
def test_confirmation_rejects_stale_or_changed_settings(tmp_path, fault):
    raw, hashes, dirs = _panel(tmp_path)
    decision = _audit(dirs, raw, hashes)
    kwargs = dict(steps=12, width=8, burn_in=2, seeds=list(range(194, 294)))
    if fault == "unapproved":
        decision["approved"] = False
    if fault == "source":
        hashes = {"measurement.py": "changed"}
    if fault == "protocol":
        raw += b" "
    if fault == "width":
        kwargs["width"] = 9
    if fault == "burn":
        kwargs["burn_in"] = 3
    if fault == "seeds":
        kwargs["seeds"] = [94]
    if fault == "horizon":
        kwargs["steps"] = 13
    with pytest.raises(ValueError):
        validation.validate_confirmation(raw, decision, hashes, **kwargs)


def test_confirmation_cli_without_decision_creates_no_output(tmp_path):
    out = tmp_path / "forbidden"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "development.run_regime_characterization",
            "--mode",
            "confirmation",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "decision" in result.stderr.lower()
    assert not out.exists()


def test_missing_known_map_verification_prevents_confirmation(tmp_path):
    raw, hashes, dirs = _panel(tmp_path)
    assert not validation.audit_development(dirs, raw, hashes)["approved"]


def test_modified_development_evidence_invalidates_existing_decision(tmp_path):
    raw, hashes, dirs = _panel(tmp_path)
    decision = _audit(dirs, raw, hashes)
    _rewrite(dirs[0] / "records.jsonl", lambda records: records[0]["result"].update(rate=-0.50001))
    with pytest.raises(ValueError, match="stale"):
        validation.validate_confirmation(
            raw, decision, hashes, steps=12, width=8, burn_in=2, seeds=list(range(194, 294))
        )


def test_development_cli_archives_exact_sources(tmp_path):
    out = tmp_path / "development"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "development.run_regime_characterization",
            "--reference-hidden-size",
            "8",
            "--steps",
            "16",
            "--burn-in",
            "4",
            "--seeds",
            "94",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    for name, digest in manifest["implementation_hashes"].items():
        data = (out / "sources" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert data == (validation.ROOT / name).read_bytes()
    assert json.loads((out / "summary.json").read_text())["all_passed"]


def test_campaign_validation_only_never_runs_fresh_seeds(tmp_path):
    protocol = json.loads((validation.ROOT / "docs/REGIME_MATCHED_PROTOCOL.json").read_text())
    protocol.update(
        architectures=["rnn"],
        reference_hidden_size=2,
        development_seeds=[94],
        direction_seeds=[1701],
        horizons=[40, 48],
        burn_in=32,
        state_budget_elements=14,
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    out = tmp_path / "campaign"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "development.run_regime_campaign",
            "--protocol",
            str(path),
            "--out",
            str(out),
            "--validation-only",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    status = json.loads((out / "status.json").read_text())
    assert status["stage"] == "development_validated"
    assert not list(out.glob("confirmation_*"))
    assert json.loads((out / "decision.json").read_text())["approved"]


def test_local_import_boundary_change_invalidates_existing_decision(tmp_path, monkeypatch):
    # A different exported base class must not reuse earlier regime evidence.
    root = tmp_path / "repo"
    for name in validation.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n")
    boundary = root / "development/substrates/legacy.py"
    boundary.parent.mkdir(parents=True)
    boundary.write_text("from .model_a import DemianNativeV9Substrate\n")
    monkeypatch.setattr(validation, "ROOT", root)
    before = validation.source_hashes()
    assert "development/substrates/legacy.py" in before
    raw, hashes, dirs = _panel(tmp_path)
    for directory in dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["implementation_hashes"] = before
        (directory / "manifest.json").write_text(json.dumps(manifest))
        for name in before:
            target = directory / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    decision = _audit(dirs, raw, before)
    validation.validate_confirmation(raw, decision, before, steps=12, width=8, burn_in=2, seeds=list(range(194, 294)))
    boundary.write_text("from .model_b import DemianNativeV9Substrate\n")
    after = validation.source_hashes()
    assert before != after
    with pytest.raises(ValueError, match="mismatch"):
        validation.validate_confirmation(
            raw, decision, after, steps=12, width=8, burn_in=2, seeds=list(range(194, 294))
        )
