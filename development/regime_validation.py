"""Development audits and fail-closed confirmation authorization."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "development/regime_characterization.py",
    "development/regime_validation.py",
    "development/run_regime_characterization.py",
    "development/run_regime_campaign.py",
    "development/closed_loop_comparators.py",
    "development/closed_loop_demian.py",
    "development/closed_loop_world.py",
    "development/demian_v1_gate_state.py",
    "development/substrate_lab.py",
    "tests/test_regime_characterization.py",
    "tests/test_regime_validation.py",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256((ROOT / name).read_bytes()) for name in _source_files()}


def _source_files() -> list[str]:
    # Freeze all local runtime sources, including package import/re-export boundaries.
    # A new or unrelated edited module conservatively invalidates the decision too.
    return sorted(set(SOURCE_FILES) | {str(path.relative_to(ROOT)) for path in (ROOT / "development").rglob("*.py")})


def snapshot_sources(out: Path) -> dict[str, str]:
    hashes = {}
    for name in _source_files():
        contents = (ROOT / name).read_bytes()
        target = out / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        hashes[name] = sha256(contents)
    return hashes


def _classification(rate: float, tolerance: float) -> str:
    return "contracting" if rate < -tolerance else "expanding" if rate > tolerance else "near_zero"


def _operation_names(architecture: str) -> set[str]:
    if architecture == "rnn":
        return {"rnn_tanh"}
    if architecture == "gru":
        return {"gru_reset", "gru_update", "gru_candidate"}
    return {
        f"{channel}_{suffix}"
        for channel in ("fast", "slow", "control", "message", "carrier", "gate")
        for suffix in ("gate", "mix")
    } | {
        "gate_input",
        "fast_to_message",
        "message_to_carrier",
        "fast_to_slow",
        "carrier_to_slow",
        "fast_slow_to_control",
        "control_readout",
        "message_to_fast",
        "carrier_to_fast",
        "message_readout",
        "carrier_readout",
        "gate_readout",
        "gate_pressure",
        "fast_integrated",
    }


def audit_development(
    run_dirs: list[Path], protocol_bytes: bytes, hashes: dict[str, str], *, test_verification: dict | None = None
) -> dict:
    """Audit exact cells; tolerances are development-calibrated, frozen before confirmation."""
    protocol = json.loads(protocol_bytes)
    digest = sha256(protocol_bytes)
    reasons, records, evidence = [], {}, {}
    if (
        not test_verification
        or test_verification.get("returncode") != 0
        or test_verification.get("implementation_hashes") != hashes
        or "tests/test_regime_characterization.py" not in test_verification.get("command", [])
    ):
        reasons.append("known-map test verification missing, stale or failed")
    expected = set(
        itertools.product(
            protocol["horizons"],
            protocol["architectures"],
            protocol["development_seeds"],
            protocol["direction_seeds"],
        )
    )
    for directory in run_dirs:
        path = Path(directory).resolve()
        try:
            for name in ("manifest.json", "records.jsonl", "summary.json"):
                evidence[str(path / name)] = sha256((path / name).read_bytes())
            manifest = json.loads((path / "manifest.json").read_bytes())
            summary = json.loads((path / "summary.json").read_bytes())
            config = manifest["actual_config"]
            horizon = config["steps"]
            if (
                manifest["protocol_sha256"] != digest
                or manifest["protocol"] != protocol
                or manifest["implementation_hashes"] != hashes
                or manifest["threads"] != 1
                or config["reference_hidden_size"] != protocol["reference_hidden_size"]
                or config["burn_in"] != protocol["burn_in"]
                or config["seeds"] != protocol["development_seeds"]
                or horizon not in protocol["horizons"]
            ):
                reasons.append(f"manifest mismatch: {path}")
            for name, source_digest in hashes.items():
                if sha256((path / "sources" / name).read_bytes()) != source_digest:
                    reasons.append(f"source snapshot mismatch: {path}/{name}")
            count = 0
            with (path / "records.jsonl").open() as stream:
                for line in stream:
                    record = json.loads(line)
                    result = record["result"]
                    key = (horizon, record["architecture"], record["seed"], result["direction_seed"])
                    if key in records:
                        reasons.append(f"duplicate cell: {key}")
                    records[key] = record
                    count += 1
                    if key not in expected or record["protocol_sha256"] != digest:
                        reasons.append(f"unexpected cell or protocol: {key}")
                    if result["status"] != "ok" or not math.isfinite(result["rate"]):
                        reasons.append(f"unresolved rate: {key}")
                    if result["measured_ticks"] != horizon - protocol["burn_in"]:
                        reasons.append(f"measurement horizon mismatch: {key}")
                    if (
                        record["measurement_dtype"] != "float64"
                        or record["numerical_state_bytes"] != 8 * record["state_elements"]
                        or record["state_elements"] != protocol["state_budget_elements"]
                        or record["budget_reference_dtype"] != "float32"
                        or record["hidden_size"]
                        != (
                            protocol["reference_hidden_size"]
                            if record["architecture"].startswith("demian")
                            else protocol["state_budget_elements"]
                        )
                    ):
                        reasons.append(f"numerical state mismatch: {key}")
                    for field in ("jvp_validation", "endpoint_jvp_validation"):
                        check = record[field]
                        if (
                            not check["passed"]
                            or [c["epsilon"] for c in check["checks"]] != protocol["finite_difference_steps"]
                            or any(
                                not math.isfinite(c["relative_error"])
                                or c["relative_error"] > protocol["finite_difference_relative_error_tolerance"]
                                or c["relative_error"] < 0
                                for c in check["checks"]
                            )
                        ):
                            reasons.append(f"precision failure: {key}/{field}")
                    sampled_ticks = list(range(0, result["measured_ticks"], protocol["sample_every"]))
                    if sampled_ticks[-1] != result["measured_ticks"] - 1:
                        sampled_ticks.append(result["measured_ticks"] - 1)
                    if [sample["tick"] for sample in result["samples"]] != sampled_ticks:
                        reasons.append(f"missing samples: {key}")
                    expected_operations = _operation_names(record["architecture"])
                    for sample in result["samples"]:
                        point = sample["operating_point"]
                        fractions = point["operation_saturation"]
                        error = point["transition_reconstruction_max_error"]
                        if (
                            point["saturation_coverage"] != "complete_declared_operations"
                            or set(fractions) != expected_operations
                            or any(not math.isfinite(f) or not 0 <= f <= 1 for f in fractions.values())
                            or not math.isfinite(error)
                            or not 0 <= error <= protocol["reconstruction_absolute_tolerance"]
                            or any(
                                not math.isfinite(v)
                                for v in [
                                    sample["state_norm"],
                                    *point["component_norms"],
                                    *point["component_increment_norms"],
                                ]
                            )
                        ):
                            reasons.append(f"operating-point validation failure: {key}")
            if (
                not summary["all_passed"]
                or summary["records"] != count
                or summary["precision_and_finiteness_passed"] != count
            ):
                reasons.append(f"incomplete summary: {path}")
        except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
            reasons.append(f"invalid evidence: {path}: {exc}")
    if set(records) != expected:
        reasons.append(
            f"cell coverage mismatch: missing={len(expected - set(records))}, extra={len(set(records) - expected)}"
        )
    audits = []
    if not reasons:
        for architecture, seed in itertools.product(protocol["architectures"], protocol["development_seeds"]):
            horizon_means, labels = [], set()
            for horizon in protocol["horizons"]:
                rates = [
                    records[(horizon, architecture, seed, direction)]["result"]["rate"]
                    for direction in protocol["direction_seeds"]
                ]
                spread = max(rates) - min(rates)
                horizon_means.append(mean(rates))
                labels.update(_classification(rate, protocol["classification_tolerance_per_tick"]) for rate in rates)
                if spread > protocol["direction_spread_tolerance_per_tick"]:
                    reasons.append(f"direction dependence: {architecture}/{seed}/{horizon}")
            horizon_difference = max(horizon_means) - min(horizon_means)
            if horizon_difference > protocol["horizon_difference_tolerance_per_tick"] or len(labels) != 1:
                reasons.append(f"horizon/classification dependence: {architecture}/{seed}")
            audits.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "horizon_means": horizon_means,
                    "horizon_difference": horizon_difference,
                    "sampled_classifications": sorted(labels),
                }
            )
    return {
        "schema_version": 1,
        "approved": not reasons,
        "reasons": reasons,
        "records_checked": len(records),
        "protocol_sha256": digest,
        "implementation_hashes": hashes,
        "evidence_hashes": evidence,
        "run_dirs": [str(Path(p).resolve()) for p in run_dirs],
        "audit": audits,
        "test_verification": test_verification,
        "authorized_stage": "phase0_fresh_seed_directional_confirmation",
        "confirmed_scientific_claims": [],
    }


def validate_confirmation(
    protocol_bytes: bytes,
    decision: dict,
    hashes: dict[str, str],
    *,
    steps: int,
    width: int,
    burn_in: int,
    seeds: list[int],
) -> None:
    protocol = json.loads(protocol_bytes)
    expected_seeds = list(range(protocol["confirmation_seed_range"][0], protocol["confirmation_seed_range"][1] + 1))
    if (
        decision.get("approved") is not True
        or decision.get("protocol_sha256") != sha256(protocol_bytes)
        or decision.get("implementation_hashes") != hashes
        or seeds != expected_seeds
        or width != protocol["reference_hidden_size"]
        or burn_in != protocol["burn_in"]
        or steps not in protocol["horizons"]
    ):
        raise ValueError("confirmation decision or frozen settings mismatch")
    fresh = audit_development(
        [Path(p) for p in decision["run_dirs"]],
        protocol_bytes,
        hashes,
        test_verification=decision.get("test_verification"),
    )
    if not fresh["approved"] or fresh["evidence_hashes"] != decision["evidence_hashes"]:
        raise ValueError("confirmation decision evidence is stale or invalid")
