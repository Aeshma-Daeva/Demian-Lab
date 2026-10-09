"""Phase 0b: frozen development grid after a successful confirmation audit."""

import argparse
import json
from pathlib import Path

import torch

from development.audit_regime_confirmation import audit_campaign
from development.closed_loop_comparators import ClosedLoopModelAdapter
from development.closed_loop_world import ActionAcknowledgement, WorldObservation
from development.regime_characterization import StateMap, check_jvp, tangent_window
from development.regime_gain_calibration import select_gain
from development.regime_validation import ROOT, sha256, snapshot_sources, source_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/REGIME_GAIN_PROTOCOL.json")
    parser.add_argument("--out", required=True, type=Path, help="Fresh calibration directory")
    args = parser.parse_args()
    audit_bytes = args.audit.read_bytes()
    audit = json.loads(audit_bytes)
    campaign = Path(audit["campaign"])
    fresh = audit_campaign(campaign)
    if not audit["approved"] or not fresh["approved"] or fresh["evidence_hashes"] != audit["evidence_hashes"]:
        parser.error("completed Phase 0 audit is invalid or stale")
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    base = json.loads((campaign / "protocol.json").read_bytes())
    if (
        protocol["seeds"] != base["development_seeds"]
        or protocol["directions"] != base["direction_seeds"]
        or protocol["horizons"] != base["horizons"]
        or protocol["grid"] != base["rnn_weight_hh_multiplier_grid"]
        or protocol["state_elements"] != base["state_budget_elements"]
        or protocol["burn_in"] != base["burn_in"]
        or protocol["reference_hidden_size"] != base["reference_hidden_size"]
    ):
        parser.error("calibration changes the original development/budget protocol")
    hashes = source_hashes()
    # Historical archives remain auditable after adding a new runner; active dynamics must not change.
    for name, digest in audit["implementation_hashes"].items():
        if name.startswith("development/") and name not in {
            "development/regime_validation.py",
            "development/run_regime_characterization.py",
            "development/run_regime_campaign.py",
        }:
            if hashes.get(name) != digest:
                parser.error(f"active source changed since Phase 0: {name}")
    targets = {}
    target_hashes = {}
    for horizon in protocol["horizons"]:
        path = campaign / f"development_h{horizon}/records.jsonl"
        target_hashes[str(path)] = sha256(path.read_bytes())
        values = [json.loads(line) for line in path.read_text().splitlines()]
        rates = [r["result"]["rate"] for r in values if r["architecture"] == "demian"]
        targets[horizon] = sum(rates) / len(rates)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if snapshot_sources(args.out) != hashes:
        raise RuntimeError("source changed during snapshot")
    manifest = {
        "protocol": protocol,
        "protocol_sha256": sha256(raw),
        "phase0_audit_sha256": sha256(audit_bytes),
        "implementation_hashes": hashes,
        "targets": targets,
        "target_evidence_hashes": target_hashes,
        "torch_version": torch.__version__,
        "threads": 1,
        "later_phases_authorized": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out / "status.json").write_text(json.dumps({"stage": "calibration_running"}) + "\n")
    records = []
    with (args.out / "records.jsonl").open("x") as output:
        for horizon in protocol["horizons"]:
            for gain in protocol["grid"]:
                for seed in protocol["seeds"]:
                    adapter = ClosedLoopModelAdapter(
                        architecture="rnn", seed=seed, hidden_size=protocol["state_elements"]
                    )
                    mapping = StateMap(adapter, recurrent_multiplier=gain)
                    ack = ActionAcknowledgement("noop", None, None, None)
                    inputs = [
                        adapter.encoder.encode(
                            WorldObservation(
                                tick=tick,
                                phase="cue" if tick == 0 else "query" if tick == horizon - 1 else "delay",
                                cue_symbol=seed % 3 if tick == 0 else None,
                                query=tick == horizon - 1,
                            ),
                            ack,
                        ).double()
                        for tick in range(horizon)
                    ]
                    state = mapping.initial_state
                    with torch.no_grad():
                        for encoded in inputs[: protocol["burn_in"]]:
                            state = mapping.transition(state, encoded)
                        endpoint = state
                        for encoded in inputs[protocol["burn_in"] :]:
                            endpoint = mapping.transition(endpoint, encoded)
                    for direction in protocol["directions"]:
                        checks = [
                            check_jvp(
                                mapping.transition,
                                z,
                                x,
                                direction_seed=direction,
                                epsilons=base["finite_difference_steps"],
                                tolerance=base["finite_difference_relative_error_tolerance"],
                            )
                            for z, x in [(state, inputs[protocol["burn_in"]]), (endpoint, inputs[-1])]
                        ]
                        result = tangent_window(
                            mapping.transition,
                            mapping.readout,
                            state,
                            inputs[protocol["burn_in"] :],
                            direction_seed=direction,
                            sample_every=base["sample_every"],
                            visibility_floor=base["readout_visibility_floor"],
                            operating_point=lambda z, x: mapping.operating_point(
                                z, x, base["nonlinearity_derivative_floor"]
                            ),
                        )
                        record = {
                            "multiplier": gain,
                            "seed": seed,
                            "direction_seed": direction,
                            "horizon": horizon,
                            "rate": result["rate"],
                            "passed": result["status"] == "ok" and all(c["passed"] for c in checks),
                            "checks": checks,
                            "result": result,
                        }
                        output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                        output.flush()
                        records.append(record)
                        print(
                            f"horizon={horizon} gain={gain} seed={seed} direction={direction} passed={record['passed']}",
                            flush=True,
                        )
    result = select_gain(records, targets, protocol)
    (args.out / "selection.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.out / "status.json").write_text(
        json.dumps({"stage": "calibration_collected", "selection_status": result["status"]}) + "\n"
    )


if __name__ == "__main__":
    main()
