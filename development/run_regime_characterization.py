"""Stream Phase 0 development checks before any confirmation or gain-matching run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter, build_budget_tracks
from development.closed_loop_world import ActionAcknowledgement, WorldObservation
from development.regime_characterization import StateMap, check_jvp, tangent_window


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/REGIME_MATCHED_PROTOCOL.json")
    parser.add_argument("--reference-hidden-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--burn-in", type=int, default=128)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument(
        "--out", type=Path, required=True, help="New run directory; existing runs are never overwritten."
    )
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or not set(seeds).issubset(protocol["development_seeds"]):
        parser.error("this validation command permits only protocol development seeds")
    if not 0 <= args.burn_in < args.steps or args.reference_hidden_size < 1:
        parser.error("require positive width and 0 <= burn-in < steps")
    torch.set_num_threads(1)
    hashes = {
        name: hashlib.sha256((ROOT / "development" / name).read_bytes()).hexdigest()
        for name in (
            "regime_characterization.py",
            "run_regime_characterization.py",
            "closed_loop_comparators.py",
            "closed_loop_demian.py",
            "demian_v1_gate_state.py",
            "substrate_lab.py",
        )
    }
    metadata = {
        "schema_version": 1,
        "status": "phase0_instrumentation_validation",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "implementation_hashes": hashes,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "actual_config": {
            "reference_hidden_size": args.reference_hidden_size,
            "steps": args.steps,
            "burn_in": args.burn_in,
            "seeds": seeds,
        },
        "confirmation_gate": "pending_horizon_direction_and_precision_validation",
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    specifications = build_budget_tracks(reference_hidden_size=args.reference_hidden_size, seed=seeds[0])["state"][
        "specifications"
    ]
    total, passed = 0, 0
    with (args.out / "records.jsonl").open("x", encoding="utf-8") as output:
        for architecture in protocol["architectures"]:
            for seed in seeds:
                adapter = ClosedLoopModelAdapter(
                    architecture=architecture, seed=seed, hidden_size=specifications[architecture]["hidden_size"]
                )
                mapping = StateMap(adapter)
                acknowledgement = ActionAcknowledgement("noop", None, None, None)
                inputs = [
                    adapter.encoder.encode(
                        WorldObservation(
                            tick=tick,
                            phase="cue" if tick == 0 else ("query" if tick == args.steps - 1 else "delay"),
                            cue_symbol=seed % 3 if tick == 0 else None,
                            query=tick == args.steps - 1,
                        ),
                        acknowledgement,
                    ).double()
                    for tick in range(args.steps)
                ]
                state = mapping.initial_state
                with torch.no_grad():
                    for encoded in inputs[: args.burn_in]:
                        state = mapping.transition(state, encoded)
                for direction_seed in protocol["direction_seeds"]:
                    precision = check_jvp(
                        mapping.transition,
                        state,
                        inputs[args.burn_in],
                        direction_seed=direction_seed,
                        epsilons=protocol["finite_difference_steps"],
                        tolerance=protocol["finite_difference_relative_error_tolerance"],
                    )
                    result = tangent_window(
                        mapping.transition,
                        mapping.readout,
                        state,
                        inputs[args.burn_in :],
                        direction_seed=direction_seed,
                        visibility_floor=protocol["readout_visibility_floor"],
                        operating_point=lambda z, x: mapping.operating_point(
                            z, x, protocol["nonlinearity_derivative_floor"]
                        ),
                    )
                    record = {
                        "architecture": architecture,
                        "seed": seed,
                        "hidden_size": adapter.hidden_size,
                        "state_elements": state.numel(),
                        "numerical_state_bytes": state.numel() * state.element_size(),
                        "budget_reference_dtype": "float32",
                        "measurement_dtype": "float64",
                        "protocol_sha256": metadata["protocol_sha256"],
                        "jvp_validation": precision,
                        "result": result,
                    }
                    # A second precision check at the measurement endpoint catches operating-region changes.
                    end_state = state
                    with torch.no_grad():
                        for encoded in inputs[args.burn_in :]:
                            end_state = mapping.transition(end_state, encoded)
                    record["endpoint_jvp_validation"] = check_jvp(
                        mapping.transition,
                        end_state,
                        inputs[-1],
                        direction_seed=direction_seed,
                        epsilons=protocol["finite_difference_steps"],
                        tolerance=protocol["finite_difference_relative_error_tolerance"],
                    )
                    output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    output.flush()
                    total += 1
                    passed += int(
                        precision["passed"] and record["endpoint_jvp_validation"]["passed"] and result["status"] == "ok"
                    )
                    print(
                        f"{architecture} seed={seed} direction={direction_seed} status={result['status']} rate={result.get('rate')}",
                        flush=True,
                    )
    summary = {
        "records": total,
        "precision_and_finiteness_passed": passed,
        "all_passed": passed == total,
        "confirmation_gate": "pending_horizon_direction_and_precision_validation",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
