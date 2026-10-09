"""Stream Phase 0 development checks before any confirmation or gain-matching run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter, build_budget_tracks
from development.closed_loop_world import ActionAcknowledgement, WorldObservation
from development.regime_characterization import StateMap, check_jvp, tangent_window
from development.regime_validation import sha256, snapshot_sources, source_hashes, validate_confirmation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "confirmation"), default="development")
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/REGIME_MATCHED_PROTOCOL.json")
    parser.add_argument("--reference-hidden-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--seeds")
    parser.add_argument(
        "--out", type=Path, required=True, help="New run directory; existing runs are never overwritten."
    )
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    args.reference_hidden_size = (
        args.reference_hidden_size if args.reference_hidden_size is not None else protocol["reference_hidden_size"]
    )
    args.steps = (
        args.steps if args.steps is not None else protocol["horizons"][-1 if args.mode == "confirmation" else 0]
    )
    args.burn_in = args.burn_in if args.burn_in is not None else protocol["burn_in"]
    seeds = (
        [int(value) for value in args.seeds.split(",")]
        if args.seeds
        else (
            protocol["development_seeds"]
            if args.mode == "development"
            else list(range(protocol["confirmation_seed_range"][0], protocol["confirmation_seed_range"][1] + 1))
        )
    )
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("require nonempty unique seeds")
    if args.mode == "development" and not set(seeds).issubset(protocol["development_seeds"]):
        parser.error("development permits only protocol development seeds")
    if not 0 <= args.burn_in < args.steps or args.reference_hidden_size < 1:
        parser.error("require positive width and 0 <= burn-in < steps")
    torch.set_num_threads(1)
    hashes = source_hashes()
    decision_digest = None
    if args.mode == "confirmation":
        if args.decision is None:
            parser.error("confirmation requires a validated --decision artifact")
        try:
            decision_bytes = args.decision.read_bytes()
            validate_confirmation(
                protocol_bytes,
                json.loads(decision_bytes),
                hashes,
                steps=args.steps,
                width=args.reference_hidden_size,
                burn_in=args.burn_in,
                seeds=seeds,
            )
            decision_digest = sha256(decision_bytes)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    metadata = {
        "schema_version": 1,
        "status": "phase0_instrumentation_validation"
        if args.mode == "development"
        else "phase0_directional_confirmation",
        "protocol": protocol,
        "protocol_sha256": sha256(protocol_bytes),
        "implementation_hashes": hashes,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "actual_config": {
            "reference_hidden_size": args.reference_hidden_size,
            "steps": args.steps,
            "burn_in": args.burn_in,
            "seeds": seeds,
        },
        "confirmation_gate": "validated_decision" if decision_digest else "pending_development_audit",
        "decision_sha256": decision_digest,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    if snapshot_sources(args.out) != hashes:
        raise RuntimeError("source changed during snapshot; abort")
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
                    end_state = state
                    for encoded in inputs[args.burn_in :]:
                        end_state = mapping.transition(end_state, encoded)
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
                        sample_every=protocol["sample_every"],
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
        "confirmation_gate": metadata["confirmation_gate"],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(args.out, flush=True)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
