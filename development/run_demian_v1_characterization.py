#!/usr/bin/env python3
"""Run the phase-one six-channel Demian characterization campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

from development.afp_v2_characterization import (
    AFPV2Config,
    characterize_model,
    classify_trace_result,
    null_control_results,
)
from development.demian_v1_measurement import V1TraceConfig, run_v1_measurement
from development.probe_demian_v1_continuation import run_v1_continuation_probe
from development.probe_v9_capsule_continuity import default_v9_five_channel


VALID_UPDATE_MODES = ("active", "gate_disabled", "gate_frozen")
VALID_HISTORY_CONDITIONS = ("uninterrupted", "surface_perturbed")


def _scaled_config(base: AFPV2Config, tail: int, multiplier: float) -> AFPV2Config:
    return replace(
        base,
        tail_steps=tail,
        surface_delta_tol=base.surface_delta_tol * multiplier,
        surface_drift_tol=base.surface_drift_tol * multiplier,
        hidden_delta_min=base.hidden_delta_min * multiplier,
        min_curvature_ratio=base.min_curvature_ratio * multiplier,
        continuation_gap_min=base.continuation_gap_min * multiplier,
        numerical_drift_multiplier=base.numerical_drift_multiplier * multiplier,
    )


def _robustness(
    trace: Any,
    gap: float | None,
    base: AFPV2Config,
    tails: Sequence[int],
    multipliers: Sequence[float],
    primary: str,
) -> dict[str, Any]:
    slices = {}
    for tail in tails:
        for multiplier in multipliers:
            key = f"tail={tail}:threshold={multiplier:g}"
            aligned = tail == base.tail_steps
            effective = _scaled_config(base, tail, multiplier)
            result = classify_trace_result(trace, gap if aligned else None, effective)
            label = result["full_state"]["classification"]
            slices[key] = {
                "classification": label,
                "changes_primary": label != primary,
                "continuation_evidence": "aligned" if aligned else "not_run_for_tail",
                "effective_config": asdict(effective),
                "varied_thresholds": [
                    "surface_delta_tol",
                    "surface_drift_tol",
                    "hidden_delta_min",
                    "min_curvature_ratio",
                    "continuation_gap_min",
                    "numerical_drift_multiplier",
                ],
                "fixed_controls": [
                    "hidden_stationarity_min_ratio",
                    "hidden_stationarity_max_ratio",
                    "min_velocity_coherence",
                ],
            }
    return slices


def run_demian_v1_campaign(
    *,
    seeds: Sequence[int],
    hidden_size: int = 16,
    steps: int = 512,
    perturb_step: int = 256,
    perturb_scale: float = 0.05,
    update_modes: Sequence[str] = VALID_UPDATE_MODES,
    history_conditions: Sequence[str] = VALID_HISTORY_CONDITIONS,
    robustness_tails: Sequence[int] = (24, 48, 96),
    threshold_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
    include_historical_control: bool = True,
) -> dict[str, Any]:
    """Run unselected six-channel dynamics and a labeled five-channel control."""

    if not seeds:
        raise ValueError("at least one seed is required")
    if (
        not robustness_tails
        or any(not isinstance(tail, int) or isinstance(tail, bool) or tail < 3 for tail in robustness_tails)
        or steps < max(robustness_tails)
    ):
        raise ValueError("robustness tails must be integers from 3 through steps")
    if not update_modes or any(mode not in VALID_UPDATE_MODES for mode in update_modes):
        raise ValueError("unsupported update mode")
    if not history_conditions or any(
        history not in VALID_HISTORY_CONDITIONS for history in history_conditions
    ):
        raise ValueError("unsupported history condition")
    if not 1 <= perturb_step <= steps:
        raise ValueError("perturb_step must fall inside the trace")
    if not threshold_multipliers or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
        for value in threshold_multipliers
    ):
        raise ValueError("threshold multipliers must be finite and positive")

    primary_tail = min(24, steps - 1)
    base = AFPV2Config(tail_steps=primary_tail)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for update_mode in update_modes:
            for history in history_conditions:
                trace_config = V1TraceConfig(
                    seed=seed,
                    hidden_size=hidden_size,
                    steps=steps,
                    parameter_provenance="unselected_random",
                    history_condition=history,
                    update_mode=update_mode,  # type: ignore[arg-type]
                    perturb_step=perturb_step if history == "surface_perturbed" else None,
                    perturb_scale=perturb_scale if history == "surface_perturbed" else 0.0,
                )
                trace = run_v1_measurement(trace_config)
                control = None
                gap = None
                control = run_v1_continuation_probe(
                    trace_config,
                    pause_steps=steps - primary_tail,
                    resume_steps=primary_tail,
                )
                gap = float(control["arms"]["body_surface"]["mean_step_gap_vs_uninterrupted"])
                classified = classify_trace_result(trace, gap, base)
                primary = str(classified["full_state"]["classification"])
                runs.append(
                    {
                        "run_id": f"demian_v1_gate_state:seed={seed}:{update_mode}:{history}",
                        "system": "demian_v1_gate_state",
                        "seed": seed,
                        "parameter_provenance": trace_config.parameter_provenance,
                        "history_condition": history,
                        "update_mode": update_mode,
                        "post_step_clamp": trace_config.post_step_clamp,
                        "classification": classified,
                        "continuation_control": control,
                        "robustness": _robustness(
                            trace,
                            gap,
                            base,
                            robustness_tails,
                            threshold_multipliers,
                            primary,
                        ),
                    }
                )

    if include_historical_control:
        for seed in seeds:
            for history in history_conditions:
                result = characterize_model(
                    "v9_five_channel",
                    default_v9_five_channel,
                    hidden_size=hidden_size,
                    seed=seed,
                    steps=steps,
                    perturb_step=perturb_step if history == "surface_perturbed" else None,
                    perturb_scale=perturb_scale if history == "surface_perturbed" else 0.0,
                    config=base,
                )
                runs.append(
                    {
                        "run_id": result["run_id"],
                        "system": "v9_five_channel",
                        "seed": seed,
                        "parameter_provenance": "historical_control",
                        "history_condition": history,
                        "update_mode": "not_applicable",
                        "post_step_clamp": "none",
                        "classification": {"full_state": result},
                        "continuation_control": None,
                        "robustness": {},
                    }
                )

    return {
        "schema_version": 1,
        "experiment": "demian-v1-six-channel-characterization",
        "campaign": {
            "seeds": list(seeds),
            "hidden_size": hidden_size,
            "steps": steps,
            "perturb_step": perturb_step,
            "perturb_scale": perturb_scale,
            "update_modes": list(update_modes),
            "history_conditions": list(history_conditions),
            "robustness_tails": list(robustness_tails),
            "threshold_multipliers": list(threshold_multipliers),
        },
        "afp_v2_config": asdict(base),
        "null_controls": null_control_results(base),
        "runs": runs,
        "evidence_status": {
            "observation": "Six-channel surface and internal regimes are measured separately.",
            "hypothesis": "Projected convergence may coexist with structured internal dynamics.",
            "control": "Tail-aligned exact, surface, channel, sham, and random resumes.",
            "interpretation": "Continuation effects establish causal influence, not task utility.",
            "untested_speculation": "Selected regimes may support memory or uncertainty tasks.",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--perturb-step", type=int, default=256)
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument(
        "--out",
        default="data/diagnostics/demian_v1_characterization_20261005/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_demian_v1_campaign(
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        hidden_size=args.hidden_size,
        steps=args.steps,
        perturb_step=args.perturb_step,
        perturb_scale=args.perturb_scale,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
