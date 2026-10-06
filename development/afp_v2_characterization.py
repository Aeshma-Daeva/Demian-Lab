#!/usr/bin/env python3
"""Operational AFP-v2 classification for surface and full-state traces."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any, Sequence


Vector = Sequence[float]
Trace = Sequence[Vector]


@dataclass(frozen=True)
class AFPV2Config:
    tail_steps: int = 24
    windows: int = 4
    surface_delta_tol: float = 1e-2
    surface_drift_tol: float = 1e-4
    hidden_delta_min: float = 1e-3
    numerical_epsilon: float = 1.1920928955078125e-7
    numerical_drift_multiplier: float = 64.0
    zero_delta_tol: float = 1e-15
    hidden_stationarity_min_ratio: float = 0.8
    hidden_stationarity_max_ratio: float = 1.25
    min_curvature_ratio: float = 0.05
    min_velocity_coherence: float = 0.8
    continuation_gap_min: float = 1e-3


def _rms(vector: Sequence[float]) -> float:
    if not vector:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in vector) / len(vector))


def _deltas(trace: Trace) -> list[float]:
    return [
        _rms([float(right) - float(left) for left, right in zip(previous, current)])
        for previous, current in zip(trace, trace[1:])
    ]


def _curvatures(trace: Trace) -> list[float]:
    return [
        _rms(
            [
                float(after) - 2.0 * float(current) + float(before)
                for before, current, after in zip(first, second, third)
            ]
        )
        for first, second, third in zip(trace, trace[1:], trace[2:])
    ]


def _velocities(trace: Trace) -> list[list[float]]:
    return [
        [float(right) - float(left) for left, right in zip(previous, current)]
        for previous, current in zip(trace, trace[1:])
    ]


def _cosine(left: Vector, right: Vector) -> float:
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator <= 1e-20:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def _velocity_coherence(velocities: Trace, max_lag: int = 4) -> float:
    """Return the strongest signed mean directional correlation across short lags."""
    correlations = []
    for lag in range(1, min(max_lag, len(velocities) - 1) + 1):
        cosines = [
            _cosine(left, right)
            for left, right in zip(velocities, velocities[lag:])
        ]
        correlations.append(abs(sum(cosines) / len(cosines)))
    return max(correlations, default=0.0)


def _window_medians(values: Sequence[float], windows: int) -> list[float]:
    count = min(max(windows, 1), len(values))
    if count == 0:
        return []
    return [
        float(median(values[start:end]))
        for index in range(count)
        if (start := index * len(values) // count) < (end := (index + 1) * len(values) // count)
    ]


def _validate_trace(name: str, trace: Trace) -> None:
    if len(trace) < 3:
        raise ValueError(f"{name} trace requires at least 3 states")
    width = len(trace[0])
    if width == 0 or any(len(row) != width for row in trace):
        raise ValueError(f"{name} trace must contain equal-width, non-empty states")


def classify_afp_v2(
    surface_trace: Trace,
    hidden_trace: Trace,
    *,
    fixed_body_continuation_gap: float | None = None,
    config: AFPV2Config | None = None,
) -> dict[str, Any]:
    """Classify projected convergence separately from full-state dynamics."""
    cfg = config or AFPV2Config()
    _validate_trace("surface", surface_trace)
    _validate_trace("hidden", hidden_trace)
    if len(surface_trace) != len(hidden_trace):
        raise ValueError("surface and hidden traces must have equal length")

    tail_length = min(cfg.tail_steps, len(surface_trace))
    surface_tail = surface_trace[-tail_length:]
    hidden_tail = hidden_trace[-tail_length:]
    surface_deltas = _deltas(surface_tail)
    hidden_deltas = _deltas(hidden_tail)
    hidden_windows = _window_medians(hidden_deltas, cfg.windows)
    state_scale = max(1.0, max(_rms(row) for row in hidden_tail))
    numerical_floor = cfg.numerical_epsilon * cfg.numerical_drift_multiplier * state_scale
    surface_max_delta = max(surface_deltas, default=0.0)
    surface_drift_rate = _rms(
        [float(right) - float(left) for left, right in zip(surface_tail[0], surface_tail[-1])]
    ) / max(len(surface_tail) - 1, 1)
    hidden_max_delta = max(hidden_deltas, default=0.0)
    surface_small_step = surface_max_delta <= cfg.surface_delta_tol
    surface_converged = surface_small_step and surface_drift_rate <= cfg.surface_drift_tol
    active_all_windows = bool(hidden_windows) and all(
        value >= cfg.hidden_delta_min and value > numerical_floor for value in hidden_windows
    )
    stationarity_ratio = hidden_windows[-1] / max(hidden_windows[0], cfg.zero_delta_tol)
    hidden_stationary = (
        cfg.hidden_stationarity_min_ratio
        <= stationarity_ratio
        <= cfg.hidden_stationarity_max_ratio
    )
    persistent_hidden_change = active_all_windows and hidden_stationary

    curvature = _curvatures(hidden_tail)
    curvature_ratio = float(median(curvature)) / max(float(median(hidden_deltas)), cfg.zero_delta_tol)
    velocities = _velocities(hidden_tail)
    velocity_coherence = _velocity_coherence(velocities)
    structured_change = (
        persistent_hidden_change
        and curvature_ratio >= cfg.min_curvature_ratio
        and velocity_coherence >= cfg.min_velocity_coherence
    )
    continuation_relevant = (
        None
        if fixed_body_continuation_gap is None
        else fixed_body_continuation_gap >= cfg.continuation_gap_min
    )

    if not surface_small_step:
        classification = "surface_nonconvergent"
    elif not surface_converged:
        classification = "surface_small_step_drift"
    elif hidden_max_delta <= cfg.zero_delta_tol:
        classification = "surface_fixed_internal_fixed"
    elif hidden_max_delta <= numerical_floor:
        classification = "surface_fixed_numerical_drift"
    elif not active_all_windows:
        classification = (
            "surface_fixed_transient_internal"
            if any(value >= cfg.hidden_delta_min for value in hidden_windows)
            else "surface_fixed_weak_internal_change"
        )
    elif not hidden_stationary:
        classification = "surface_fixed_transient_internal"
    elif not structured_change:
        classification = (
            "surface_fixed_trivial_accumulation"
            if curvature_ratio < cfg.min_curvature_ratio
            else "surface_fixed_unstructured_change"
        )
    elif continuation_relevant is True:
        classification = "afp_v2_supported"
    elif continuation_relevant is False:
        classification = "afp_v2_continuation_null"
    else:
        classification = "afp_v2_candidate"

    return {
        "classification": classification,
        "surface_converged": surface_converged,
        "persistent_hidden_change": persistent_hidden_change,
        "structured_hidden_change": structured_change,
        "continuation_relevant": continuation_relevant,
        "metrics": {
            "surface_max_delta": surface_max_delta,
            "surface_drift_rate": surface_drift_rate,
            "hidden_max_delta": hidden_max_delta,
            "hidden_window_medians": hidden_windows,
            "hidden_stationarity_ratio": stationarity_ratio,
            "numerical_floor": numerical_floor,
            "curvature_ratio": curvature_ratio,
            "velocity_coherence": velocity_coherence,
            "fixed_body_continuation_gap": fixed_body_continuation_gap,
        },
        "config": asdict(cfg),
    }


def classify_internal_dynamics(
    hidden_trace: Trace,
    config: AFPV2Config | None = None,
) -> dict[str, Any]:
    """Classify internal dynamics independently of the exposed surface."""

    cfg = config or AFPV2Config()
    _validate_trace("hidden", hidden_trace)
    tail = hidden_trace[-min(cfg.tail_steps, len(hidden_trace)) :]
    deltas = _deltas(tail)
    windows = _window_medians(deltas, cfg.windows)
    state_scale = max(1.0, max(_rms(row) for row in tail))
    numerical_floor = cfg.numerical_epsilon * cfg.numerical_drift_multiplier * state_scale
    maximum = max(deltas, default=0.0)
    active = bool(windows) and all(
        value >= cfg.hidden_delta_min and value > numerical_floor for value in windows
    )
    stationarity_ratio = windows[-1] / max(windows[0], cfg.zero_delta_tol)
    stationary = cfg.hidden_stationarity_min_ratio <= stationarity_ratio <= cfg.hidden_stationarity_max_ratio
    curvature_ratio = float(median(_curvatures(tail))) / max(
        float(median(deltas)), cfg.zero_delta_tol
    )
    coherence = _velocity_coherence(_velocities(tail))
    persistent = active and stationary
    structured = (
        persistent
        and curvature_ratio >= cfg.min_curvature_ratio
        and coherence >= cfg.min_velocity_coherence
    )
    if maximum <= cfg.zero_delta_tol:
        classification = "internal_fixed"
    elif maximum <= numerical_floor:
        classification = "internal_numerical_drift"
    elif not active or not stationary:
        classification = "internal_transient"
    elif curvature_ratio < cfg.min_curvature_ratio:
        classification = "internal_trivial_accumulation"
    elif not structured:
        classification = "internal_unstructured_change"
    else:
        classification = "internal_structured_persistent"
    return {
        "classification": classification,
        "persistent": persistent,
        "structured": structured,
        "metrics": {
            "max_delta": maximum,
            "window_medians": windows,
            "stationarity_ratio": stationarity_ratio,
            "numerical_floor": numerical_floor,
            "curvature_ratio": curvature_ratio,
            "velocity_coherence": coherence,
        },
    }


def classify_trace_result(
    trace: Any,
    continuation_gap: float | None,
    config: AFPV2Config | None = None,
) -> dict[str, Any]:
    """Classify one recorded trace without conflating causality and task utility."""

    cfg = config or AFPV2Config()
    full_state = classify_afp_v2(
        trace.surfaces,
        trace.full_states,
        fixed_body_continuation_gap=continuation_gap,
        config=cfg,
    )
    channels = {
        name: classify_afp_v2(trace.surfaces, values, config=cfg)
        for name, values in trace.channels.items()
    }
    return {
        "surface": {
            "converged": full_state["surface_converged"],
            "max_delta": full_state["metrics"]["surface_max_delta"],
            "drift_rate": full_state["metrics"]["surface_drift_rate"],
        },
        "full_state": full_state,
        "channels": channels,
        "internal_regimes": {
            "full_state": classify_internal_dynamics(trace.full_states, cfg),
            "channels": {
                name: classify_internal_dynamics(values, cfg)
                for name, values in trace.channels.items()
            },
        },
        "functional_relevance": {
            "continuation_causal": full_state["continuation_relevant"],
            "task_utility": None,
        },
    }


def surface_matched_regimes(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Return surface classes that contain more than one internal regime."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = "surface_converged" if result["surface_converged"] else "surface_nonconvergent"
        grouped.setdefault(key, []).append(result)

    return {
        key: {
            "classifications": sorted({str(row["classification"]) for row in rows}),
            "run_ids": sorted(str(row["run_id"]) for row in rows),
        }
        for key, rows in grouped.items()
        if len({str(row["classification"]) for row in rows}) > 1
    }


def characterize_model(
    label: str,
    factory: Any,
    *,
    hidden_size: int,
    seed: int,
    steps: int,
    fixed_body_continuation_gap: float | None = None,
    perturb_step: int | None = None,
    perturb_scale: float = 0.0,
    device_name: str = "cpu",
    config: AFPV2Config | None = None,
) -> dict[str, Any]:
    """Run one system and classify its surface, full state, and channels."""
    import torch

    cfg = config or AFPV2Config()
    device = torch.device(device_name)
    torch.manual_seed(seed)
    model = factory(hidden_size).to(device=device, dtype=torch.float32)
    state = model.initial_state(1, device)
    surfaces: list[list[float]] = []
    full_states: list[list[float]] = []
    channels: dict[str, list[list[float]]] = {}

    with torch.no_grad():
        for step_index in range(1, steps + 1):
            if perturb_step == step_index and perturb_scale > 0.0:
                surface = model.state_vector(state)
                generator = torch.Generator(device=device).manual_seed(seed + step_index)
                noise = torch.randn(surface.shape, generator=generator, device=device, dtype=surface.dtype)
                state = model.write_surface_state(state, surface + perturb_scale * noise)
            state = model.step(state)
            surface = model.state_vector(state).reshape(-1).detach().float().cpu().tolist()
            components = {
                name: tensor.reshape(-1).detach().float().cpu().tolist()
                for name, tensor in model.state_components(state).items()
            }
            surfaces.append(surface)
            full_states.append([value for values in components.values() for value in values])
            for name, values in components.items():
                channels.setdefault(name, []).append(values)

    result = classify_afp_v2(
        surfaces,
        full_states,
        fixed_body_continuation_gap=fixed_body_continuation_gap,
        config=cfg,
    )
    result.pop("config", None)
    variant = "perturbed" if perturb_step is not None else "baseline"
    channel_regimes: dict[str, dict[str, Any]] = {}
    for name, trace in channels.items():
        channel_result = classify_afp_v2(surfaces, trace, config=cfg)
        channel_result.pop("config", None)
        channel_regimes[name] = channel_result
    result.update(
        {
            "run_id": f"{label}:seed={seed}:{variant}",
            "system": label,
            "seed": seed,
            "variant": variant,
            "steps": steps,
            "channel_regimes": channel_regimes,
        }
    )
    return result


def null_control_results(config: AFPV2Config | None = None) -> dict[str, dict[str, Any]]:
    """Evaluate the three explicit AFP-v2 exclusion controls."""
    cfg = config or AFPV2Config()
    fixed_surface = [[1.0, -1.0] for _ in range(32)]
    structured = [
        [math.cos(step * math.pi / 8.0), math.sin(step * math.pi / 8.0)]
        for step in range(32)
    ]
    rng = random.Random(7)
    random_walk = [[0.0, 0.0]]
    for _ in range(31):
        random_walk.append(
            [
                random_walk[-1][0] + rng.gauss(0.0, 0.02),
                random_walk[-1][1] + rng.gauss(0.0, 0.02),
            ]
        )
    cases = {
        "numerical_drift": (
            fixed_surface,
            [[0.2 + step * 1e-12, -0.2] for step in range(32)],
        ),
        "ordinary_transient": (
            fixed_surface,
            [[0.2 * step, 0.0] if step < 16 else [3.0, 0.0] for step in range(32)],
        ),
        "trivial_accumulation": (
            fixed_surface,
            [[0.02 * step, -0.01 * step] for step in range(32)],
        ),
        "small_step_surface_drift": (
            [[0.005 * step, 0.0] for step in range(32)],
            structured,
        ),
        "stochastic_accumulation": (fixed_surface, random_walk),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (surface, trace) in cases.items():
        result = classify_afp_v2(surface, trace, config=cfg)
        result.pop("config", None)
        results[name] = result
    return results


def build_campaign_summary(
    runs: Sequence[dict[str, Any]],
    config: AFPV2Config | None = None,
) -> dict[str, Any]:
    """Build the compact evidence artifact from classified runs."""
    cfg = config or AFPV2Config()
    classifications: dict[str, int] = {}
    by_variant: dict[str, dict[str, int]] = {}
    for run in runs:
        name = str(run["classification"])
        classifications[name] = classifications.get(name, 0) + 1
        variant = str(run.get("variant", "baseline"))
        counts = by_variant.setdefault(variant, {})
        counts[name] = counts.get(name, 0) + 1
    baseline_counts = by_variant.get("baseline", {})
    supported = baseline_counts.get("afp_v2_supported", 0)
    baseline_total = sum(baseline_counts.values())
    return {
        "schema_version": 1,
        "experiment": "afp-v2-characterization",
        "definition": {
            "system": "z_(t+1) = F(z_t, x_t); y_t = R(z_t)",
            "surface_convergence": "delta_y_t -> 0",
            "afp_v2": "surface convergence plus persistent structured full-state change",
        },
        "config": asdict(cfg),
        "classification_counts": dict(sorted(classifications.items())),
        "classification_counts_by_variant": {
            variant: dict(sorted(counts.items()))
            for variant, counts in sorted(by_variant.items())
        },
        "null_controls": null_control_results(cfg),
        "surface_matched_regimes": surface_matched_regimes(runs),
        "runs": list(runs),
        "evidence_status": {
            "observation": f"{supported}/{baseline_total} baseline runs satisfy the operational AFP-v2 definition and fixed-body control.",
            "hypothesis": "A converged readout can coexist with continuation-relevant full-state dynamics.",
            "control": "body_surface restores model weights and exposed state while zeroing other recurrent channels.",
            "interpretation": (
                f"{supported}/{baseline_total} baseline runs pass; a nonzero body_surface gap only "
                "establishes exposed-state insufficiency."
                if supported
                else "No tested baseline passes; a nonzero body_surface gap only establishes exposed-state insufficiency."
            ),
            "untested_speculation": "Distinct regimes have reproducible Jacobian or finite-time Lyapunov signatures and semantic function.",
        },
    }


def tail_control_window(steps: int, tail_steps: int) -> dict[str, Any]:
    """Place the continuation intervention immediately before the classified tail."""
    resume_steps = min(tail_steps, steps - 1)
    pause_steps = steps - resume_steps
    interval = [pause_steps + 1, steps]
    return {
        "pause_steps": pause_steps,
        "resume_steps": resume_steps,
        "classified_steps": interval,
        "control_steps": interval,
    }


def run_default_campaign(
    *,
    seeds: Sequence[int],
    hidden_size: int = 16,
    steps: int = 512,
    perturb_scale: float = 0.05,
    device_name: str = "cpu",
    config: AFPV2Config | None = None,
) -> dict[str, Any]:
    """Run current v9, five-channel, and historical dual-GRU systems."""
    from development.probe_v9_capsule_continuity import (
        default_v9_five_channel,
        run_capsule_probe,
    )
    from development.substrates.legacy import DemianNativeV9Substrate, _make_substrate

    cfg = config or AFPV2Config()
    specifications = (
        (
            "demian_native_v9",
            lambda size: DemianNativeV9Substrate(size),
            ("fast", "slow", "control"),
        ),
        (
            "v9_five_channel",
            default_v9_five_channel,
            ("fast", "slow", "control", "message", "carrier"),
        ),
        (
            "dual_gru_v3b",
            lambda size: _make_substrate("dual_gru_v3b", size, None),
            ("fast", "slow", "message"),
        ),
    )
    runs: list[dict[str, Any]] = []
    control_window = tail_control_window(steps, cfg.tail_steps)
    for label, factory, component_names in specifications:
        for seed in seeds:
            control = run_capsule_probe(
                label,
                factory,
                component_names,
                hidden_size,
                seed,
                control_window["pause_steps"],
                control_window["resume_steps"],
                device_name,
            )
            continuation_gap = float(
                control["arms"]["body_surface"]["mean_step_gap_vs_uninterrupted"]
            )
            baseline = characterize_model(
                label,
                factory,
                hidden_size=hidden_size,
                seed=seed,
                steps=steps,
                fixed_body_continuation_gap=continuation_gap,
                device_name=device_name,
                config=cfg,
            )
            baseline["fixed_body_control"] = control["fixed_body_control"]
            baseline["fixed_body_control"].update(control_window)
            baseline["component_restore_gaps"] = {
                name: control["arms"][f"{name}_only"]["mean_step_gap_vs_uninterrupted"]
                for name in component_names
            }
            runs.append(baseline)
            runs.append(
                characterize_model(
                    label,
                    factory,
                    hidden_size=hidden_size,
                    seed=seed,
                    steps=steps,
                    perturb_step=steps // 2,
                    perturb_scale=perturb_scale,
                    device_name=device_name,
                    config=cfg,
                )
            )
    summary = build_campaign_summary(runs, cfg)
    summary["campaign"] = {
        "systems": [item[0] for item in specifications],
        "seeds": list(seeds),
        "hidden_size": hidden_size,
        "steps": steps,
        "perturb_scale": perturb_scale,
    }
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="94,95,96")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--perturb-scale", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="data/diagnostics/afp_v2_characterization_20261004/summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    payload = run_default_campaign(
        seeds=seeds,
        hidden_size=args.hidden_size,
        steps=args.steps,
        perturb_scale=args.perturb_scale,
        device_name=args.device,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
