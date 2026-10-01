#!/usr/bin/env python3
"""Longitudinal accumulating-fixed-point probe across the Demian lineage.

This experiment asks where the surface/latent separation appears, disappears,
or changes form across selected non-plastic Demian ancestors and the current
Demian v1 synthesis.

It deliberately distinguishes:
- a mathematical fixed point of the complete dynamical state;
- a fixed or nearly fixed exposed surface;
- continuing motion in the complete recurrent state tuple.

The probe is descriptive and falsifiable. It does not treat an AFP label as a
claim about cognition, memory, agency, or consciousness.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from development.demian_v1_gate_state import DemianV1GateState
from development.probe_v9_message_carrier_strange import (
    ExperimentalV9MessageCarrier,
    accumulator_kwargs,
)
from development.substrate_lab import (
    DemianNativeV2Substrate,
    DemianNativeV3Substrate,
    DemianNativeV8Substrate,
    DemianNativeV9Substrate,
)
from development.substrates.legacy import SelfLoopRunner


TUNED_V9 = {
    "init_scale": 0.15,
    "state_gain": 1.4,
    "slow_decay": 0.88,
    "control_decay": 0.58,
    "slow_readout_scale": 0.2,
    "control_to_fast_scale": 0.5,
    "fast_to_slow_gate_bias": 0.0,
}


@dataclass(frozen=True)
class LineageConfig:
    hidden_size: int = 32
    steps: int = 384
    tail_steps: int = 96
    seeds: tuple[int, ...] = (94, 95, 96, 97, 98, 99, 100, 101)
    perturb_step: int = 192
    perturb_scale: float = 0.35
    surface_rel_epsilon: float = 1e-3
    latent_rel_floor: float = 5e-4
    latent_surface_ratio_floor: float = 5.0
    exploratory_surface_rms_epsilon: float = 1e-3
    exploratory_latent_rms_floor: float = 1e-2
    exploratory_absolute_ratio_floor: float = 5.0


@dataclass(frozen=True)
class Variant:
    name: str
    factory: Callable[[int, int], torch.nn.Module]
    note: str


def _seeded(factory: Callable[[int], torch.nn.Module], hidden_size: int, seed: int) -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return factory(hidden_size)


def _variants() -> tuple[Variant, ...]:
    def v2(hidden: int, seed: int) -> torch.nn.Module:
        return _seeded(lambda size: DemianNativeV2Substrate(size), hidden, seed)

    def v3(hidden: int, seed: int) -> torch.nn.Module:
        return _seeded(lambda size: DemianNativeV3Substrate(size), hidden, seed)

    def v8(hidden: int, seed: int) -> torch.nn.Module:
        return _seeded(lambda size: DemianNativeV8Substrate(size), hidden, seed)

    def v9(hidden: int, seed: int) -> torch.nn.Module:
        return _seeded(lambda size: DemianNativeV9Substrate(size), hidden, seed)

    def v9_five_default(hidden: int, seed: int) -> torch.nn.Module:
        return _seeded(lambda size: ExperimentalV9MessageCarrier(size), hidden, seed)

    def v9_five_accumulator(hidden: int, seed: int) -> torch.nn.Module:
        kwargs = accumulator_kwargs(TUNED_V9)
        return _seeded(lambda size: ExperimentalV9MessageCarrier(size, **kwargs), hidden, seed)

    def v9_five_accumulator_zero_init(hidden: int, seed: int) -> torch.nn.Module:
        kwargs = {
            **accumulator_kwargs(TUNED_V9),
            "initial_message_scale": 0.0,
            "initial_carrier_scale": 0.0,
        }
        return _seeded(lambda size: ExperimentalV9MessageCarrier(size, **kwargs), hidden, seed)

    def v1(hidden: int, seed: int) -> torch.nn.Module:
        return _seeded(lambda size: DemianV1GateState(size), hidden, seed)

    return (
        Variant("native_v2", v2, "Explicit recruitment bias into slow."),
        Variant("native_v3", v3, "Adds endogenous tightness state."),
        Variant("native_v8", v8, "Compressed seven-channel comparison with tightness."),
        Variant("native_v9", v9, "Minimal fast/slow/control scaffold; fast is exposed surface."),
        Variant("v9_5ch_default", v9_five_default, "v9 plus message/carrier with default coupling."),
        Variant("v9_5ch_accumulator", v9_five_accumulator, "Historical tuned accumulator configuration."),
        Variant(
            "v9_5ch_accumulator_zero_init",
            v9_five_accumulator_zero_init,
            "Accumulator configuration with message/carrier initialized to zero, matching later evolution setup more closely.",
        ),
        Variant("demian_v1", v1, "Six-channel explicit gate-state synthesis."),
    )


def _state_vector(state: object) -> torch.Tensor:
    """Flatten the actual recurrent state tuple, avoiding state_components aliases."""

    if isinstance(state, torch.Tensor):
        return state.detach().reshape(-1)
    if isinstance(state, tuple):
        parts = []
        for item in state:
            if not isinstance(item, torch.Tensor):
                raise TypeError(f"unsupported state component: {type(item)!r}")
            parts.append(item.detach().reshape(-1))
        return torch.cat(parts)
    raise TypeError(f"unsupported recurrent state: {type(state)!r}")


def _rms(vector: torch.Tensor) -> float:
    flat = vector.detach().float().reshape(-1)
    return float(torch.linalg.vector_norm(flat).item() / math.sqrt(max(flat.numel(), 1)))


def _relative_velocity(mean_delta: float, mean_norm: float) -> float:
    return mean_delta / max(mean_norm, 1e-12)


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = float(len(values))
    mean_x = (n - 1.0) / 2.0
    mean_y = sum(values) / n
    numerator = sum((idx - mean_x) * (value - mean_y) for idx, value in enumerate(values))
    denominator = sum((idx - mean_x) ** 2 for idx in range(len(values)))
    return numerator / max(denominator, 1e-12)


def _perturb_state(state: object, *, scale: float, seed: int) -> object:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def perturb(tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)
        return tensor + scale * noise

    if isinstance(state, torch.Tensor):
        return perturb(state)
    if isinstance(state, tuple):
        return tuple(perturb(item) for item in state)
    raise TypeError(f"unsupported recurrent state: {type(state)!r}")


def _is_afp(
    surface_rel: float,
    latent_rel: float,
    *,
    surface_epsilon: float,
    latent_floor: float,
    ratio_floor: float,
) -> bool:
    return (
        surface_rel <= surface_epsilon
        and latent_rel >= latent_floor
        and latent_rel / max(surface_rel, 1e-12) >= ratio_floor
    )


def _run_tail_probe(
    model: torch.nn.Module,
    *,
    config: LineageConfig,
    seed: int,
    perturbed: bool,
) -> dict[str, Any]:
    model.eval()
    device = torch.device("cpu")

    with torch.no_grad():
        torch.manual_seed(seed)
        state = model.initial_state(1, device)  # type: ignore[attr-defined]

        surfaces: list[torch.Tensor] = []
        latents: list[torch.Tensor] = []
        surface_deltas: list[float] = []
        latent_deltas: list[float] = []
        surface_norms: list[float] = []
        latent_norms: list[float] = []

        prev_surface: torch.Tensor | None = None
        prev_latent: torch.Tensor | None = None

        for step in range(1, config.steps + 1):
            if perturbed and step == config.perturb_step:
                state = _perturb_state(
                    state,
                    scale=config.perturb_scale,
                    seed=seed + 50_000 + step,
                )

            state = model.step(state)  # type: ignore[attr-defined]
            surface = model.state_vector(state).detach().reshape(-1).cpu()  # type: ignore[attr-defined]
            latent = _state_vector(state).cpu()

            surfaces.append(surface)
            latents.append(latent)
            surface_norms.append(_rms(surface))
            latent_norms.append(_rms(latent))

            if prev_surface is not None and prev_latent is not None:
                surface_deltas.append(_rms(surface - prev_surface))
                latent_deltas.append(_rms(latent - prev_latent))
            else:
                surface_deltas.append(0.0)
                latent_deltas.append(0.0)

            prev_surface = surface
            prev_latent = latent

    tail = min(config.tail_steps, len(surfaces))
    surface_tail_deltas = surface_deltas[-tail:]
    latent_tail_deltas = latent_deltas[-tail:]
    surface_tail_norms = surface_norms[-tail:]
    latent_tail_norms = latent_norms[-tail:]

    mean_surface_delta = sum(surface_tail_deltas) / tail
    mean_latent_delta = sum(latent_tail_deltas) / tail
    mean_surface_norm = sum(surface_tail_norms) / tail
    mean_latent_norm = sum(latent_tail_norms) / tail
    surface_rel = _relative_velocity(mean_surface_delta, mean_surface_norm)
    latent_rel = _relative_velocity(mean_latent_delta, mean_latent_norm)

    absolute_ratio = mean_latent_delta / max(mean_surface_delta, 1e-12)
    exploratory_absolute_candidate = (
        mean_surface_delta <= config.exploratory_surface_rms_epsilon
        and mean_latent_delta >= config.exploratory_latent_rms_floor
        and absolute_ratio >= config.exploratory_absolute_ratio_floor
    )

    return {
        "surface_dim": int(surfaces[-1].numel()),
        "latent_dim": int(latents[-1].numel()),
        "surface_exposure_fraction": float(surfaces[-1].numel() / max(latents[-1].numel(), 1)),
        "tail_mean_surface_delta": mean_surface_delta,
        "tail_mean_latent_delta": mean_latent_delta,
        "tail_mean_surface_norm": mean_surface_norm,
        "tail_mean_latent_norm": mean_latent_norm,
        "surface_rel_velocity": surface_rel,
        "latent_rel_velocity": latent_rel,
        "latent_surface_velocity_ratio": latent_rel / max(surface_rel, 1e-12),
        "absolute_latent_surface_delta_ratio": absolute_ratio,
        "tail_surface_delta_slope": _slope(surface_tail_deltas),
        "tail_latent_delta_slope": _slope(latent_tail_deltas),
        "tail_surface_norm_slope": _slope(surface_tail_norms),
        "tail_latent_norm_slope": _slope(latent_tail_norms),
        "tail_surface_path_length": sum(surface_tail_deltas),
        "tail_latent_path_length": sum(latent_tail_deltas),
        "strict_afp_candidate": _is_afp(
            surface_rel,
            latent_rel,
            surface_epsilon=config.surface_rel_epsilon,
            latent_floor=config.latent_rel_floor,
            ratio_floor=config.latent_surface_ratio_floor,
        ),
        "exploratory_absolute_candidate": exploratory_absolute_candidate,
    }


def _historical_summary(
    variant: Variant,
    *,
    config: LineageConfig,
    seed: int,
    perturbed: bool,
) -> dict[str, Any]:
    model = variant.factory(config.hidden_size, seed)
    runner = SelfLoopRunner(model, device="cpu")
    _, summary, _ = runner.run(
        steps=config.steps,
        seed=seed,
        perturb_step=config.perturb_step if perturbed else None,
        perturb_scale=config.perturb_scale if perturbed else 0.0,
    )
    return {
        "attractor_type": summary.attractor_type,
        "interior_class": summary.interior_class,
        "attractor_confidence": summary.attractor_confidence,
        "mean_delta": summary.mean_delta,
        "mean_fast_delta": summary.mean_fast_delta,
        "mean_slow_delta": summary.mean_slow_delta,
        "mean_message_delta": summary.mean_message_delta,
        "max_slow_norm": summary.max_slow_norm,
        "max_message_norm": summary.max_message_norm,
        "mean_message_contraction": summary.mean_message_contraction,
        "max_message_contraction": summary.max_message_contraction,
    }


def _absolute_sensitivity(rows: list[dict[str, Any]], ratio_floor: float) -> list[dict[str, Any]]:
    output = []
    for surface_epsilon in (5e-3, 2e-3, 1e-3, 5e-4, 2e-4):
        for latent_floor in (1e-3, 5e-3, 1e-2, 2e-2, 5e-2):
            count = sum(
                row["tail_mean_surface_delta"] <= surface_epsilon
                and row["tail_mean_latent_delta"] >= latent_floor
                and row["absolute_latent_surface_delta_ratio"] >= ratio_floor
                for row in rows
            )
            output.append(
                {
                    "surface_rms_epsilon": surface_epsilon,
                    "latent_rms_floor": latent_floor,
                    "ratio_floor": ratio_floor,
                    "count": count,
                    "fraction": count / max(len(rows), 1),
                }
            )
    return output


def _sensitivity(rows: list[dict[str, Any]], ratio_floor: float) -> list[dict[str, Any]]:
    output = []
    for surface_epsilon in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5):
        for latent_floor in (1e-3, 3e-4, 1e-4, 1e-5, 1e-6):
            count = sum(
                _is_afp(
                    row["surface_rel_velocity"],
                    row["latent_rel_velocity"],
                    surface_epsilon=surface_epsilon,
                    latent_floor=latent_floor,
                    ratio_floor=ratio_floor,
                )
                for row in rows
            )
            output.append(
                {
                    "surface_epsilon": surface_epsilon,
                    "latent_floor": latent_floor,
                    "ratio_floor": ratio_floor,
                    "count": count,
                    "fraction": count / max(len(rows), 1),
                }
            )
    return output


def run_experiment(config: LineageConfig) -> dict[str, Any]:
    if config.steps < 4:
        raise ValueError("steps must be >= 4")
    if config.tail_steps < 2 or config.tail_steps > config.steps:
        raise ValueError("tail_steps must be in [2, steps]")
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if config.perturb_step < 1 or config.perturb_step > config.steps:
        raise ValueError("perturb_step must be in [1, steps]")

    payload: dict[str, Any] = {
        "protocol": "demian-afp-lineage-v1",
        "definition": {
            "afp": "exposed surface approximately stationary while complete recurrent state tuple continues moving",
            "boundary": "this is a projected/observable regime, not a mathematical fixed point of the complete dynamical system",
        },
        "config": asdict(config),
        "variants": {},
    }

    for variant in _variants():
        arms: dict[str, Any] = {}
        for arm_name, perturbed in (("clean", False), ("perturbed", True)):
            rows = []
            for seed in config.seeds:
                model = variant.factory(config.hidden_size, seed)
                metrics = _run_tail_probe(
                    model,
                    config=config,
                    seed=seed,
                    perturbed=perturbed,
                )
                historical = _historical_summary(
                    variant,
                    config=config,
                    seed=seed,
                    perturbed=perturbed,
                )
                rows.append(
                    {
                        "seed": seed,
                        **metrics,
                        "historical": historical,
                    }
                )

            strict_count = sum(bool(row["strict_afp_candidate"]) for row in rows)
            exploratory_absolute_count = sum(
                bool(row["exploratory_absolute_candidate"]) for row in rows
            )
            historical_accumulating = sum(
                row["historical"]["interior_class"] == "accumulating_fixed_point"
                for row in rows
            )
            arms[arm_name] = {
                "rows": rows,
                "aggregate": {
                    "strict_afp_count": strict_count,
                    "strict_afp_fraction": strict_count / len(rows),
                    "exploratory_absolute_count": exploratory_absolute_count,
                    "exploratory_absolute_fraction": exploratory_absolute_count / len(rows),
                    "historical_accumulating_count": historical_accumulating,
                    "historical_accumulating_fraction": historical_accumulating / len(rows),
                    "mean_surface_rel_velocity": sum(row["surface_rel_velocity"] for row in rows) / len(rows),
                    "mean_latent_rel_velocity": sum(row["latent_rel_velocity"] for row in rows) / len(rows),
                    "mean_latent_surface_velocity_ratio": sum(
                        row["latent_surface_velocity_ratio"] for row in rows
                    )
                    / len(rows),
                    "mean_absolute_latent_surface_delta_ratio": sum(
                        row["absolute_latent_surface_delta_ratio"] for row in rows
                    )
                    / len(rows),
                    "mean_tail_latent_norm_slope": sum(
                        row["tail_latent_norm_slope"] for row in rows
                    )
                    / len(rows),
                    "mean_tail_surface_delta_slope": sum(
                        row["tail_surface_delta_slope"] for row in rows
                    )
                    / len(rows),
                    "mean_surface_exposure_fraction": sum(
                        row["surface_exposure_fraction"] for row in rows
                    )
                    / len(rows),
                    "attractor_counts": _counts(row["historical"]["attractor_type"] for row in rows),
                    "interior_class_counts": _counts(row["historical"]["interior_class"] for row in rows),
                },
                "relative_sensitivity": _sensitivity(rows, config.latent_surface_ratio_floor),
                "absolute_sensitivity": _absolute_sensitivity(
                    rows, config.exploratory_absolute_ratio_floor
                ),
            }

        payload["variants"][variant.name] = {
            "note": variant.note,
            "clean": arms["clean"],
            "perturbed": arms["perturbed"],
        }

    return payload


def _counts(values: Any) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        key = str(value)
        output[key] = output.get(key, 0) + 1
    return output


def write_outputs(payload: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "afp_lineage_results.json"
    csv_path = out_dir / "afp_lineage_summary.csv"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "arm",
                "strict_afp_count",
                "strict_afp_fraction",
                "exploratory_absolute_count",
                "exploratory_absolute_fraction",
                "historical_accumulating_count",
                "historical_accumulating_fraction",
                "mean_surface_rel_velocity",
                "mean_latent_rel_velocity",
                "mean_latent_surface_velocity_ratio",
                "mean_absolute_latent_surface_delta_ratio",
                "mean_tail_latent_norm_slope",
                "mean_tail_surface_delta_slope",
                "mean_surface_exposure_fraction",
                "attractor_counts",
                "interior_class_counts",
            ],
        )
        writer.writeheader()
        for variant_name, variant in payload["variants"].items():
            for arm_name in ("clean", "perturbed"):
                aggregate = dict(variant[arm_name]["aggregate"])
                writer.writerow(
                    {
                        "variant": variant_name,
                        "arm": arm_name,
                        **{
                            **aggregate,
                            "attractor_counts": json.dumps(aggregate["attractor_counts"], sort_keys=True),
                            "interior_class_counts": json.dumps(
                                aggregate["interior_class_counts"], sort_keys=True
                            ),
                        },
                    }
                )

    return json_path, csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/afp-lineage"))
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--tail-steps", type=int, default=96)
    parser.add_argument("--seeds", default="94,95,96,97,98,99,100,101")
    parser.add_argument("--perturb-step", type=int, default=192)
    parser.add_argument("--perturb-scale", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    config = LineageConfig(
        hidden_size=args.hidden_size,
        steps=args.steps,
        tail_steps=args.tail_steps,
        seeds=seeds,
        perturb_step=args.perturb_step,
        perturb_scale=args.perturb_scale,
    )
    payload = run_experiment(config)
    json_path, csv_path = write_outputs(payload, args.output_dir)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "summary": {
                    variant: {
                        arm: data[arm]["aggregate"]
                        for arm in ("clean", "perturbed")
                    }
                    for variant, data in payload["variants"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
