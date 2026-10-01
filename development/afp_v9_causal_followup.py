#!/usr/bin/env python3
"""Candidate-specific causal state surgery for native-v9 AFP checkpoints.

The exposed native-v9 surface is exactly the `fast` channel. This lets the
experiment preserve the current surface tensor while changing only hidden
`slow` and `control` state, then compare deterministic future trajectories.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from development.substrate_lab import DemianNativeV9Substrate


CONFIRMED_CANDIDATE_SEEDS = (104, 112, 123, 126)


@dataclass(frozen=True)
class CausalConfig:
    hidden_size: int = 32
    checkpoint_step: int = 384
    stale_lags: tuple[int, ...] = (16, 32, 64, 96)
    future_steps: int = 64
    impulse_scale: float = 0.15
    impulse_seed: int = 91_104
    seeds: tuple[int, ...] = tuple(range(102, 130))
    candidate_seeds: tuple[int, ...] = CONFIRMED_CANDIDATE_SEEDS


def _clone_state(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(part.detach().clone() for part in state)  # type: ignore[return-value]


def _rms(vector: torch.Tensor) -> float:
    flat = vector.detach().float().reshape(-1)
    return float(torch.linalg.vector_norm(flat).item() / math.sqrt(max(flat.numel(), 1)))


def _latent_vector(
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    fast, slow, control = state
    return torch.cat([fast.reshape(-1), slow.reshape(-1), control.reshape(-1)])


def _initial_model(
    hidden_size: int,
    seed: int,
) -> DemianNativeV9Substrate:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = DemianNativeV9Substrate(hidden_size)
    model.eval()
    return model


def _run_to_checkpoint(
    model: DemianNativeV9Substrate,
    *,
    config: CausalConfig,
    seed: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    needed_steps = {
        config.checkpoint_step,
        *(config.checkpoint_step - lag for lag in config.stale_lags),
    }
    if min(needed_steps) < 1:
        raise ValueError("stale lag reaches before step 1")

    with torch.no_grad():
        torch.manual_seed(seed)
        state = model.initial_state(1, torch.device("cpu"))
        checkpoints: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for step in range(1, config.checkpoint_step + 1):
            state = model.step(state)
            if step in needed_steps:
                checkpoints[step] = _clone_state(state)

    return checkpoints[config.checkpoint_step], checkpoints


def _intervention_states(
    current: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    checkpoints: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    config: CausalConfig,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    fast, slow, control = current
    arms: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
        "full_clone": _clone_state(current),
        "surface_only": (
            fast.detach().clone(),
            torch.zeros_like(slow),
            torch.zeros_like(control),
        ),
        "slow_reset": (
            fast.detach().clone(),
            torch.zeros_like(slow),
            control.detach().clone(),
        ),
        "control_reset": (
            fast.detach().clone(),
            slow.detach().clone(),
            torch.zeros_like(control),
        ),
    }

    for lag in config.stale_lags:
        past = checkpoints[config.checkpoint_step - lag]
        _, stale_slow, stale_control = past
        arms[f"stale_{lag}"] = (
            fast.detach().clone(),
            stale_slow.detach().clone(),
            stale_control.detach().clone(),
        )

    return arms


def _impulse(config: CausalConfig) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.impulse_seed)
    return config.impulse_scale * torch.tanh(
        torch.randn(config.hidden_size, generator=generator)
    )


def _future_gap(
    model: DemianNativeV9Substrate,
    *,
    reference_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    intervention_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: CausalConfig,
    future_condition: str,
) -> dict[str, Any]:
    reference_model = copy.deepcopy(model)
    intervention_model = copy.deepcopy(model)
    reference = _clone_state(reference_state)
    intervention = _clone_state(intervention_state)

    initial_surface_gap = _rms(
        reference_model.state_vector(reference)
        - intervention_model.state_vector(intervention)
    )
    initial_latent_gap = _rms(_latent_vector(reference) - _latent_vector(intervention))

    if initial_surface_gap > 1e-10:
        raise RuntimeError(
            f"surface_not_preserved:{future_condition}:{initial_surface_gap:.12g}"
        )

    if future_condition == "shared_impulse":
        impulse = _impulse(config)
        reference = reference_model.inject_coupling_message(reference, impulse, 1.0)
        intervention = intervention_model.inject_coupling_message(
            intervention, impulse, 1.0
        )
        post_impulse_gap = _rms(
            reference_model.state_vector(reference)
            - intervention_model.state_vector(intervention)
        )
        if post_impulse_gap > 1e-10:
            raise RuntimeError(
                f"shared_impulse_broke_surface_match:{post_impulse_gap:.12g}"
            )
    elif future_condition != "autonomous":
        raise ValueError(f"unknown future condition: {future_condition}")

    gaps: list[float] = []
    with torch.no_grad():
        for _ in range(config.future_steps):
            reference = reference_model.step(reference)
            intervention = intervention_model.step(intervention)
            gaps.append(
                _rms(
                    reference_model.state_vector(reference)
                    - intervention_model.state_vector(intervention)
                )
            )

    return {
        "initial_surface_gap": initial_surface_gap,
        "initial_latent_gap": initial_latent_gap,
        "first_step_gap": gaps[0] if gaps else 0.0,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "max_gap": max(gaps) if gaps else 0.0,
        "final_gap": gaps[-1] if gaps else 0.0,
        "gap_area": sum(gaps),
        "gaps": gaps,
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['group']}|{row['future_condition']}|{row['intervention']}"
        grouped.setdefault(key, []).append(row)

    output: dict[str, Any] = {}
    for key, group_rows in grouped.items():
        output[key] = {
            "n": len(group_rows),
            "mean_initial_latent_gap": sum(r["initial_latent_gap"] for r in group_rows)
            / len(group_rows),
            "mean_first_step_gap": sum(r["first_step_gap"] for r in group_rows)
            / len(group_rows),
            "mean_future_gap": sum(r["mean_gap"] for r in group_rows) / len(group_rows),
            "mean_max_gap": sum(r["max_gap"] for r in group_rows) / len(group_rows),
            "mean_final_gap": sum(r["final_gap"] for r in group_rows) / len(group_rows),
            "mean_gap_area": sum(r["gap_area"] for r in group_rows) / len(group_rows),
            "max_initial_surface_gap": max(r["initial_surface_gap"] for r in group_rows),
        }
    return output


def run_experiment(config: CausalConfig) -> dict[str, Any]:
    if config.future_steps < 1:
        raise ValueError("future_steps must be >= 1")
    if config.checkpoint_step <= max(config.stale_lags):
        raise ValueError("checkpoint_step must exceed every stale lag")
    if not config.seeds:
        raise ValueError("at least one seed is required")

    rows: list[dict[str, Any]] = []
    trajectories: dict[str, list[float]] = {}

    for seed in config.seeds:
        model = _initial_model(config.hidden_size, seed)
        current, checkpoints = _run_to_checkpoint(model, config=config, seed=seed)
        arms = _intervention_states(current, checkpoints, config=config)
        group = "candidate" if seed in config.candidate_seeds else "noncandidate"

        for future_condition in ("autonomous", "shared_impulse"):
            for intervention, intervention_state in arms.items():
                result = _future_gap(
                    model,
                    reference_state=current,
                    intervention_state=intervention_state,
                    config=config,
                    future_condition=future_condition,
                )
                row = {
                    "seed": seed,
                    "group": group,
                    "future_condition": future_condition,
                    "intervention": intervention,
                    **{key: value for key, value in result.items() if key != "gaps"},
                }
                rows.append(row)
                trajectories[
                    f"{seed}|{future_condition}|{intervention}"
                ] = result["gaps"]

    return {
        "protocol": "native-v9-afp-causal-v1",
        "config": asdict(config),
        "candidate_seeds": list(config.candidate_seeds),
        "claim_boundary": (
            "Same current fast surface, altered slow/control latent context, "
            "identical deterministic future. Future divergence measures causal "
            "relevance of latent state, not semantic memory or cognition."
        ),
        "rows": rows,
        "group_summary": _group_summary(rows),
        "trajectories": trajectories,
    }


def write_outputs(
    payload: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "v9_afp_causal_results.json"
    csv_path = out_dir / "v9_afp_causal_summary.csv"

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "seed",
        "group",
        "future_condition",
        "intervention",
        "initial_surface_gap",
        "initial_latent_gap",
        "first_step_gap",
        "mean_gap",
        "max_gap",
        "final_gap",
        "gap_area",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])

    return json_path, csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v9-afp-causal"))
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--checkpoint-step", type=int, default=384)
    parser.add_argument("--future-steps", type=int, default=64)
    parser.add_argument("--stale-lags", default="16,32,64,96")
    parser.add_argument("--seeds", default="102-129")
    parser.add_argument("--candidate-seeds", default="104,112,123,126")
    parser.add_argument("--impulse-scale", type=float, default=0.15)
    return parser.parse_args()


def _parse_int_set(spec: str) -> tuple[int, ...]:
    values: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(piece))
    return tuple(values)


def main() -> int:
    args = _parse_args()
    config = CausalConfig(
        hidden_size=args.hidden_size,
        checkpoint_step=args.checkpoint_step,
        stale_lags=_parse_int_set(args.stale_lags),
        future_steps=args.future_steps,
        impulse_scale=args.impulse_scale,
        seeds=_parse_int_set(args.seeds),
        candidate_seeds=_parse_int_set(args.candidate_seeds),
    )
    payload = run_experiment(config)
    json_path, csv_path = write_outputs(payload, args.output_dir)

    candidate_summary = {
        key: value
        for key, value in payload["group_summary"].items()
        if key.startswith("candidate|")
    }
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "candidate_summary": candidate_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
