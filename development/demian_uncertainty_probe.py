"""Stateless public-evidence encoder and auditable fixed-weight Demian runner."""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch

from development.demian_uncertainty_world import EvidenceObservation
from development.demian_v1_gate_state import DemianV1GateState, V1_CHANNELS, capture_v1_runtime, restore_v1_runtime


_PHASES = ("evidence", "delay", "resolve", "complete")


def encode_evidence(observation: EvidenceObservation, hidden_size: int, *, scale: float = 0.25) -> torch.Tensor:
    """Fixed collision-free slots: phase(4), symbol(2), reliability(3), observed(1)."""
    if hidden_size < 10:
        raise ValueError("uncertainty encoder requires hidden_size >= 10")
    if observation.phase not in _PHASES:
        raise ValueError("unknown observation phase")
    encoded = torch.zeros((1, hidden_size), dtype=torch.float32)
    encoded[0, _PHASES.index(observation.phase)] = 1.0
    if observation.symbol is None:
        if observation.reliability is not None:
            raise ValueError("missing evidence cannot carry reliability")
    else:
        if observation.symbol not in {0, 1} or observation.reliability not in {0.5, 0.75, 0.95}:
            raise ValueError("invalid public evidence")
        encoded[0, 4 + observation.symbol] = 1.0
        encoded[0, 6 + (0.5, 0.75, 0.95).index(observation.reliability)] = 1.0
        encoded[0, 9] = 1.0
    return encoded * scale


def _state_hash(state: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for value in state:
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def run_uncertainty_episode(
    *,
    model_seed: int,
    hidden_size: int,
    public_frames: Sequence[EvidenceObservation],
    checkpoint_phases: Sequence[str] = ("evidence", "resolve"),
    capture_route_tensors: bool = False,
) -> dict[str, object]:
    """Run public frames only; evaluator truth/labels never enter this interface."""
    if not public_frames:
        raise ValueError("at least one public frame is required")
    invalid_phases = set(checkpoint_phases) - set(_PHASES)
    if invalid_phases:
        raise ValueError(f"unknown checkpoint phases: {sorted(invalid_phases)}")
    torch.manual_seed(model_seed)
    model = DemianV1GateState(hidden_size, trace_routes=True)
    state = model.initial_state(1, torch.device("cpu"))
    snapshots: dict[str, object] = {}
    measurements: list[dict[str, object]] = []
    route_tensors: list[dict[str, torch.Tensor]] = []
    hashes: list[str] = []
    for observation in public_frames:
        before = tuple(value.detach().clone() for value in state)
        state = (state[0] + encode_evidence(observation, hidden_size), *state[1:])
        state = model.step(state)
        route = model.route_trace()
        if capture_route_tensors:
            if route is None:
                raise RuntimeError("route capture requested but model did not emit a route trace")
            route_tensors.append({name: value.detach().clone() for name, value in route.values.items()})
        measurements.append({
            "tick": observation.tick,
            "phase": observation.phase,
            "state_norm": float(torch.cat([part.reshape(-1) for part in state]).norm().detach()),
            "channel_norms": {name: float(value.norm().detach()) for name, value in zip(V1_CHANNELS, state, strict=True)},
            "increment_norm": float(torch.cat([(now - prior).reshape(-1) for now, prior in zip(state, before, strict=True)]).norm().detach()),
            "gate_pressure_mean": float(torch.sigmoid(state[-1]).mean().detach()),
            "carrier_to_slow_norm": None if route is None else float(route.values["carrier_to_slow"].norm().detach()),
        })
        if observation.phase in checkpoint_phases:
            snapshots[f"{observation.phase}:{observation.tick}"] = capture_v1_runtime(model, state)
        hashes.append(_state_hash(state))
    final_snapshot = capture_v1_runtime(model, state)
    clone = DemianV1GateState(hidden_size, trace_routes=True)
    restored = restore_v1_runtime(clone, final_snapshot)
    return {
        "state_hashes": hashes,
        "measurements": measurements,
        "route_tensors": route_tensors,
        "checkpoints": snapshots,
        "replay_exact": _state_hash(restored) == hashes[-1],
        "input_count": len(public_frames),
    }
