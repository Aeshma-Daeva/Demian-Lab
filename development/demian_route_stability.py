"""Fixed-forcing paired route-intervention measurements for Demian v1."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from development.closed_loop_comparators import ClosedLoopModelAdapter
from development.closed_loop_demian import InputFrame
from development.demian_channel_visibility import channel_pulse, full_state_pulse
from development.demian_v1_gate_state import DemianV1GateState, V1_CHANNELS


ROUTE_SCALE_ATTRIBUTES = {
    "carrier_to_slow": "carrier_to_slow_scale",
    "message_to_fast": "message_to_fast_scale",
    "carrier_to_fast": "carrier_to_fast_scale",
}


def _state_vector(adapter: ClosedLoopModelAdapter) -> torch.Tensor:
    return torch.cat([component.detach().reshape(-1) for component in adapter.state])


def _surface_vector(adapter: ClosedLoopModelAdapter) -> torch.Tensor:
    assert isinstance(adapter.model, DemianV1GateState)
    return adapter.model.state_vector(adapter.state).detach().reshape(-1)


def _channel_metrics(adapter: ClosedLoopModelAdapter, previous: tuple[torch.Tensor, ...]) -> dict[str, dict[str, float]]:
    return {
        "channel_norms": {
            channel: float(component.norm()) for channel, component in zip(V1_CHANNELS, adapter.state, strict=True)
        },
        "channel_increment_norms": {
            channel: float((component - prior).norm())
            for channel, component, prior in zip(V1_CHANNELS, adapter.state, previous, strict=True)
        },
    }


def _pre_pulse_features(adapter: ClosedLoopModelAdapter, previous: tuple[torch.Tensor, ...]) -> dict[str, float | None]:
    """State features available before the perturbation is applied."""
    _, slow, _, _, carrier, gate = adapter.state
    carrier_norm = carrier.norm()
    slow_norm = slow.norm()
    denominator = carrier_norm * slow_norm
    model = adapter.model
    assert isinstance(model, DemianV1GateState)
    return {
        "carrier_norm": float(carrier_norm),
        "slow_norm": float(slow_norm),
        "gate_norm": float(gate.norm()),
        "gate_pressure_mean": float(torch.sigmoid(gate).mean()),
        "state_velocity": float((_state_vector(adapter) - torch.cat([value.reshape(-1) for value in previous])).norm()),
        "carrier_slow_alignment": 0.0 if denominator == 0 else float(torch.dot(carrier.reshape(-1), slow.reshape(-1)) / denominator),
        "previous_transition_carrier_to_slow_norm": None
        if model._step_aux.get("carrier_to_slow_norm") is None
        else float(model._step_aux["carrier_to_slow_norm"]),
    }


def _pulse_state(
    state: tuple[torch.Tensor, ...], *, channel: str, amplitude: float, direction_seed: int, scope: str
) -> tuple[torch.Tensor, ...]:
    if scope == "channel":
        return channel_pulse(state, channel=channel, amplitude=amplitude, direction_seed=direction_seed, sign=1)
    if scope == "full_state":
        return full_state_pulse(
            state,
            amplitude=amplitude,
            reference_norm=state[V1_CHANNELS.index(channel)].norm(),
            direction_seed=direction_seed,
            sign=1,
        )
    raise ValueError("random pulse scope must be 'channel' or 'full_state'")


def _set_route_multiplier(adapter: ClosedLoopModelAdapter, *, route_id: str, multiplier: float) -> None:
    if route_id not in ROUTE_SCALE_ATTRIBUTES:
        raise ValueError(f"unsupported route intervention: {route_id}")
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError("route multiplier must be finite and nonnegative")
    assert isinstance(adapter.model, DemianV1GateState)
    attribute = ROUTE_SCALE_ATTRIBUTES[route_id]
    setattr(adapter.model, attribute, getattr(adapter.model, attribute) * multiplier)


def _measurement(
    *,
    tick: int,
    transitions_after_pulse: int,
    baseline: ClosedLoopModelAdapter,
    arm: ClosedLoopModelAdapter,
    previous_arm_state: tuple[torch.Tensor, ...],
    initial_pulse_norm: float | None,
) -> dict[str, object]:
    state_separation = float((_state_vector(arm) - _state_vector(baseline)).norm())
    surface_separation = float((_surface_vector(arm) - _surface_vector(baseline)).norm())
    return {
        "tick": tick,
        "transitions_after_pulse": transitions_after_pulse,
        "input_replay": True,
        "state_separation": state_separation,
        "surface_separation": surface_separation,
        "finite_time_state_gain": None if initial_pulse_norm is None else state_separation / initial_pulse_norm,
        **_channel_metrics(arm, previous_arm_state),
    }


def _interaction_measurement(
    *,
    tick: int,
    transitions_after_pulse: int,
    baseline: ClosedLoopModelAdapter,
    route_only: ClosedLoopModelAdapter,
    pulse_route: ClosedLoopModelAdapter,
    pulse_sham: ClosedLoopModelAdapter,
) -> dict[str, object]:
    """Difference-of-differences for a route's effect on the pulse response."""
    state_delta = (
        (_state_vector(pulse_route) - _state_vector(route_only))
        - (_state_vector(pulse_sham) - _state_vector(baseline))
    )
    surface_delta = (
        (_surface_vector(pulse_route) - _surface_vector(route_only))
        - (_surface_vector(pulse_sham) - _surface_vector(baseline))
    )
    return {
        "tick": tick,
        "transitions_after_pulse": transitions_after_pulse,
        "state_interaction_norm": float(state_delta.norm()),
        "surface_interaction_norm": float(surface_delta.norm()),
    }


def _recovery(trajectory: Sequence[dict[str, object]], *, threshold: float, dwell: int) -> dict[str, object]:
    consecutive = 0
    for point in trajectory:
        if int(point["transitions_after_pulse"]) == 0:
            continue
        if float(point["state_separation"]) <= threshold:
            consecutive += 1
            if consecutive >= dwell:
                return {"recovered": True, "censored": False, "tick": point["tick"], "dwell": dwell}
        else:
            consecutive = 0
    return {"recovered": False, "censored": True, "tick": None, "dwell": dwell}


def run_route_stability_cell(
    *,
    seed: int,
    hidden_size: int,
    frames: Sequence[InputFrame],
    channel: str,
    route_id: str,
    intervention_tick: int,
    amplitude: float,
    direction_seed: int,
    random_direction_seed: int,
    recovery_threshold: float,
    recovery_dwell: int,
    route_multiplier: float = 0.0,
    include_random_route_interaction: bool = False,
    random_pulse_scope: str = "channel",
    history_pulses: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """Replay one stream across a targeted lesion and two matched pulse controls."""
    if not frames:
        raise ValueError("at least one prerecorded input frame is required")
    if channel not in V1_CHANNELS:
        raise ValueError(f"unknown Demian channel: {channel}")
    if route_id not in ROUTE_SCALE_ATTRIBUTES:
        raise ValueError(f"unsupported route intervention: {route_id}")
    if not 0 <= intervention_tick < len(frames):
        raise ValueError("intervention tick must index frames")
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    if not math.isfinite(recovery_threshold) or recovery_threshold <= 0:
        raise ValueError("recovery threshold must be finite and positive")
    if recovery_dwell < 1:
        raise ValueError("recovery dwell must be positive")
    if not math.isfinite(route_multiplier) or route_multiplier < 0:
        raise ValueError("route multiplier must be finite and nonnegative")
    if random_pulse_scope not in {"channel", "full_state"}:
        raise ValueError("random pulse scope must be 'channel' or 'full_state'")
    normalized_history: list[dict[str, object]] = []
    for pulse in history_pulses:
        tick = pulse.get("tick")
        history_channel = pulse.get("channel")
        history_amplitude = pulse.get("amplitude")
        history_direction_seed = pulse.get("direction_seed")
        if not isinstance(tick, int) or not 0 <= tick < intervention_tick:
            raise ValueError("history pulse tick must precede intervention")
        if history_channel not in V1_CHANNELS or not isinstance(history_direction_seed, int):
            raise ValueError("history pulse must specify an implemented channel and direction seed")
        if not isinstance(history_amplitude, (int, float)) or not math.isfinite(history_amplitude) or history_amplitude <= 0:
            raise ValueError("history pulse amplitude must be finite and positive")
        normalized_history.append({"tick": tick, "channel": history_channel, "amplitude": float(history_amplitude), "direction_seed": history_direction_seed})

    baseline = ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size)
    arms = {
        "lesion_only": ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size),
        "targeted_lesion": ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size),
        "sham": ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size),
        "random_direction": ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=hidden_size),
    }
    if include_random_route_interaction:
        arms["random_targeted_lesion"] = ClosedLoopModelAdapter(
            architecture="demian", seed=seed, hidden_size=hidden_size
        )
    results: dict[str, dict[str, object]] = {
        "baseline": {"route_multiplier": 1.0, "trajectory": []},
        "lesion_only": {"route_multiplier": route_multiplier, "trajectory": []},
        "targeted_lesion": {"route_multiplier": route_multiplier, "trajectory": []},
        "sham": {"route_multiplier": 1.0, "trajectory": []},
        "random_direction": {"route_multiplier": 1.0, "trajectory": []},
    }
    if include_random_route_interaction:
        results["random_targeted_lesion"] = {"route_multiplier": route_multiplier, "trajectory": []}
    prior_states = {name: tuple(value.detach().clone() for value in adapter.state) for name, adapter in arms.items()}
    prior_baseline_state = tuple(value.detach().clone() for value in baseline.state)
    normalization_pulse_norm: float | None = None
    interaction_trajectory: list[dict[str, object]] = []
    random_interaction_trajectory: list[dict[str, object]] = []
    pre_pulse_features: dict[str, float | None] | None = None

    for tick, frame in enumerate(frames):
        for history_pulse in normalized_history:
            if tick == history_pulse["tick"]:
                for adapter in (baseline, *arms.values()):
                    adapter.state = channel_pulse(
                        adapter.state,
                        channel=str(history_pulse["channel"]),
                        amplitude=float(history_pulse["amplitude"]),
                        direction_seed=int(history_pulse["direction_seed"]),
                        sign=1,
                    )
        if tick == intervention_tick:
            pre_pulse_features = _pre_pulse_features(baseline, prior_baseline_state)
            _set_route_multiplier(arms["lesion_only"], route_id=route_id, multiplier=route_multiplier)
            _set_route_multiplier(arms["targeted_lesion"], route_id=route_id, multiplier=route_multiplier)
            _set_route_multiplier(arms["sham"], route_id=route_id, multiplier=1.0)
            if include_random_route_interaction:
                _set_route_multiplier(
                    arms["random_targeted_lesion"], route_id=route_id, multiplier=route_multiplier
                )
            normalization_pulse_norm = None
            for name, adapter in arms.items():
                if name == "lesion_only":
                    initial_norm = 0.0
                else:
                    pulse_seed = random_direction_seed if name.startswith("random_") else direction_seed
                    pulse_scope = random_pulse_scope if name.startswith("random_") else "channel"
                    adapter.state = _pulse_state(
                        adapter.state, channel=channel, amplitude=amplitude, direction_seed=pulse_seed, scope=pulse_scope
                    )
                    initial_norm = float((_state_vector(adapter) - _state_vector(baseline)).norm())
                    if name == "sham":
                        normalization_pulse_norm = initial_norm
                results[name]["initial_pulse_norm"] = initial_norm
            assert normalization_pulse_norm is not None
            for name, adapter in arms.items():
                trajectory = results[name]["trajectory"]
                assert isinstance(trajectory, list)
                trajectory.append(
                    _measurement(
                        tick=tick,
                        transitions_after_pulse=0,
                        baseline=baseline,
                        arm=adapter,
                        previous_arm_state=prior_states[name],
                        initial_pulse_norm=None if name == "lesion_only" else normalization_pulse_norm,
                    )
                )
                prior_states[name] = tuple(value.detach().clone() for value in adapter.state)
            baseline_trajectory = results["baseline"]["trajectory"]
            assert isinstance(baseline_trajectory, list)
            baseline_trajectory.append(
                _measurement(
                    tick=tick,
                    transitions_after_pulse=0,
                    baseline=baseline,
                    arm=baseline,
                    previous_arm_state=prior_baseline_state,
                    initial_pulse_norm=None,
                )
            )
            interaction_trajectory.append(
                _interaction_measurement(
                    tick=tick,
                    transitions_after_pulse=0,
                    baseline=baseline,
                    route_only=arms["lesion_only"],
                    pulse_route=arms["targeted_lesion"],
                    pulse_sham=arms["sham"],
                )
            )
            if include_random_route_interaction:
                random_interaction_trajectory.append(
                    _interaction_measurement(
                        tick=tick,
                        transitions_after_pulse=0,
                        baseline=baseline,
                        route_only=arms["lesion_only"],
                        pulse_route=arms["random_targeted_lesion"],
                        pulse_sham=arms["random_direction"],
                    )
                )

        prior_baseline_state = tuple(value.detach().clone() for value in baseline.state)
        prior_arm_states = {
            name: tuple(value.detach().clone() for value in adapter.state) for name, adapter in arms.items()
        }
        baseline.advance(frame.observation, frame.acknowledgement)
        for name, adapter in arms.items():
            adapter.advance(frame.observation, frame.acknowledgement)
            if tick >= intervention_tick:
                assert normalization_pulse_norm is not None
                trajectory = results[name]["trajectory"]
                assert isinstance(trajectory, list)
                trajectory.append(
                    _measurement(
                        tick=tick,
                        transitions_after_pulse=tick - intervention_tick + 1,
                        baseline=baseline,
                        arm=adapter,
                        previous_arm_state=prior_states[name],
                        initial_pulse_norm=None if name == "lesion_only" else normalization_pulse_norm,
                    )
                )
            prior_states[name] = prior_arm_states[name]
        if tick >= intervention_tick:
            baseline_trajectory = results["baseline"]["trajectory"]
            assert isinstance(baseline_trajectory, list)
            baseline_trajectory.append(
                _measurement(
                    tick=tick,
                    transitions_after_pulse=tick - intervention_tick + 1,
                    baseline=baseline,
                    arm=baseline,
                    previous_arm_state=prior_baseline_state,
                    initial_pulse_norm=None,
                )
            )
            interaction_trajectory.append(
                _interaction_measurement(
                    tick=tick,
                    transitions_after_pulse=tick - intervention_tick + 1,
                    baseline=baseline,
                    route_only=arms["lesion_only"],
                    pulse_route=arms["targeted_lesion"],
                    pulse_sham=arms["sham"],
                )
            )
            if include_random_route_interaction:
                random_interaction_trajectory.append(
                    _interaction_measurement(
                        tick=tick,
                        transitions_after_pulse=tick - intervention_tick + 1,
                        baseline=baseline,
                        route_only=arms["lesion_only"],
                        pulse_route=arms["random_targeted_lesion"],
                        pulse_sham=arms["random_direction"],
                    )
                )

    for name in arms:
        trajectory = results[name]["trajectory"]
        assert isinstance(trajectory, list)
        results[name]["recovery"] = _recovery(trajectory, threshold=recovery_threshold, dwell=recovery_dwell)
    payload = {
        "schema_version": 1,
        "experiment": "demian-fixed-forcing-route-stability",
        "seed": seed,
        "channel": channel,
        "route_id": route_id,
        "history_pulses": normalized_history,
        "config": {
            "hidden_size": hidden_size,
            "intervention_tick": intervention_tick,
            "amplitude": amplitude,
            "direction_seed": direction_seed,
            "random_direction_seed": random_direction_seed,
            "recovery_threshold": recovery_threshold,
            "recovery_dwell": recovery_dwell,
            "route_multiplier": route_multiplier,
            "random_pulse_scope": random_pulse_scope,
        },
        "pre_pulse_features": pre_pulse_features,
        "arms": results,
        "interaction_trajectory": interaction_trajectory,
    }
    if include_random_route_interaction:
        payload["random_interaction_trajectory"] = random_interaction_trajectory
    return payload
