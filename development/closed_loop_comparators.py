"""Common untrained closed-loop adapters for small recurrent comparators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from time import perf_counter
from typing import Literal, Sequence

import torch
from torch import nn

from development.closed_loop_demian import FrozenObservationEncoder, InputFrame
from development.closed_loop_world import (
    ActionAcceptance,
    ActionAcknowledgement,
    ActionProposal,
    CueDelayQueryWorld,
    WorldConnector,
    WorldObservation,
    WorldRuntimeSnapshot,
    capture_world_runtime,
    restore_world_runtime,
)
from development.demian_v1_gate_state import DemianV1GateState, V1State


Architecture = Literal["mlp", "rnn", "gru", "demian", "demian_route_ablation"]
Condition = Literal["baseline", "internal", "environment", "both"]
EnvironmentExposure = Literal["natural", "forced_read"]

PERTURBATION_RATE = 0.01
INTERNAL_AMPLITUDE = 0.01


@dataclass(frozen=True)
class AdapterStep:
    surface: list[float]
    proposal: ActionProposal


@dataclass(frozen=True)
class ModelRuntimeSnapshot:
    state: tuple[torch.Tensor, ...]
    configuration_fingerprint: str


@dataclass(frozen=True)
class ComparatorRecord:
    observation: WorldObservation
    executed_acknowledgement: ActionAcknowledgement
    input_frame: InputFrame
    proposal: ActionProposal
    acceptance: ActionAcceptance
    surface: list[float]


@dataclass(frozen=True)
class ComparatorRuntimeSnapshot:
    adapter: ModelRuntimeSnapshot
    world: WorldRuntimeSnapshot
    configuration_fingerprint: str


class ClosedLoopModelAdapter:
    """One declared state owner with a shared input encoder and action head."""

    def __init__(self, *, architecture: Architecture, seed: int, hidden_size: int, symbol_count: int = 3) -> None:
        self.architecture = architecture
        self.seed = seed
        self.hidden_size = hidden_size
        self.symbol_count = symbol_count
        torch.manual_seed(seed)
        self.encoder = FrozenObservationEncoder(hidden_size=hidden_size, symbol_count=symbol_count)
        self.model: nn.Module
        self.state: tuple[torch.Tensor, ...]
        if architecture == "mlp":
            self.model = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size))
            self.state = ()
        elif architecture == "rnn":
            self.model = nn.RNNCell(hidden_size, hidden_size, nonlinearity="tanh")
            self.state = (torch.zeros((1, hidden_size)),)
        elif architecture == "gru":
            self.model = nn.GRUCell(hidden_size, hidden_size)
            self.state = (torch.zeros((1, hidden_size)),)
        else:
            gate_disabled = architecture == "demian_route_ablation"
            self.model = DemianV1GateState(hidden_size, gate_disabled=gate_disabled, trace_routes=False)
            self.state = tuple(self.model.initial_state(1, torch.device("cpu")))
        self.action_head = nn.Linear(hidden_size, 4 + symbol_count)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters()) + sum(
            parameter.numel() for parameter in self.action_head.parameters()
        )

    @property
    def state_tensor_count(self) -> int:
        return len(self.state)

    @property
    def state_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.state)

    def advance(self, observation: WorldObservation, acknowledgement: ActionAcknowledgement | None) -> AdapterStep:
        encoded = self.encoder.encode(observation, acknowledgement)
        return self._advance_encoded(encoded)

    def advance_autonomous(self) -> AdapterStep:
        return self._advance_encoded(torch.zeros((1, self.hidden_size), dtype=torch.float32))

    def _advance_encoded(self, encoded: torch.Tensor) -> AdapterStep:
        if self.architecture == "mlp":
            surface = self.model(encoded)
        elif self.architecture in {"rnn", "gru"}:
            next_state = self.model(encoded, self.state[0])
            self.state = (next_state,)
            surface = next_state
        else:
            assert isinstance(self.model, DemianV1GateState)
            state: V1State = self.state  # type: ignore[assignment]
            state = self.encoder.inject(state, encoded)
            state = self.model.step(state)
            self.state = tuple(state)
            surface = self.model.state_vector(state)
        return AdapterStep(
            surface=surface.reshape(-1).detach().cpu().tolist(),
            proposal=self._decode(surface),
        )

    def capture_runtime(self) -> ModelRuntimeSnapshot:
        return ModelRuntimeSnapshot(
            state=tuple(value.detach().clone() for value in self.state),
            configuration_fingerprint=self._configuration_fingerprint(),
        )

    def restore_runtime(self, snapshot: ModelRuntimeSnapshot) -> None:
        if snapshot.configuration_fingerprint != self._configuration_fingerprint():
            raise ValueError("comparator configuration mismatch")
        self.state = tuple(value.detach().clone() for value in snapshot.state)

    def _decode(self, surface: torch.Tensor) -> ActionProposal:
        logits = self.action_head(surface).reshape(-1)
        operation = ("noop", "read", "write", "answer")[int(torch.argmax(logits[:4]).item())]
        if operation in {"write", "answer"}:
            operand = int(torch.argmax(logits[4:]).item()) % self.symbol_count
            return ActionProposal(operation=operation, operand=operand)  # type: ignore[arg-type]
        return ActionProposal(operation=operation)  # type: ignore[arg-type]

    def _configuration_fingerprint(self) -> str:
        configuration = {
            "architecture": self.architecture,
            "seed": self.seed,
            "hidden_size": self.hidden_size,
            "symbol_count": self.symbol_count,
            "encoder_scale": self.encoder.scale,
        }
        digest = sha256(json.dumps(configuration, sort_keys=True).encode("utf-8"))
        for name, value in list(self.model.state_dict().items()) + list(self.action_head.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(repr(value.detach().cpu().tolist()).encode("utf-8"))
        return digest.hexdigest()


class ComparatorRunner:
    """Common world ordering for a declared comparator adapter."""

    def __init__(
        self,
        *,
        architecture: Architecture,
        seed: int,
        hidden_size: int,
        cue_symbol: int,
        delay_steps: int,
    ) -> None:
        self.adapter = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
        self.world = CueDelayQueryWorld(symbol_count=3, cue_symbol=cue_symbol, delay_steps=delay_steps)
        self.connector = WorldConnector(symbol_count=3)

    def step(self) -> ComparatorRecord:
        if self.world.completed:
            raise RuntimeError("closed-loop episode is completed")
        acknowledgement = self.connector.advance(self.world)
        observation = self.world.observe()
        input_frame = InputFrame(observation=observation, acknowledgement=acknowledgement)
        step = self.adapter.advance(observation, acknowledgement)
        acceptance = self.connector.submit(step.proposal)
        self.world.advance_time()
        return ComparatorRecord(
            observation=observation,
            executed_acknowledgement=acknowledgement,
            input_frame=input_frame,
            proposal=step.proposal,
            acceptance=acceptance,
            surface=step.surface,
        )


def capture_comparator_runtime(runner: ComparatorRunner) -> ComparatorRuntimeSnapshot:
    return ComparatorRuntimeSnapshot(
        adapter=runner.adapter.capture_runtime(),
        world=capture_world_runtime(runner.world, runner.connector),
        configuration_fingerprint=_runner_fingerprint(runner),
    )


def restore_comparator_runtime(runner: ComparatorRunner, snapshot: ComparatorRuntimeSnapshot) -> None:
    if snapshot.configuration_fingerprint != _runner_fingerprint(runner):
        raise ValueError("comparator runner configuration mismatch")
    runner.adapter.restore_runtime(snapshot.adapter)
    restore_world_runtime(runner.world, runner.connector, snapshot.world)


def replay_comparator_frames(
    *, architecture: Architecture, seed: int, hidden_size: int, frames: list[InputFrame]
) -> list[tuple[list[float], ActionProposal]]:
    adapter = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
    return [
        (step.surface, step.proposal)
        for frame in frames
        for step in [adapter.advance(frame.observation, frame.acknowledgement)]
    ]


def _runner_fingerprint(runner: ComparatorRunner) -> str:
    payload = {
        "adapter": runner.adapter._configuration_fingerprint(),
        "world": (runner.world.symbol_count, runner.world.cue_symbol, runner.world.delay_steps),
        "connector": (runner.connector.symbol_count, runner.connector.storage_enabled, runner.connector.storage_read_only),
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def matched_perturbation_ticks(*, seed: int, steps: int, rate: float, salt: int) -> tuple[int, ...]:
    if steps < 1 or not 0 < rate <= 1:
        raise ValueError("steps and rate must be positive")
    count = max(1, round(steps * rate))
    return tuple(sorted(random.Random(f"{seed}:{salt}").sample(range(steps), count)))


def apply_relative_pulse(adapter: ClosedLoopModelAdapter, *, amplitude: float, seed: int, tick: int) -> None:
    if not adapter.state or amplitude == 0.0:
        return
    state_norm = sum(value.square().sum().item() for value in adapter.state) ** 0.5
    if state_norm == 0.0:
        return
    directions = []
    for index, value in enumerate(adapter.state):
        generator = torch.Generator(device=value.device).manual_seed(seed * 100_000 + tick * 10 + index)
        directions.append(torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype))
    direction_norm = sum(value.square().sum().item() for value in directions) ** 0.5
    adapter.state = tuple(
        state + direction * (state_norm * amplitude / direction_norm)
        for state, direction in zip(adapter.state, directions)
    )


def run_comparator_condition(
    *,
    architecture: Architecture,
    condition: Condition,
    seeds: Sequence[int],
    hidden_size: int,
    steps: int,
    delay_steps: int,
    environment_exposure: EnvironmentExposure = "natural",
    sample_every: int = 1,
) -> dict[str, object]:
    """Run one matched intervention condition with a declared comparator."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if steps < 1:
        raise ValueError("steps must be positive")
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    if environment_exposure not in {"natural", "forced_read"}:
        raise ValueError("unknown environment exposure mode")
    include_internal = condition in {"internal", "both"}
    include_environment = condition in {"environment", "both"}
    runs: list[dict[str, object]] = []
    for position, seed in enumerate(seeds):
        runner = ComparatorRunner(
            architecture=architecture,
            seed=seed,
            hidden_size=hidden_size,
            cue_symbol=seed % 3,
            delay_steps=delay_steps,
        )
        internal_ticks = set(
            matched_perturbation_ticks(seed=seed, steps=steps, rate=PERTURBATION_RATE, salt=11)
            if include_internal
            else ()
        )
        environment_ticks = set(
            matched_perturbation_ticks(seed=seed, steps=steps, rate=PERTURBATION_RATE, salt=17)
            if include_environment
            else ()
        )
        environment_mode = "clear" if position % 2 == 0 else "replace"
        events: list[dict[str, object]] = []
        trace: list[dict[str, object]] = []
        state_samples: list[dict[str, object]] = []
        submitted_storage_operations = 0
        executed_storage_operations = 0
        invalid_actions = 0
        answer_tick: int | None = None
        started = perf_counter()
        for step_index in range(steps):
            if runner.world.completed:
                break
            if step_index in internal_ticks:
                apply_relative_pulse(runner.adapter, amplitude=INTERNAL_AMPLITUDE, seed=seed, tick=step_index)
                events.append({"tick": step_index, "kind": "internal_pulse", "amplitude": INTERNAL_AMPLITUDE})
            if step_index in environment_ticks:
                disturbance = _apply_register_disturbance(runner, mode=environment_mode, seed=seed, tick=step_index)
                disturbance["observed"] = False
                disturbance["read_tick"] = None
                if environment_exposure == "forced_read":
                    disturbance["replaced_pending_action"] = (
                        asdict(runner.connector.pending) if runner.connector.pending is not None else None
                    )
                    runner.connector.pending = ActionProposal("read")
                events.append(disturbance)
            record = runner.step()
            for event in events:
                if (
                    event.get("tick") == step_index
                    and str(event.get("kind")).startswith("register_")
                    and record.executed_acknowledgement.executed_operation == "read"
                ):
                    event["observed"] = record.executed_acknowledgement.read_value == event["after"]
                    event["read_tick"] = record.observation.tick
            if record.proposal.operation in {"read", "write"}:
                submitted_storage_operations += 1
            if record.executed_acknowledgement.executed_operation in {"read", "write"}:
                executed_storage_operations += 1
            if not record.acceptance.accepted:
                invalid_actions += 1
            if record.executed_acknowledgement.answer_correct is not None:
                answer_tick = record.observation.tick
            trace.append(
                {
                    "tick": record.observation.tick,
                    "phase": record.observation.phase,
                    "surface_l2": math.sqrt(sum(value * value for value in record.surface)),
                    "state_l2": _state_l2(runner.adapter),
                    "state_component_l2": [
                        math.sqrt(value.square().sum().item()) for value in runner.adapter.state
                    ],
                    "proposal": asdict(record.proposal),
                    "acceptance": asdict(record.acceptance),
                    "executed_acknowledgement": asdict(record.executed_acknowledgement),
                }
            )
            if step_index % sample_every == 0 or step_index == steps - 1 or runner.world.completed:
                state_samples.append(
                    {
                        "tick": record.observation.tick,
                        "surface": record.surface,
                        "full_state": [value.detach().reshape(-1).cpu().tolist() for value in runner.adapter.state],
                    }
                )
        runs.append(
            {
                "seed": seed,
                "cue_symbol": seed % 3,
                "trace": trace,
                "state_samples": state_samples,
                "events": events,
                "executed_tick_count": len(trace),
                "final_register": runner.world.register,
                "final_score": runner.world.score,
                "answer_tick": answer_tick,
                "state_bytes": runner.adapter.state_bytes,
                "parameter_count": runner.adapter.parameter_count,
                "submitted_storage_operations": submitted_storage_operations,
                "executed_storage_operations": executed_storage_operations,
                "invalid_actions": invalid_actions,
                "wall_seconds": perf_counter() - started,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "closed-loop-comparator-condition",
        "architecture": architecture,
        "condition": condition,
        "config": {
            "hidden_size": hidden_size,
            "steps": steps,
            "delay_steps": delay_steps,
            "internal_amplitude": INTERNAL_AMPLITUDE,
            "perturbation_rate": PERTURBATION_RATE,
            "environment_exposure": environment_exposure,
            "sample_every": sample_every,
        },
        "runs": runs,
        "metrics": {
            "trajectory_count": len(runs),
            "scheduled_internal_events": sum(
                sum(event["kind"] == "internal_pulse" for event in run["events"]) for run in runs
            ),
            "scheduled_environment_events": sum(
                sum(str(event["kind"]).startswith("register_") for event in run["events"]) for run in runs
            ),
            "observed_environment_events": sum(
                sum(event.get("observed") is True for event in run["events"]) for run in runs
            ),
            "correct_answers": sum(run["final_score"] for run in runs),
            "answered": sum(run["answer_tick"] is not None for run in runs),
            "wrong_answers": sum(run["answer_tick"] is not None and run["final_score"] == 0 for run in runs),
            "no_answers": sum(run["answer_tick"] is None for run in runs),
            "submitted_storage_operations": sum(run["submitted_storage_operations"] for run in runs),
            "executed_storage_operations": sum(run["executed_storage_operations"] for run in runs),
            "invalid_actions": sum(run["invalid_actions"] for run in runs),
        },
    }


def _state_l2(adapter: ClosedLoopModelAdapter) -> float:
    return math.sqrt(sum(value.square().sum().item() for value in adapter.state))


def _apply_register_disturbance(
    runner: ComparatorRunner,
    *,
    mode: Literal["clear", "replace"],
    seed: int,
    tick: int,
) -> dict[str, object]:
    before = runner.world.register
    if mode == "clear":
        runner.world.reset_storage()
    else:
        candidates = [value for value in range(runner.world.symbol_count) if value != before]
        runner.world.register = random.Random(f"register:{seed}:{tick}").choice(candidates)
    return {"tick": tick, "kind": f"register_{mode}", "before": before, "after": runner.world.register}


def build_budget_tracks(*, reference_hidden_size: int, seed: int) -> dict[str, object]:
    """Select deterministic nearest-width specifications for separate budget tracks."""
    if reference_hidden_size < 1:
        raise ValueError("reference_hidden_size must be positive")
    reference = ClosedLoopModelAdapter(architecture="demian", seed=seed, hidden_size=reference_hidden_size)
    architectures: tuple[Architecture, ...] = ("mlp", "rnn", "gru", "demian_route_ablation", "demian")
    return {
        "state": _build_budget_track(
            architectures=architectures,
            target=reference.state_bytes,
            metric="state_bytes",
            reference_hidden_size=reference_hidden_size,
            seed=seed,
        ),
        "parameters": _build_budget_track(
            architectures=architectures,
            target=reference.parameter_count,
            metric="parameter_count",
            reference_hidden_size=reference_hidden_size,
            seed=seed,
        ),
    }


def _build_budget_track(
    *,
    architectures: tuple[Architecture, ...],
    target: int,
    metric: Literal["state_bytes", "parameter_count"],
    reference_hidden_size: int,
    seed: int,
) -> dict[str, object]:
    specifications: dict[str, dict[str, object]] = {}
    for architecture in architectures:
        if metric == "state_bytes" and architecture == "mlp":
            specifications[architecture] = {
                "hidden_size": reference_hidden_size,
                "eligible": False,
                "reason": "memoryless_control_has_zero_persistent_state",
                "matched_value": 0,
                "target_value": target,
            }
            continue
        selected_hidden_size, value = _nearest_budget_width(
            architecture=architecture,
            metric=metric,
            target=target,
            maximum_hidden_size=reference_hidden_size * 8,
            seed=seed,
        )
        specifications[architecture] = {
            "hidden_size": selected_hidden_size,
            "eligible": True,
            "matched_value": value,
            "target_value": target,
            "absolute_error": abs(value - target),
        }
    target_specification = specifications["demian"]
    return {"metric": metric, "target": target_specification, "specifications": specifications}


def _nearest_budget_width(
    *,
    architecture: Architecture,
    metric: Literal["state_bytes", "parameter_count"],
    target: int,
    maximum_hidden_size: int,
    seed: int,
) -> tuple[int, int]:
    """Find the nearest monotonic width without materializing every candidate model."""
    values: dict[int, int] = {}

    def value_at(hidden_size: int) -> int:
        if hidden_size not in values:
            adapter = ClosedLoopModelAdapter(architecture=architecture, seed=seed, hidden_size=hidden_size)
            values[hidden_size] = getattr(adapter, metric)
        return values[hidden_size]

    low, high = 1, maximum_hidden_size
    while low < high:
        middle = (low + high) // 2
        if value_at(middle) < target:
            low = middle + 1
        else:
            high = middle
    candidates = {max(1, low - 1), low}
    selected = min(
        candidates,
        key=lambda hidden_size: (abs(value_at(hidden_size) - target), hidden_size),
    )
    return selected, value_at(selected)
