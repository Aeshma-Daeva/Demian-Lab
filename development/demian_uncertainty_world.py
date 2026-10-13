"""Public ambiguous-evidence environment and evaluator oracle for Demian."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence


Phase = Literal["evidence", "delay", "resolve", "complete"]
_RELIABILITIES = {0.5, 0.75, 0.95}


@dataclass(frozen=True)
class EvidenceObservation:
    tick: int
    phase: Phase
    symbol: int | None
    reliability: float | None


@dataclass(frozen=True)
class WorldSnapshot:
    index: int


def _validate_event(symbol: int, reliability: float) -> None:
    if symbol not in {0, 1}:
        raise ValueError("evidence symbol must be 0 or 1")
    if reliability not in _RELIABILITIES:
        raise ValueError("reliability must be one of 0.5, 0.75, 0.95")


def reference_belief(public_prefix: Sequence[EvidenceObservation]) -> dict[str, float]:
    """Bayesian reference computed solely from public evidence observations."""
    log_odds = 0.0
    for observation in public_prefix:
        if observation.phase not in {"evidence", "resolve"} or observation.symbol is None:
            continue
        assert observation.reliability is not None
        _validate_event(observation.symbol, observation.reliability)
        if observation.reliability != 0.5:
            log_odds += (2 * observation.symbol - 1) * math.log(
                observation.reliability / (1 - observation.reliability)
            )
    probability = 1 / (1 + math.exp(-log_odds))
    entropy = 0.0 if probability in {0.0, 1.0} else -probability * math.log2(probability) - (1 - probability) * math.log2(1 - probability)
    return {"probability": probability, "log_odds": log_odds, "entropy": entropy}


class AmbiguousEvidenceWorld:
    """Scripted public sequence; hidden truth is retained only for evaluation."""

    def __init__(self, *, hidden_target: int, evidence: Sequence[tuple[int, float]], delay_steps: int) -> None:
        if hidden_target not in {0, 1}:
            raise ValueError("hidden target must be 0 or 1")
        if delay_steps < 0:
            raise ValueError("delay_steps must be nonnegative")
        for event in evidence:
            _validate_event(*event)
        self.hidden_target = hidden_target
        self._frames = tuple(
            [EvidenceObservation(tick, "evidence", symbol, reliability) for tick, (symbol, reliability) in enumerate(evidence)]
            + [EvidenceObservation(len(evidence) + tick, "delay", None, None) for tick in range(delay_steps)]
            + [EvidenceObservation(len(evidence) + delay_steps, "resolve", hidden_target, 0.95)]
            + [EvidenceObservation(len(evidence) + delay_steps + 1, "complete", None, None)]
        )
        self._index = 0

    @property
    def completed(self) -> bool:
        return self._index >= len(self._frames)

    def observe(self) -> EvidenceObservation:
        if self.completed:
            raise RuntimeError("episode is completed")
        return self._frames[self._index]

    def step(self) -> EvidenceObservation:
        observation = self.observe()
        self._index += 1
        return observation

    def frames(self) -> list[EvidenceObservation]:
        return list(self._frames)

    def capture(self) -> WorldSnapshot:
        return WorldSnapshot(self._index)

    def restore(self, snapshot: WorldSnapshot) -> None:
        if not 0 <= snapshot.index <= len(self._frames):
            raise ValueError("world snapshot index is invalid")
        self._index = snapshot.index
