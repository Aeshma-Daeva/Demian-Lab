"""Validated protocol for the deterministic closed-loop Demian boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClosedLoopProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(ge=1)
    protocol_id: str
    demian_state_channels: list[str]
    connector_is_external: bool
    update_order: list[str]
    checkpoint_owners: list[str]
    action_operations: list[Literal["noop", "read", "write", "answer"]]
    storage_capacity: int = Field(ge=1)
    action_latency_ticks: int = Field(ge=1)
    controls: list[str]
    score_visibility: Literal["episode_end_only"]

    @model_validator(mode="after")
    def require_closed_loop_boundary(self) -> ClosedLoopProtocol:
        if self.demian_state_channels != ["fast", "slow", "control", "message", "carrier", "gate"]:
            raise ValueError("protocol must name the implemented six Demian channels")
        if not self.connector_is_external:
            raise ValueError("connector must remain outside Demian state")
        if len(self.action_operations) != len(set(self.action_operations)):
            raise ValueError("action operations must be unique")
        return self


def load_closed_loop_protocol(path: Path) -> ClosedLoopProtocol:
    return ClosedLoopProtocol.model_validate(json.loads(path.read_text(encoding="utf-8")))
