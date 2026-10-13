"""Bounded discovery harness for public ambiguous-evidence Demian traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from development.demian_uncertainty_probe import run_uncertainty_episode
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, reference_belief


_CONDITIONS = {
    "balanced_informative": [(1, 0.75), (0, 0.75)],
    "consistent_informative": [(1, 0.75), (1, 0.75)],
    "noise_only": [(1, 0.5), (0, 0.5)],
    "order_permuted": [(0, 0.75), (1, 0.75)],
}


def _validate(protocol: dict) -> None:
    required = {"protocol_id", "hidden_size", "model_seeds", "episode_seeds", "delay_steps", "post_resolution_steps"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"protocol missing fields: {sorted(missing)}")
    if protocol["hidden_size"] < 10 or not protocol["model_seeds"] or not protocol["episode_seeds"]:
        raise ValueError("positive width and nonempty seed lists required")
    if len(set(protocol["model_seeds"])) != len(protocol["model_seeds"]):
        raise ValueError("model seeds must be unique")
    if any(delay < 0 for delay in protocol["delay_steps"]):
        raise ValueError("delays must be nonnegative")


def run_protocol(protocol: dict) -> dict[str, object]:
    _validate(protocol)
    cells = []
    for seed in protocol["model_seeds"]:
        for episode_seed in protocol["episode_seeds"]:
            target = episode_seed % 2
            for delay in protocol["delay_steps"]:
                for condition, evidence in _CONDITIONS.items():
                    world = AmbiguousEvidenceWorld(hidden_target=target, evidence=evidence, delay_steps=delay)
                    frames = world.frames()
                    trace = run_uncertainty_episode(model_seed=seed, hidden_size=protocol["hidden_size"], public_frames=frames)
                    checkpoints = trace.pop("checkpoints")
                    checkpoint_hashes = {
                        name: hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in snapshot.state)).hexdigest()
                        for name, snapshot in checkpoints.items()
                    }
                    belief = reference_belief(frames[:-1])
                    finite = all(math.isfinite(item["state_norm"]) for item in trace["measurements"])
                    cells.append({
                        "model_seed": seed, "episode_seed": episode_seed, "delay_steps": delay,
                        "condition": condition, "sampling": "scripted", "public_frames": [frame.__dict__ for frame in frames],
                        "reference_belief_pre_complete": belief, "trace": trace, "checkpoint_hashes": checkpoint_hashes, "finite": finite,
                    })
    return {"cells": cells, "integrity": {"all_replay_exact": all(cell["trace"]["replay_exact"] for cell in cells), "all_finite": all(cell["finite"] for cell in cells)}}


def write_artifacts(out: Path, protocol: dict, payload: dict[str, object]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(protocol, sort_keys=True).encode()
    (out / "manifest.json").write_text(json.dumps({"protocol": protocol, "protocol_sha256": hashlib.sha256(encoded).hexdigest(), "cell_count": len(payload["cells"])}, indent=2, sort_keys=True) + "\n")
    with (out / "cells.jsonl").open("x") as stream:
        for cell in payload["cells"]:
            stream.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
    (out / "summary.json").write_text(json.dumps({"schema_version": 1, "experiment": "demian-uncertainty-discovery", "claim_boundary": "harness and discovery trace only; not uncertainty representation, routing utility, learning, or adaptation", "cell_count": len(payload["cells"]), "integrity": payload["integrity"]}, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)
    print(json.dumps({"out": str(args.out), "cells": len(payload["cells"])}))


if __name__ == "__main__":
    main()
