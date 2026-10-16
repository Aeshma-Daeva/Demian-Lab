"""Descriptive channelwise availability audit for a frozen full-state artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from development.demian_v1_gate_state import V1_CHANNELS
from development.run_demian_frozen_readout_delayed_cue import basis
from development.run_demian_uncertainty_localization import fit_probe, predict_probe


def channel_slices(width: int) -> dict[str, tuple[int, int]]:
    """State-vector slices follow ``DemianV1GateState.initial_state`` exactly."""
    sizes = (width, width, width // 8, width, width, width)
    edges = np.cumsum((0, *sizes))
    return {channel: (int(left), int(right))
            for channel, left, right in zip(V1_CHANNELS, edges[:-1], edges[1:], strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    protocol = json.loads((args.artifact / "protocol.json").read_text())
    states = np.load(args.artifact / "states.npz")["states"]
    rows = [json.loads(line) for line in (args.artifact / "rows.jsonl").read_text().splitlines()]
    width, train_ids = int(protocol["hidden_size"]), set(protocol["train_history_ids"])
    metrics = []
    for channel, (left, right) in channel_slices(width).items():
        features = states[:, left:right]
        projection = basis(features.shape[1], 1000 + left)
        for seed in protocol["model_seeds"]:
            seed_rows = [i for i, row in enumerate(rows) if row["seed"] == seed]
            train = [i for i in seed_rows if rows[i]["id"] in train_ids]
            fit = fit_probe(features[train] @ projection, np.eye(3)[[rows[i]["cue"] for i in train]], alpha=1.0)
            for delay in protocol["delays"]:
                evaluation = [i for i in seed_rows if rows[i]["delay"] == delay and rows[i]["id"] not in train_ids]
                target = np.asarray([rows[i]["cue"] for i in evaluation])
                accuracy = float((np.argmax(predict_probe(fit, features[evaluation] @ projection), axis=1) == target).mean())
                metrics.append({"channel": channel, "seed": seed, "delay": delay, "accuracy": accuracy})
    args.out.mkdir(parents=True)
    (args.out / "summary.json").write_text(json.dumps({
        "source": str(args.artifact), "analysis": "descriptive channelwise external decoding; not causal ownership",
        "metrics": metrics,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
