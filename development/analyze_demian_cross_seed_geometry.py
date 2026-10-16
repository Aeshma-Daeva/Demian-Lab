"""Cross-seed relational geometry analysis of a hash-pinned history corpus."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def _distance_vector(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0, keepdims=True)
    distances = np.sqrt(np.maximum(0., np.sum((centered[:, None] - centered[None, :]) ** 2, axis=2)))
    return distances[np.triu_indices(len(points), 1)]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        raise ValueError("degenerate pairwise-distance geometry")
    return float(np.corrcoef(left, right)[0, 1])


def _input_history_geometry(rows: list[dict]) -> np.ndarray:
    """Ordered symbol histories: an encoder-only relational baseline."""
    return np.asarray([row["symbols"] for row in rows], dtype=np.float64)


def paired_geometry_score(left: np.ndarray, right: np.ndarray, *, shuffle_seed: int, shuffle_count: int) -> dict:
    """Compare same-history relational geometry with fixed shuffled pairings."""
    x, y = _distance_vector(left), _distance_vector(right)
    paired = _correlation(x, y)
    scale_x, scale_y = np.sqrt(np.mean(x ** 2)), np.sqrt(np.mean(y ** 2))
    discrepancy = float(np.sqrt(np.mean((x / scale_x - y / scale_y) ** 2)))
    rng = np.random.default_rng(shuffle_seed)
    shuffled = []
    for _ in range(shuffle_count):
        permutation = rng.permutation(len(right))
        shuffled.append(_correlation(x, _distance_vector(right[permutation])))
    return {"paired_distance_correlation": paired, "unit_rms_distance_discrepancy": discrepancy,
            "shuffled_distance_correlation_mean": float(np.mean(shuffled)),
            "paired_minus_shuffled": paired - float(np.mean(shuffled))}


def _ordered_group(states: np.ndarray, rows: list[dict], seed: int, checkpoint: str) -> np.ndarray:
    selection = [(tuple(row["symbols"]), states[index]) for index, row in enumerate(rows)
                 if row["model_seed"] == seed and row["order"] == "train" and row["checkpoint_id"] == checkpoint]
    if len(selection) != 64 or len({history for history, _ in selection}) != 64:
        raise ValueError(f"expected exactly 64 unique histories: seed={seed}, checkpoint={checkpoint}")
    selection.sort(key=lambda item: item[0])
    result = np.asarray([state for _, state in selection])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite source state")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    protocol = json.loads(args.protocol.read_text())
    source = Path(protocol["source_artifact"])
    states_path, rows_path = source / "states.npz", source / "state_metadata.jsonl"
    states = np.load(states_path)["states"]
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if len(rows) != len(states):
        raise ValueError("metadata/state row count mismatch")
    seeds = sorted({row["model_seed"] for row in rows if row["order"] == "train"})
    checkpoints = ["cue6", "delay2", "delay8", "delay32", "delay64"]
    metrics = []
    for checkpoint in checkpoints:
        groups = {seed: _ordered_group(states, rows, seed, checkpoint) for seed in seeds}
        group_rows = {seed: sorted([row for row in rows if row["model_seed"] == seed and row["order"] == "train" and row["checkpoint_id"] == checkpoint],
                                  key=lambda row: tuple(row["symbols"])) for seed in seeds}
        for left_seed, right_seed in itertools.combinations(seeds, 2):
            metric = paired_geometry_score(groups[left_seed], groups[right_seed],
                                           shuffle_seed=left_seed * 100000 + right_seed * 10 + len(checkpoint),
                                           shuffle_count=32)
            left, right = groups[left_seed], groups[right_seed]
            raw_cosine = np.sum(left * right, axis=1) / np.linalg.norm(left, axis=1) / np.linalg.norm(right, axis=1)
            input_geometry = _distance_vector(_input_history_geometry(group_rows[left_seed]))
            metrics.append({"checkpoint": checkpoint, "left_seed": left_seed, "right_seed": right_seed,
                            "left_input_distance_correlation": _correlation(_distance_vector(left), input_geometry),
                            "right_input_distance_correlation": _correlation(_distance_vector(right), input_geometry),
                            "same_history_raw_cosine_mean": float(np.mean(raw_cosine)), **metric})
    args.out.mkdir(parents=True)
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps({
        "source_hashes": {"states.npz": _sha256(states_path), "state_metadata.jsonl": _sha256(rows_path)},
        "source_seeds": seeds, "metrics": metrics,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
