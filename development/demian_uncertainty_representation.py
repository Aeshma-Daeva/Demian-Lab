"""Frozen-probe utilities for the uncertainty representation gate."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def split_episode_rows(rows: Sequence[dict[str, Any]], *, heldout_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete public episodes, never individual timepoint rows."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must lie strictly between zero and one")
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    if len(episode_ids) < 2:
        raise ValueError("at least two complete episodes are required")
    heldout_count = max(1, round(len(episode_ids) * heldout_fraction))
    heldout_ids = set(episode_ids[-heldout_count:])
    train = [dict(row) for row in rows if int(row["episode_id"]) not in heldout_ids]
    heldout = [dict(row) for row in rows if int(row["episode_id"]) in heldout_ids]
    return train, heldout


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    centered = y_true - np.mean(y_true)
    total = float(centered @ centered)
    if total <= 1e-12:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / total


def evaluate_ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> dict[str, float]:
    """Fit a standardized ridge probe and score a distinct episode split."""
    train = np.asarray(x_train, dtype=np.float64)
    evaluate = np.asarray(x_eval, dtype=np.float64)
    train_target = np.asarray(y_train, dtype=np.float64).reshape(-1)
    eval_target = np.asarray(y_eval, dtype=np.float64).reshape(-1)
    if train.ndim != 2 or evaluate.ndim != 2 or train.shape[1] != evaluate.shape[1]:
        raise ValueError("probe features must be two-dimensional with matching widths")
    if len(train) != len(train_target) or len(evaluate) != len(eval_target):
        raise ValueError("probe feature and target counts must match")
    if alpha < 0 or not all(np.isfinite(value).all() for value in (train, evaluate, train_target, eval_target)):
        raise ValueError("probe features and targets must be finite with nonnegative alpha")
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    normalized_train = (train - mean) / np.where(scale > 1e-12, scale, 1.0)
    normalized_eval = (evaluate - mean) / np.where(scale > 1e-12, scale, 1.0)
    augmented_train = np.column_stack((normalized_train, np.ones(len(normalized_train))))
    augmented_eval = np.column_stack((normalized_eval, np.ones(len(normalized_eval))))
    regularizer = alpha * np.eye(augmented_train.shape[1])
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(augmented_train.T @ augmented_train + regularizer, augmented_train.T @ train_target)
    prediction = augmented_eval @ weights
    constant_prediction = np.full_like(eval_target, train_target.mean())
    return {
        "r2": _r2(eval_target, prediction),
        "constant_r2": _r2(eval_target, constant_prediction),
        "mae": float(np.mean(np.abs(eval_target - prediction))),
    }
