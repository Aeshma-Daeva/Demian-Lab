"""Frozen-probe utilities for the uncertainty representation gate."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def fixed_random_projection(features: np.ndarray, *, output_dim: int, seed: int) -> np.ndarray:
    """Project features through a deterministic, label-independent orthonormal map."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("projection features must be a finite two-dimensional array")
    if not 0 < output_dim <= matrix.shape[1]:
        raise ValueError("projection output_dim must be positive and no wider than the features")
    generator = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(generator.normal(size=(matrix.shape[1], output_dim)))
    return matrix @ basis[:, :output_dim]


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
    eval_standard_deviation = float(np.std(eval_target))
    if eval_standard_deviation <= 1e-12 or float(np.std(prediction)) <= 1e-12:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(eval_target, prediction)[0, 1])
    mae = float(np.mean(np.abs(eval_target - prediction)))
    return {
        "r2": _r2(eval_target, prediction),
        "constant_r2": _r2(eval_target, constant_prediction),
        "mae": mae,
        "pearson_r": pearson_r,
        "normalized_mae": mae / eval_standard_deviation if eval_standard_deviation > 1e-12 else 0.0,
    }
