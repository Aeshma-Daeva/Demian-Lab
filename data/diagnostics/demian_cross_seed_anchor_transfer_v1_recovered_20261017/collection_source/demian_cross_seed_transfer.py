"""Frozen cross-seed diagnostics. Every seed calibration uses anchors only."""
from __future__ import annotations

import numpy as np

from development.run_demian_uncertainty_localization import fit_probe, predict_probe


HISTORIES = np.array([list(map(int, f"{i:06b}")) for i in range(64)], dtype=np.int64)
BITS = 2.0 * HISTORIES - 1.0


def make_split(seed):
    pairs = np.random.default_rng(seed).permutation(32)
    parts = {}
    for name, chosen in zip(("anchors", "fit", "evaluation"), (pairs[:8], pairs[8:20], pairs[20:]), strict=True):
        parts[name] = sorted([int(i) for pair in chosen for i in (pair, 63-pair)])
    design = np.c_[BITS[parts["anchors"]], np.ones(16)]
    if np.linalg.matrix_rank(design) != 7:
        raise ValueError("anchor design does not span all six cue positions")
    return parts


def anchor_features(states, anchors):
    x = np.asarray(states, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite state")
    a = x[anchors]
    center = a.mean(axis=0)
    scatter = float(np.sqrt(np.mean(np.sum((a-center)**2, axis=1))))
    norm = float(np.sqrt(np.mean(np.sum(a*a, axis=1))))
    floor = 10*np.finfo(np.float32).eps*max(norm, 1.0)
    limited = scatter <= floor
    features = (np.zeros((len(x), len(a))) if limited else
                np.sqrt(np.maximum(0, np.sum((x[:, None]-a[None, :])**2, axis=2)))/scatter)
    return features, {"center": center, "scatter": scatter, "precision_floor": floor,
                      "precision_limited": bool(limited)}


def feature_maps(states, split, projection):
    anchor, calibration = anchor_features(states, split["anchors"])
    x = np.asarray(states, dtype=np.float64)
    raw = np.zeros((64, projection.shape[1])) if calibration["precision_limited"] else (
        (x-calibration["center"])@projection/calibration["scatter"])
    design = np.c_[BITS, np.ones(64)]
    coeff = np.linalg.lstsq(design[split["anchors"]], x[split["anchors"]], rcond=None)[0]
    linear_states = design@coeff
    surrogate, _ = anchor_features(linear_states, split["anchors"])
    return {"anchor": anchor, "raw": raw, "linear": surrogate}, {**calibration, "linear_coefficients": coeff}


def fit_readouts(training_states, targets, split, *, projection_seed, shuffle_seed, alpha=1):
    data = np.asarray(training_states, dtype=np.float64)
    projection = np.linalg.qr(np.random.default_rng(projection_seed).normal(size=(data.shape[-1], 16)))[0]
    maps = [feature_maps(x, split, projection)[0] for x in data]
    fit_ids = split["fit"]
    y = np.asarray(targets)[fit_ids]
    frozen = {"projection": projection}
    for feature in ("anchor", "raw", "linear"):
        x = np.concatenate([m[feature][fit_ids] for m in maps])
        fit = fit_probe(x, np.tile(y, len(data)), alpha=alpha)
        for key, value in fit.items():
            frozen[f"{feature}_{key}"] = value
    shuffled = y.copy()
    rng = np.random.default_rng(shuffle_seed)
    for symbol in (0, 1):
        pos = np.flatnonzero(HISTORIES[fit_ids, -1] == symbol)
        shuffled[pos] = rng.permutation(shuffled[pos])
    fit = fit_probe(np.concatenate([m["anchor"][fit_ids] for m in maps]), np.tile(shuffled, len(data)), alpha=alpha)
    for key, value in fit.items():
        frozen[f"shuffle_{key}"] = value
    return frozen, {"shuffled_training_labels": shuffled,
                    "changed_labels": int(np.sum(np.abs(shuffled-y) > 1e-12))}


def predict_readouts(frozen, states, split):
    maps, _ = feature_maps(states, split, frozen["projection"])
    predictions = {}
    for name in ("anchor", "raw", "linear", "shuffle"):
        fit = {key: frozen[f"{name}_{key}"] for key in ("mean", "scale", "weights")}
        predictions[name] = predict_probe(fit, maps["anchor" if name == "shuffle" else name])
    return predictions


def r2(target, predicted):
    y, p = np.asarray(target), np.asarray(predicted)
    total = np.sum((y-y.mean())**2)
    if total <= 0 or not np.isfinite(p).all():
        raise ValueError("undefined R2")
    return float(1-np.sum((y-p)**2)/total)


def distance_matrix(x):
    z = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.maximum(0, np.sum((z[:, None]-z[None, :])**2, axis=2)))


def geometry_pair(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x, y = x-x.mean(0), y-y.mean(0)
    dx, dy = distance_matrix(x), distance_matrix(y)
    upper = np.triu_indices(len(x), 1)
    vx, vy = dx[upper], dy[upper]
    gx, gy = x@x.T, y@y.T
    denom = np.linalg.norm(gx)*np.linalg.norm(gy)
    if denom == 0 or vx.std() == 0 or vy.std() == 0:
        return {"distance_r": None, "cka": None, "knn5": None, "discrepancy": None}
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    nx, ny = np.argsort(dx, axis=1)[:, :5], np.argsort(dy, axis=1)[:, :5]
    return {"distance_r": float(np.corrcoef(vx, vy)[0, 1]),
            "cka": float(np.sum(gx*gy)/denom),
            "knn5": float(np.mean([len(set(a)&set(b))/5 for a, b in zip(nx, ny, strict=True)])),
            "discrepancy": float(np.sqrt(np.mean((vx/np.sqrt(np.mean(vx*vx))-vy/np.sqrt(np.mean(vy*vy)))**2)))}


def precision_valid(x, original):
    x, original = np.asarray(x, dtype=np.float64), np.asarray(original, dtype=np.float64)
    scatter = np.sqrt(np.mean(np.sum((x-x.mean(0))**2, axis=1)))
    floor = 10*np.finfo(np.float32).eps*max(1., np.sqrt(np.mean(np.sum(original*original, axis=1))))
    return bool(scatter > floor)


def guarded_geometry(x, y, original_x=None, original_y=None):
    valid = precision_valid(x, x if original_x is None else original_x) and precision_valid(y, y if original_y is None else original_y)
    measured = geometry_pair(x, y) if valid else dict.fromkeys(("distance_r", "cka", "knn5", "discrepancy"))
    return {**measured, "precision_limited": not valid}


def shape_diagnostics(x, split):
    x = np.asarray(x, dtype=np.float64)
    centered = x-x.mean(0)
    eigen = np.maximum(0, np.linalg.eigvalsh(centered@centered.T))
    total = eigen.sum()
    train = split["fit"]
    evaluate = split["evaluation"]
    design = np.c_[BITS, np.ones(64)]
    beta = np.linalg.lstsq(design[train], x[train], rcond=None)[0]
    residual = x-design@beta
    denom = np.sum((x[evaluate]-x[train].mean(0))**2)
    valid = precision_valid(x, x)
    return {"precision_limited": not valid,
            "residual_precision_limited": not precision_valid(residual[evaluate], x[evaluate]),
            "pc1_fraction": float(eigen[-1]/total) if valid else None,
            "effective_rank": float(total**2/np.sum(eigen**2)) if valid else None,
            "heldout_bit_linear_variance_explained": float(1-np.sum(residual[evaluate]**2)/denom) if valid else None}, residual
