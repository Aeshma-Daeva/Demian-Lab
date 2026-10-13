"""Two-axis heldout geometry against a fixed two-delay local Jacobian."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from development.demian_uncertainty_probe import encode_evidence
from development.demian_uncertainty_world import EvidenceObservation
from development.demian_v1_gate_state import DemianV1GateState
from development.run_demian_uncertainty_similarity import (
    _basis, _excluded_fraction, collect_clouds, fit_similarity,
    predict_similarity, residual_metrics, write_artifacts,
)


def split_history_indices(*, history_length: int, seed: int) -> tuple[list[int], list[int]]:
    if history_length < 2:
        raise ValueError('each terminal-symbol stratum requires multiple histories')
    rng = np.random.default_rng(seed)
    train, heldout = [], []
    for symbol in (0, 1):
        stratum = np.arange(symbol, 2 ** history_length, 2)
        rng.shuffle(stratum)
        train.extend(int(value) for value in stratum[:len(stratum)//2])
        heldout.extend(int(value) for value in stratum[len(stratum)//2:])
    return sorted(train), sorted(heldout)


def make_transition(*, seed: int, hidden_size: int, step_index: int, dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    torch.manual_seed(seed)
    model = DemianV1GateState(hidden_size, trace_routes=False)
    initial = model.initial_state(1, torch.device('cpu'))
    widths = [part.numel() for part in initial]
    model.to(dtype=dtype)
    model.requires_grad_(False)
    delay = encode_evidence(EvidenceObservation(step_index, 'delay', None, None), hidden_size).to(dtype=dtype)

    def transition(vector: torch.Tensor) -> torch.Tensor:
        # Autodiff can reevaluate this closure: restore runtime on every call.
        model._step_index = step_index
        model._frozen_gate = None
        model._route_trace = None
        state = tuple(part.reshape(1, -1) for part in torch.split(vector, widths))
        for _ in range(2):
            state = (state[0] + delay, *state[1:])
            state = model.step(state)
        return torch.cat([part.reshape(-1) for part in state])

    return transition


def linearized_predictions(transition: Callable, center: np.ndarray, points: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    reference = torch.as_tensor(center, dtype=torch.float64)
    directions = points - center
    outside = np.zeros_like(directions)
    if basis is not None:
        projected = directions @ basis @ basis.T
        outside = directions - projected
        directions = projected
    predictions = []
    for direction, perpendicular in zip(directions, outside, strict=True):
        value, derivative = torch.autograd.functional.jvp(
            transition, reference, torch.as_tensor(direction, dtype=torch.float64), strict=True,
        )
        predictions.append((value + derivative).detach().numpy() + perpendicular)
    return np.vstack(predictions)


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    train_ids, heldout_ids = split_history_indices(history_length=len(protocol['train_reliability_schedule']), seed=int(protocol['split_seed']))
    if protocol.get('variants') != ['base', 'extra_delay']:
        raise ValueError('this gate permits only base and extra-delay variants')
    clouds, metadata, states = collect_clouds(protocol)
    metrics, transforms, predictions = [], {}, {}
    exact_matches = []
    double_errors = []
    step_index = len(protocol['train_reliability_schedule']) + int(protocol['base_delay_steps'])
    for raw_seed in protocol['model_seeds']:
        seed = int(raw_seed)
        train_a = clouds[(seed, 'train', 'base')][train_ids]
        train_b = clouds[(seed, 'train', 'extra_delay')][train_ids]
        basis = _basis([train_a, train_b], int(protocol['subspace_rank']))
        fit = fit_similarity(train_a, train_b, basis)
        permutation = np.random.default_rng(seed + int(protocol['permutation_seed'])).permutation(len(train_b))
        shuffled = fit_similarity(train_a, train_b[permutation], basis)
        float_transition = make_transition(seed=seed, hidden_size=int(protocol['hidden_size']), step_index=step_index, dtype=torch.float32)
        double_transition = make_transition(seed=seed, hidden_size=int(protocol['hidden_size']), step_index=step_index, dtype=torch.float64)
        transforms[f'{seed}_basis'] = basis
        for name in ('source_mean', 'target_mean', 'rotation', 'scale'):
            transforms[f'{seed}_{name}'] = np.asarray(fit[name])
        transforms[f'{seed}_train_history_indices'] = np.asarray(train_ids)
        transforms[f'{seed}_heldout_history_indices'] = np.asarray(heldout_ids)
        for family in ('train', 'heldout'):
            a, b = clouds[(seed, family, 'base')], clouds[(seed, family, 'extra_delay')]
            with torch.no_grad():
                for left, right in zip(a, b, strict=True):
                    rolled = float_transition(torch.as_tensor(left, dtype=torch.float32)).numpy()
                    exact_matches.append(bool(np.array_equal(rolled, right)))
                    double_errors.append(float(np.linalg.norm(double_transition(torch.as_tensor(left, dtype=torch.float64)).numpy() - right)))
        if not all(exact_matches):
            raise RuntimeError('independent B checkpoint differs from exact A continuation')
        for axis, family, indices in (
            ('same_order_histories', 'train', heldout_ids),
            ('unseen_order', 'heldout', list(range(len(clouds[(seed, 'heldout', 'base')])))),
        ):
            a = clouds[(seed, family, 'base')][indices]
            b = clouds[(seed, family, 'extra_delay')][indices]
            maps = {
                'identity': a,
                'translation_only': a + train_b.mean(axis=0) - train_a.mean(axis=0),
                'similarity': predict_similarity(a, fit),
                'randomized_pair_similarity': predict_similarity(a, shuffled),
                'jacobian_rank8': linearized_predictions(double_transition, train_a.mean(axis=0), a, basis),
                'jacobian_full': linearized_predictions(double_transition, train_a.mean(axis=0), a),
            }
            for name, predicted in maps.items():
                predictions[f'{seed}_{axis}_{name}'] = predicted
                metrics.append({
                    'model_seed': seed, 'axis': axis, 'map': name,
                    'eval_source_excluded_fraction': _excluded_fraction(a, basis),
                    'eval_target_excluded_fraction': _excluded_fraction(b, basis),
                    **residual_metrics(a, b, predicted, basis),
                })
    return {
        'source_commit': source_commit, 'state_count': len(states), 'states': states,
        'state_metadata': metadata, 'transforms': transforms, 'predictions': predictions,
        'metrics': metrics,
        'integrity': {
            'all_finite': bool(np.isfinite(states).all() and all(np.isfinite(value) for row in metrics for value in row.values() if isinstance(value, float))),
            'exact_float32_continuation': all(exact_matches), 'exact_continuation_count': len(exact_matches),
            'double_rollout_max_norm_error': max(double_errors),
            'scope': 'train-history-only fitted maps and fixed training-centroid JVP; two heldout axes; no labels',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)
    print(json.dumps({'out': str(args.out), 'states': payload['state_count']}))


if __name__ == '__main__':
    main()
