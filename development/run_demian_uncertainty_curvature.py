"""Directional second-order Taylor check on cached Demian horizon cases."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from development.run_demian_uncertainty_horizon import verify_source, _reference_coverage
from development.run_demian_uncertainty_jacobian import make_transition
from development.run_demian_uncertainty_similarity import _excluded_fraction, residual_metrics, write_artifacts


def quadratic_terms(transition: Callable, center: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = torch.as_tensor(center, dtype=torch.float64)
    tangent = torch.as_tensor(direction, dtype=torch.float64)

    def first_derivative(at):
        return torch.autograd.functional.jvp(transition, at, tangent, create_graph=True, strict=False)[1]

    first, second = torch.autograd.functional.jvp(first_derivative, reference, tangent, strict=False)
    value = transition(reference)
    return tuple(part.detach().numpy() for part in (value, first, second))


def second_order_predictions(transition: Callable, center: np.ndarray, points: np.ndarray) -> np.ndarray:
    results = []
    for point in points:
        value, first, second = quadratic_terms(transition, center, point - center)
        results.append(value + first + .5 * second)
    return np.vstack(results)


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    root = Path(protocol['source_artifact_directory'])
    base_root = Path(protocol['base_artifact_directory'])
    verify_source(root)
    base_manifest = verify_source(base_root)
    source_protocol = base_manifest['protocol']
    future = np.load(root / 'states.npz')['states'].astype(np.float64)
    future_metadata = [json.loads(line) for line in (root / 'state_metadata.jsonl').read_text().splitlines()]
    prior = np.load(root / 'predictions.npz')
    base = np.load(base_root / 'states.npz')['states'].astype(np.float64)
    base_metadata = [json.loads(line) for line in (base_root / 'state_metadata.jsonl').read_text().splitlines()]
    saved = np.load(base_root / 'transforms.npz')
    metrics, predictions, transforms = [], {}, {}
    derivative_checks = []
    step_index = len(source_protocol['train_reliability_schedule']) + int(source_protocol['base_delay_steps'])
    for seed in source_protocol['model_seeds']:
        seed = int(seed)
        def source_cloud(family):
            return base[[i for i, row in enumerate(base_metadata) if row['model_seed'] == seed and row['family'] == family and row['variant'] == 'base']]
        training = source_cloud('train')[saved[f'{seed}_train_history_indices']]
        center = training.mean(axis=0)
        basis = saved[f'{seed}_basis']
        transforms[f'{seed}_basis'] = basis
        transforms[f'{seed}_source_mean'] = center
        for horizon in protocol['horizons']:
            horizon = int(horizon)
            transition = make_transition(seed=seed, hidden_size=int(source_protocol['hidden_size']), step_index=step_index, dtype=torch.float64, steps=horizon)
            check_direction = source_cloud('train')[saved[f'{seed}_heldout_history_indices'][0]] - center
            _, _, second = quadratic_terms(transition, center, check_direction)
            epsilon = float(protocol['finite_difference_epsilon'])
            with torch.no_grad():
                plus = transition(torch.as_tensor(center + epsilon * check_direction)).numpy()
                zero = transition(torch.as_tensor(center)).numpy()
                minus = transition(torch.as_tensor(center - epsilon * check_direction)).numpy()
            finite_difference = (plus - 2. * zero + minus) / epsilon ** 2
            derivative_checks.append(float(np.linalg.norm(second - finite_difference) / max(np.linalg.norm(second), 1e-12)))
            for axis, family, indices in (
                ('same_order_histories', 'train', saved[f'{seed}_heldout_history_indices']),
                ('unseen_order', 'heldout', np.arange(len(source_cloud('heldout')))),
            ):
                a = source_cloud(family)[indices]
                target_ids = [i for i, row in enumerate(future_metadata) if row['model_seed'] == seed and row['axis'] == axis and row['horizon'] == horizon]
                if [future_metadata[i]['history_index'] for i in target_ids] != [int(index) for index in indices]:
                    raise ValueError('cached target history pairing mismatch')
                b = future[target_ids]
                first = prior[f'{seed}_{horizon}_{axis}_jacobian_full']
                second = second_order_predictions(transition, center, a)
                predictions[f'{seed}_{horizon}_{axis}_taylor_second'] = second
                first_error = residual_metrics(a, b, first)['residual_relative_to_scatter']
                for name, predicted in (('jacobian_full', first), ('taylor_second', second)):
                    measures = residual_metrics(a, b, predicted, basis)
                    metrics.append({
                        'model_seed': seed, 'axis': axis, 'horizon': horizon, 'map': name,
                        'first_order_relative_residual': first_error,
                        'relative_residual_reduction': 1. - measures['residual_relative_to_scatter'] / max(first_error, 1e-20),
                        'eval_source_excluded_fraction': _excluded_fraction(a, basis),
                        'eval_target_excluded_fraction': _excluded_fraction(b, basis),
                        **_reference_coverage(a, center, basis), **measures,
                    })
    if max(derivative_checks) > float(protocol['finite_difference_tolerance']):
        raise RuntimeError('directional Hessian finite-difference check failed')
    selected = [index for index, row in enumerate(future_metadata) if row['horizon'] in protocol['horizons']]
    return {
        'source_commit': source_commit, 'state_count': len(selected), 'states': future[selected],
        'state_metadata': [future_metadata[index] for index in selected], 'transforms': transforms,
        'predictions': predictions, 'metrics': metrics,
        'integrity': {
            'all_finite': bool(all(np.isfinite(value) for row in metrics for value in row.values() if isinstance(value, float))),
            'source_hashes_verified': True, 'directional_hessian_fd_max_relative_error': max(derivative_checks),
            'scope': 'fixed-centroid second-order model-check on reused cases; cached nonlinear targets unchanged',
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
    print(json.dumps({'out': str(args.out), 'metrics': len(payload['metrics'])}))


if __name__ == '__main__':
    main()
