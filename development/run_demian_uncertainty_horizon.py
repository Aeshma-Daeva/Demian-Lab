"""Fixed-centroid Jacobian horizon characterization using saved source states."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from development.run_demian_uncertainty_jacobian import linearized_predictions, make_transition
from development.run_demian_uncertainty_similarity import _excluded_fraction, residual_metrics, write_artifacts


def verify_source(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / 'manifest.json').read_text())
    for filename, digest in manifest['file_sha256'].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'source artifact hash mismatch: {filename}')
    encoded = json.dumps(manifest['protocol'], sort_keys=True).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest['protocol_sha256']:
        raise ValueError('source protocol hash mismatch')
    return manifest


def _reference_coverage(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    directions = points - center
    outside = directions - directions @ basis @ basis.T
    mean_shift = points.mean(axis=0) - center
    mean_outside = mean_shift - mean_shift @ basis @ basis.T
    return {
        'reference_direction_excluded_fraction': float(np.sum(outside ** 2)) / max(float(np.sum(directions ** 2)), 1e-20),
        'reference_centroid_shift_excluded_fraction': float(mean_outside @ mean_outside) / max(float(mean_shift @ mean_shift), 1e-20),
    }


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    root = Path(protocol['source_artifact_directory'])
    manifest = verify_source(root)
    source_protocol = manifest['protocol']
    states = np.load(root / 'states.npz')['states']
    metadata = [json.loads(line) for line in (root / 'state_metadata.jsonl').read_text().splitlines()]
    saved = np.load(root / 'transforms.npz')
    matrices: dict[tuple[int, str, str], np.ndarray] = {}
    for seed in source_protocol['model_seeds']:
        for family in ('train', 'heldout'):
            for variant in ('base', 'extra_delay'):
                indices = [index for index, row in enumerate(metadata) if row['model_seed'] == seed and row['family'] == family and row['variant'] == variant]
                matrices[(seed, family, variant)] = states[indices].astype(np.float64)
    metrics, target_states, target_metadata = [], [], []
    predictions, transforms = {}, {}
    anchor_matches = []
    double_errors = []
    step_index = len(source_protocol['train_reliability_schedule']) + int(source_protocol['base_delay_steps'])
    for seed in source_protocol['model_seeds']:
        seed = int(seed)
        train_ids = saved[f'{seed}_train_history_indices']
        heldout_ids = saved[f'{seed}_heldout_history_indices']
        center = matrices[(seed, 'train', 'base')][train_ids].mean(axis=0)
        basis = saved[f'{seed}_basis']
        transforms[f'{seed}_basis'] = basis
        transforms[f'{seed}_source_mean'] = center
        for horizon in protocol['horizons']:
            horizon = int(horizon)
            single = make_transition(seed=seed, hidden_size=int(source_protocol['hidden_size']), step_index=step_index, dtype=torch.float32, steps=horizon)
            double = make_transition(seed=seed, hidden_size=int(source_protocol['hidden_size']), step_index=step_index, dtype=torch.float64, steps=horizon)
            with torch.no_grad():
                mean_target = double(torch.as_tensor(center, dtype=torch.float64)).numpy()
            transforms[f'{seed}_{horizon}_centroid_target'] = mean_target
            for axis, family, indices in (
                ('same_order_histories', 'train', heldout_ids),
                ('unseen_order', 'heldout', np.arange(len(matrices[(seed, 'heldout', 'base')]))),
            ):
                a = matrices[(seed, family, 'base')][indices]
                targets = []
                with torch.no_grad():
                    for history_index, left in zip(indices, a, strict=True):
                        target = single(torch.as_tensor(left, dtype=torch.float32)).numpy()
                        targets.append(target.astype(np.float64))
                        if horizon == 2:
                            anchor_matches.append(bool(np.array_equal(target, matrices[(seed, family, 'extra_delay')][history_index])))
                        numerical = double(torch.as_tensor(left, dtype=torch.float64)).numpy()
                        double_errors.append(float(np.linalg.norm(numerical - target)))
                        target_states.append(target)
                        target_metadata.append({'model_seed': seed, 'axis': axis, 'horizon': horizon, 'history_index': int(history_index)})
                b = np.vstack(targets)
                maps = {
                    'identity': a,
                    'centroid_translation': a + mean_target - center,
                    'jacobian_full': linearized_predictions(double, center, a),
                }
                for name, predicted in maps.items():
                    predictions[f'{seed}_{horizon}_{axis}_{name}'] = predicted
                    metrics.append({
                        'model_seed': seed, 'axis': axis, 'horizon': horizon, 'map': name,
                        'eval_source_excluded_fraction': _excluded_fraction(a, basis),
                        'eval_target_excluded_fraction': _excluded_fraction(b, basis),
                        **_reference_coverage(a, center, basis),
                        **residual_metrics(a, b, predicted, basis),
                    })
    if not anchor_matches or not all(anchor_matches):
        raise RuntimeError('two-step anchor does not exactly match the cached B checkpoints')
    return {
        'source_commit': source_commit, 'states': np.vstack(target_states),
        'state_count': len(target_states), 'state_metadata': target_metadata,
        'transforms': transforms, 'predictions': predictions, 'metrics': metrics,
        'integrity': {
            'all_finite': bool(all(np.isfinite(value) for row in metrics for value in row.values() if isinstance(value, float))),
            'source_hashes_verified': True, 'exact_two_step_anchor': all(anchor_matches),
            'anchor_count': len(anchor_matches), 'double_rollout_max_norm_error': max(double_errors),
            'scope': 'same cached cases; fixed source-centroid local-linear models across horizons; no independent replication',
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
    print(json.dumps({'out': str(args.out), 'target_states': payload['state_count']}))


if __name__ == '__main__':
    main()
