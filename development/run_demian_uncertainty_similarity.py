"""Held-out, label-free similarity maps for paired Demian state clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from development.demian_uncertainty_probe import run_uncertainty_episode
from development.demian_uncertainty_world import AmbiguousEvidenceWorld
from development.run_demian_uncertainty_length_invariance import _base_specs


VARIANTS = ('base', 'extra_delay', '00', '01', '10', '11')


def fit_similarity(source: np.ndarray, target: np.ndarray, basis: np.ndarray) -> dict[str, Any]:
    """Fit one proper rotation and nonnegative scale in a training subspace."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    left = (source - source_mean) @ basis
    right = (target - target_mean) @ basis
    u, _, vt = np.linalg.svd(left.T @ right)
    signs = np.ones(basis.shape[1])
    signs[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = (u * signs) @ vt
    denominator = float(np.sum(left ** 2))
    scale = max(0., float(np.sum((left @ rotation) * right)) / denominator) if denominator > 1e-20 else 1.
    return {'source_mean': source_mean, 'target_mean': target_mean, 'basis': basis, 'rotation': rotation, 'scale': scale}


def predict_similarity(source: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    centered = source - fitted['source_mean']
    coordinates = centered @ fitted['basis']
    correction = fitted['scale'] * coordinates @ fitted['rotation'] - coordinates
    return fitted['target_mean'] + centered + correction @ fitted['basis'].T


def residual_metrics(source: np.ndarray, target: np.ndarray, prediction: np.ndarray, basis: np.ndarray | None = None) -> dict[str, float]:
    residual = target - prediction
    centered_residual = residual - residual.mean(axis=0)
    scatter = float(np.mean(np.sum((source - source.mean(axis=0)) ** 2, axis=1)))
    scale = max(scatter, 1e-20) ** .5
    mse = float(np.mean(np.sum(residual ** 2, axis=1)))
    displacement_mse = float(np.mean(np.sum((target - source) ** 2, axis=1)))
    metrics = {
        'residual_rmse': mse ** .5,
        'residual_relative_to_scatter': mse ** .5 / scale,
        'centered_residual_relative_to_scatter': float(np.mean(np.sum(centered_residual ** 2, axis=1))) ** .5 / scale,
        'centroid_residual_relative_to_scatter': float(np.linalg.norm(residual.mean(axis=0))) / scale,
        'displacement_fraction_explained': 1. - mse / max(displacement_mse, 1e-20),
        'source_scatter_rms': scale,
    }
    if basis is not None:
        in_subspace = residual @ basis
        outside = residual - in_subspace @ basis.T
        metrics['in_subspace_residual_relative_to_scatter'] = float(np.mean(np.sum(in_subspace ** 2, axis=1))) ** .5 / scale
        metrics['outside_subspace_residual_relative_to_scatter'] = float(np.mean(np.sum(outside ** 2, axis=1))) ** .5 / scale
    return metrics


def _basis(clouds: list[np.ndarray], rank: int) -> np.ndarray:
    centered = np.vstack([cloud - cloud.mean(axis=0) for cloud in clouds])
    if rank <= 0 or rank > min(centered.shape):
        raise ValueError('subspace rank exceeds the available training matrix')
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:rank].T


def _excluded_fraction(cloud: np.ndarray, basis: np.ndarray) -> float:
    centered = cloud - cloud.mean(axis=0)
    residual = centered - centered @ basis @ basis.T
    return float(np.sum(residual ** 2)) / max(float(np.sum(centered ** 2)), 1e-20)


def collect_clouds(protocol: dict[str, Any]) -> tuple[dict[tuple[int, str, str], np.ndarray], list[dict[str, Any]], np.ndarray]:
    if protocol['train_reliability_schedule'] == protocol['heldout_reliability_schedule']:
        raise ValueError('heldout reliability order must be disjoint')
    if sorted(protocol['train_reliability_schedule']) != sorted(protocol['heldout_reliability_schedule']):
        raise ValueError('train and heldout require the same reliability multiset')
    clouds: dict[tuple[int, str, str], np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    for seed in protocol['model_seeds']:
        for family in ('train', 'heldout'):
            specs = _base_specs([protocol[f'{family}_reliability_schedule']], family=family, allocation_seed=int(protocol['allocation_seed']) + (family == 'heldout'), offset=0)
            for variant in protocol.get('variants', VARIANTS):
                collected = []
                for spec in specs:
                    evidence = spec['base_evidence']
                    delay = int(protocol['base_delay_steps']) + (2 if variant == 'extra_delay' else 0)
                    if variant not in ('base', 'extra_delay'):
                        evidence += tuple((int(symbol), .5) for symbol in variant)
                    frames = AmbiguousEvidenceWorld(hidden_target=spec['target'], evidence=evidence, delay_steps=delay).frames()[:len(evidence) + delay]
                    trace = run_uncertainty_episode(model_seed=int(seed), hidden_size=int(protocol['hidden_size']), public_frames=frames, checkpoint_phases=('delay',))
                    parts = trace['checkpoints'][f'delay:{len(frames)-1}'].state
                    state = np.concatenate([part.detach().cpu().numpy().reshape(-1) for part in parts])
                    collected.append(state.astype(np.float64))
                    states.append(state)
                    metadata.append({'model_seed': int(seed), 'family': family, 'variant': variant, 'history_index': spec['history_index'], 'channel_widths': [int(part.numel()) for part in parts]})
                clouds[(int(seed), family, variant)] = np.vstack(collected)
    return clouds, metadata, np.vstack(states)


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    clouds, metadata, states = collect_clouds(protocol)
    metrics: list[dict[str, Any]] = []
    transforms: dict[str, np.ndarray] = {}
    for seed in protocol['model_seeds']:
        seed = int(seed)
        training = [clouds[(seed, 'train', variant)] for variant in VARIANTS]
        basis = _basis(training, int(protocol['subspace_rank']))
        transforms[f'{seed}_basis'] = basis
        train_a, eval_a = clouds[(seed, 'train', 'base')], clouds[(seed, 'heldout', 'base')]
        for variant in VARIANTS[1:]:
            train_b, eval_b = clouds[(seed, 'train', variant)], clouds[(seed, 'heldout', variant)]
            fit = fit_similarity(train_a, train_b, basis)
            permutation = np.random.default_rng(seed + int(protocol['permutation_seed'])).permutation(len(train_b))
            shuffled = fit_similarity(train_a, train_b[permutation], basis)
            for name in ('source_mean', 'target_mean', 'rotation'):
                transforms[f'{seed}_{variant}_{name}'] = fit[name]
            transforms[f'{seed}_{variant}_scale'] = np.asarray(fit['scale'])
            predictions = {
                'identity': eval_a,
                'translation_only': eval_a + train_b.mean(axis=0) - train_a.mean(axis=0),
                'similarity': predict_similarity(eval_a, fit),
                'pair_permuted_similarity': predict_similarity(eval_a, shuffled),
            }
            for name, predicted in predictions.items():
                metrics.append({
                    'model_seed': seed, 'variant': variant, 'map': name,
                    'fitted_scale': float(fit['scale']) if name == 'similarity' else None,
                    'train_centroid_shift_norm': float(np.linalg.norm(train_b.mean(axis=0) - train_a.mean(axis=0))),
                    'train_subspace_centroid_shift_norm': float(np.linalg.norm((train_b.mean(axis=0) - train_a.mean(axis=0)) @ basis)),
                    'train_source_excluded_fraction': _excluded_fraction(train_a, basis),
                    'train_target_excluded_fraction': _excluded_fraction(train_b, basis),
                    'eval_source_excluded_fraction': _excluded_fraction(eval_a, basis),
                    'eval_target_excluded_fraction': _excluded_fraction(eval_b, basis),
                    **residual_metrics(eval_a, eval_b, predicted, basis),
                })
    return {
        'source_commit': source_commit, 'state_count': len(states), 'states': states,
        'state_metadata': metadata, 'transforms': transforms, 'metrics': metrics,
        'integrity': {'all_finite': bool(np.isfinite(states).all() and all(np.isfinite(value) for row in metrics for value in row.values() if isinstance(value, float))), 'scope': 'paired training-state fit; heldout-order scoring; no evidence labels used'},
    }


def write_artifacts(out: Path, protocol: dict[str, Any], payload: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'states.npz', states=payload['states'])
    np.savez_compressed(out / 'transforms.npz', **payload['transforms'])
    if 'predictions' in payload:
        np.savez_compressed(out / 'predictions.npz', **payload['predictions'])
    for filename, rows in (('metrics.jsonl', payload['metrics']), ('state_metadata.jsonl', payload['state_metadata'])):
        with (out / filename).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    (out / 'summary.json').write_text(json.dumps({'experiment': protocol['protocol_id'], 'claim_boundary': protocol['claim_boundary'], 'state_count': payload['state_count'], 'metric_count': len(payload['metrics']), 'integrity': payload['integrity']}, indent=2, sort_keys=True) + '\n')
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in out.iterdir()}
    (out / 'manifest.json').write_text(json.dumps({'protocol': protocol, 'protocol_sha256': hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(), 'source_commit': payload['source_commit'], 'file_sha256': hashes}, indent=2, sort_keys=True) + '\n')


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
