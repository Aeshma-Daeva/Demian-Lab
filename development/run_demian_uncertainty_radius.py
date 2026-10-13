"""Fresh reachable-state local sensitivity and native precision radius check."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from development.demian_uncertainty_probe import run_uncertainty_episode
from development.demian_uncertainty_world import AmbiguousEvidenceWorld
from development.run_demian_uncertainty_jacobian import make_transition
from development.run_demian_uncertainty_similarity import write_artifacts


def quantized_perturbation(anchor: np.ndarray, direction: np.ndarray, *, requested_radius: float, sign: int) -> tuple[np.ndarray, np.ndarray]:
    perturbed = (anchor.astype(np.float64) + sign * requested_radius * direction).astype(np.float32)
    return perturbed, perturbed.astype(np.float64) - anchor.astype(np.float64)


def _checkpoint(seed: int, hidden_size: int, evidence: tuple, delay: int) -> tuple[np.ndarray, list[int]]:
    frames = AmbiguousEvidenceWorld(hidden_target=0, evidence=evidence, delay_steps=delay).frames()[:len(evidence)+delay]
    trace = run_uncertainty_episode(model_seed=seed, hidden_size=hidden_size, public_frames=frames, checkpoint_phases=('delay',))
    parts = trace['checkpoints'][f'delay:{len(frames)-1}'].state
    return np.concatenate([part.detach().numpy().reshape(-1) for part in parts]), [part.numel() for part in parts]


def run_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    schedule = protocol['reliability_schedule']
    count = len(schedule) + int(protocol['base_delay_steps'])
    states, metadata, metrics = [], [], []
    transforms = {}
    perturbed_inputs, native_outputs, mathematical_responses, linear_responses = [], [], [], []
    baseline_checks = []
    for raw_seed in protocol['model_seeds']:
        seed = int(raw_seed)
        for history_index in protocol['history_indices']:
            if not 0 <= int(history_index) < 2 ** len(schedule):
                raise ValueError('history index exceeds public history support')
            symbols = [(int(history_index) >> (len(schedule)-1-position)) & 1 for position in range(len(schedule))]
            evidence = tuple(zip(symbols, schedule, strict=True))
            anchor, widths = _checkpoint(seed, int(protocol['hidden_size']), evidence, int(protocol['base_delay_steps']))
            anchor_index = len(states)
            states.append(anchor)
            metadata.append({'model_seed': seed, 'history_index': int(history_index), 'step_index': count, 'channel_widths': widths})
            for horizon in protocol['horizons']:
                horizon = int(horizon)
                native = make_transition(seed=seed, hidden_size=int(protocol['hidden_size']), step_index=count, dtype=torch.float32, steps=horizon)
                mathematical = make_transition(seed=seed, hidden_size=int(protocol['hidden_size']), step_index=count, dtype=torch.float64, steps=horizon)
                with torch.no_grad():
                    native_base = native(torch.as_tensor(anchor)).numpy()
                    math_base = mathematical(torch.as_tensor(anchor.astype(np.float64))).numpy()
                independent, _ = _checkpoint(seed, int(protocol['hidden_size']), evidence, int(protocol['base_delay_steps']) + horizon)
                baseline_checks.append(bool(np.array_equal(native_base, independent)))
                baseline_precision = float(np.linalg.norm(native_base.astype(np.float64) - math_base.astype(np.float32).astype(np.float64)))
                for direction_seed in protocol['direction_seeds']:
                    rng = np.random.default_rng(seed * 10000 + int(direction_seed))
                    direction = rng.normal(size=anchor.size)
                    direction /= np.linalg.norm(direction)
                    transforms[f'{seed}_{direction_seed}_direction'] = direction
                    for radius in protocol['radii']:
                        for sign in protocol['signs']:
                            perturbed, effective = quantized_perturbation(anchor, direction, requested_radius=float(radius), sign=int(sign))
                            _, jvp = torch.autograd.functional.jvp(mathematical, torch.as_tensor(anchor.astype(np.float64)), torch.as_tensor(effective), strict=True)
                            linear = jvp.detach().numpy()
                            with torch.no_grad():
                                native_target = native(torch.as_tensor(perturbed)).numpy()
                                math_target = mathematical(torch.as_tensor(perturbed.astype(np.float64))).numpy()
                            response = math_target - math_base
                            native_response = native_target.astype(np.float64) - native_base.astype(np.float64)
                            induced_norm = float(np.linalg.norm(response))
                            effective_norm = float(np.linalg.norm(effective))
                            precision = float(np.linalg.norm(native_target.astype(np.float64) - math_target.astype(np.float32).astype(np.float64)))
                            floor = max(baseline_precision, precision, 1e-12)
                            valid = bool(effective_norm > 0 and induced_norm >= float(protocol['precision_floor_multiple']) * floor)
                            curvature_error = float(np.linalg.norm(response - linear))
                            native_error = float(np.linalg.norm(native_response - linear))
                            denominator = max(induced_norm, 1e-20)
                            metrics.append({
                                'case_index': len(metrics), 'anchor_index': anchor_index,
                                'model_seed': seed, 'history_index': int(history_index),
                                'horizon': horizon, 'direction_seed': int(direction_seed),
                                'requested_radius': float(radius), 'sign': int(sign),
                                'effective_radius': effective_norm,
                                'radius_representability_ratio': effective_norm / float(radius),
                                'changed_coordinate_fraction': float(np.mean(perturbed != anchor)),
                                'induced_response_norm64': induced_norm,
                                'response_gain64': induced_norm / max(effective_norm, 1e-20),
                                'curvature_absolute_error': curvature_error,
                                'curvature_relative_error': curvature_error / denominator,
                                'native_absolute_error': native_error,
                                'native_relative_error': native_error / denominator,
                                'native_precision_discrepancy': precision,
                                'native_baseline_precision_discrepancy': baseline_precision,
                                'signal_to_precision_floor': induced_norm / floor,
                                'response_valid_above_precision_floor': valid,
                                'mathematical_linearity_within_threshold': bool(valid and curvature_error / denominator <= float(protocol['relative_linearity_threshold'])),
                            })
                            perturbed_inputs.append(perturbed)
                            native_outputs.append(native_target)
                            mathematical_responses.append(response)
                            linear_responses.append(linear)
    if not all(baseline_checks):
        raise RuntimeError('reachable anchor continuation differs from independent full replay')
    return {
        'source_commit': source_commit, 'state_count': len(states), 'states': np.vstack(states),
        'state_metadata': metadata, 'metrics': metrics, 'transforms': transforms,
        'predictions': {
            'perturbed_inputs32': np.vstack(perturbed_inputs),
            'native_outputs32': np.vstack(native_outputs),
            'mathematical_responses64': np.vstack(mathematical_responses),
            'linear_responses64': np.vstack(linear_responses),
        },
        'integrity': {
            'all_finite': bool(all(np.isfinite(value) for row in metrics for value in row.values() if isinstance(value, float))),
            'exact_baseline_continuation': all(baseline_checks), 'baseline_check_count': len(baseline_checks),
            'scope': 'fresh reachable anchors; two signed sampled directions; precision-limited cells remain inconclusive',
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
    print(json.dumps({'out': str(args.out), 'cases': len(payload['metrics'])}))


if __name__ == '__main__':
    main()
