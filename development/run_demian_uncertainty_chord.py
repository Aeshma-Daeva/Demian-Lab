"""Reachable-history chords versus norm-matched isotropic disturbances."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from development.run_demian_uncertainty_jacobian import make_transition
from development.run_demian_uncertainty_radius import _checkpoint, quantized_perturbation
from development.run_demian_uncertainty_similarity import write_artifacts


def oriented_pairs(length: int, position: int) -> list[tuple[int, int]]:
    if not 0 <= position < length:
        raise ValueError('flip position outside history')
    mask = 1 << (length - 1 - position)
    return [(index, index ^ mask) for index in range(2 ** length)]


def quantized_chord(anchor: np.ndarray, endpoint: np.ndarray, fraction: float):
    point = (anchor.astype(np.float64) + fraction * (endpoint.astype(np.float64) - anchor.astype(np.float64))).astype(np.float32)
    return point, point.astype(np.float64) - anchor.astype(np.float64)


def summarize_metrics(rows: list[dict], direction_seeds: list[int]) -> list[dict]:
    """Descriptive seedwise summaries; no independence or status inference."""
    group_keys = ('model_seed', 'flip_position', 'horizon', 'fraction')
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        natural = [row for row in group if row['condition'] == 'natural_chord']
        controls = [row for row in group if row['condition'] == 'norm_matched_random']
        by_source = defaultdict(dict)
        for row in controls:
            source = by_source[row['history_index']]
            if row['direction_seed'] in source:
                raise ValueError('duplicate source control direction')
            source[row['direction_seed']] = row
        deltas = []
        for row in natural:
            matched = by_source[row['history_index']]
            if set(matched) != set(direction_seeds):
                raise ValueError('missing or mismatched source control directions')
            if row['response_valid_above_precision_floor'] and all(c['response_valid_above_precision_floor'] for c in matched.values()):
                deltas.append(row['curvature_relative_error'] - float(np.mean([c['curvature_relative_error'] for c in matched.values()])))
        valid_natural = [row for row in natural if row['response_valid_above_precision_floor']]
        valid_controls = [row for row in controls if row['response_valid_above_precision_floor']]
        summaries.append({
            **dict(zip(group_keys, key, strict=True)), 'natural_case_count': len(natural),
            'natural_precision_valid_count': len(valid_natural),
            'control_case_count': len(controls), 'control_precision_valid_count': len(valid_controls),
            'natural_threshold_failure_count': sum(not row['mathematical_linearity_within_threshold'] for row in valid_natural),
            'control_threshold_failure_count': sum(not row['mathematical_linearity_within_threshold'] for row in valid_controls),
            'natural_median_error': float(np.median([row['curvature_relative_error'] for row in valid_natural])) if valid_natural else None,
            'natural_max_error': max((row['curvature_relative_error'] for row in valid_natural), default=None),
            'control_median_error': float(np.median([row['curvature_relative_error'] for row in valid_controls])) if valid_controls else None,
            'paired_valid_count': len(deltas),
            'median_paired_natural_minus_control_error': float(np.median(deltas)) if deltas else None,
            'requested_radius_median': float(np.median([row['requested_radius'] for row in natural])),
        })
    return summaries


def run_protocol(protocol: dict) -> dict:
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    schedule = protocol['reliability_schedule']
    length = len(schedule)
    count = length + int(protocol['base_delay_steps'])
    states, metadata, metrics = [], [], []
    transforms = {}
    predictions = {name: [] for name in ('perturbed_inputs32', 'native_outputs32', 'mathematical_responses64', 'linear_responses64')}
    baseline_checks, endpoint_checks = [], []
    for seed in protocol['model_seeds']:
        seed = int(seed)
        anchors, histories = [], []
        state_offset = len(states)
        for index in range(2 ** length):
            symbols = [(index >> (length-1-position)) & 1 for position in range(length)]
            evidence = tuple(zip(symbols, schedule, strict=True))
            anchor, widths = _checkpoint(seed, int(protocol['hidden_size']), evidence, int(protocol['base_delay_steps']))
            anchors.append(anchor)
            histories.append(evidence)
            states.append(anchor)
            metadata.append({'model_seed': seed, 'history_index': index, 'channel_widths': widths, 'step_index': count})
        directions = {}
        for direction_seed in protocol['direction_seeds']:
            direction = np.random.default_rng(seed*10000+int(direction_seed)).normal(size=anchors[0].size)
            direction /= np.linalg.norm(direction)
            directions[int(direction_seed)] = direction
            transforms[f'{seed}_{direction_seed}_direction'] = direction
        for horizon in protocol['horizons']:
            horizon = int(horizon)
            native = make_transition(seed=seed, hidden_size=int(protocol['hidden_size']), step_index=count, dtype=torch.float32, steps=horizon)
            mathematical = make_transition(seed=seed, hidden_size=int(protocol['hidden_size']), step_index=count, dtype=torch.float64, steps=horizon)
            native_bases, math_bases = [], []
            for anchor, evidence in zip(anchors, histories, strict=True):
                with torch.no_grad():
                    nb = native(torch.as_tensor(anchor)).numpy()
                    mb = mathematical(torch.as_tensor(anchor.astype(np.float64))).numpy()
                replay, _ = _checkpoint(seed, int(protocol['hidden_size']), evidence, int(protocol['base_delay_steps'])+horizon)
                baseline_checks.append(bool(np.array_equal(nb, replay)))
                native_bases.append(nb)
                math_bases.append(mb)
            for position in protocol['flip_positions']:
                for index, counterpart in oriented_pairs(length, int(position)):
                    anchor, endpoint = anchors[index], anchors[counterpart]
                    chord_norm = float(np.linalg.norm(endpoint.astype(np.float64)-anchor.astype(np.float64)))
                    native_base, math_base = native_bases[index], math_bases[index]
                    baseline_precision = float(np.linalg.norm(native_base.astype(np.float64)-math_base.astype(np.float32).astype(np.float64)))
                    for fraction in protocol['fractions']:
                        fraction = float(fraction)
                        radius = fraction * chord_norm
                        natural, natural_delta = quantized_chord(anchor, endpoint, fraction)
                        cases = [('natural_chord', None, natural, natural_delta)]
                        for direction_seed, direction in directions.items():
                            point, delta = quantized_perturbation(anchor, direction, requested_radius=radius, sign=1)
                            cases.append(('norm_matched_random', direction_seed, point, delta))
                        for condition, direction_seed, point, delta in cases:
                            _, derivative = torch.autograd.functional.jvp(mathematical, torch.as_tensor(anchor.astype(np.float64)), torch.as_tensor(delta), strict=True)
                            linear = derivative.detach().numpy()
                            with torch.no_grad():
                                nt = native(torch.as_tensor(point)).numpy()
                                mt = mathematical(torch.as_tensor(point.astype(np.float64))).numpy()
                            if condition == 'natural_chord' and fraction == 1.:
                                endpoint_checks.append(bool(np.array_equal(point, endpoint) and np.array_equal(nt, native_bases[counterpart])))
                            response = mt-math_base
                            native_response = nt.astype(np.float64)-native_base.astype(np.float64)
                            signal = float(np.linalg.norm(response))
                            effective = float(np.linalg.norm(delta))
                            precision = float(np.linalg.norm(nt.astype(np.float64)-mt.astype(np.float32).astype(np.float64)))
                            floor = max(baseline_precision, precision, 1e-12)
                            valid = bool(effective > 0 and signal >= float(protocol['precision_floor_multiple'])*floor)
                            error = float(np.linalg.norm(response-linear))
                            relative = error/max(signal, 1e-20)
                            metrics.append({
                                'case_index': len(metrics), 'anchor_index': state_offset+index,
                                'counterpart_index': state_offset+counterpart, 'model_seed': seed,
                                'history_index': index, 'counterpart_history_index': counterpart,
                                'flip_position': int(position), 'flipped_reliability': float(schedule[int(position)]),
                                'horizon': horizon, 'fraction': fraction, 'condition': condition,
                                'direction_seed': direction_seed, 'requested_radius': radius,
                                'effective_radius': effective, 'radius_representability_ratio': effective/max(radius, 1e-20),
                                'radius_to_anchor_norm': radius/max(float(np.linalg.norm(anchor.astype(np.float64))), 1e-20),
                                'changed_coordinate_fraction': float(np.mean(point != anchor)),
                                'induced_response_norm64': signal, 'response_gain64': signal/max(effective, 1e-20),
                                'curvature_absolute_error': error, 'curvature_relative_error': relative,
                                'native_relative_error': float(np.linalg.norm(native_response-linear))/max(signal, 1e-20),
                                'native_precision_discrepancy': precision, 'native_baseline_precision_discrepancy': baseline_precision,
                                'signal_to_precision_floor': signal/floor,
                                'response_valid_above_precision_floor': valid,
                                'mathematical_linearity_within_threshold': bool(valid and relative <= float(protocol['relative_linearity_threshold'])),
                            })
                            for key, value in zip(predictions, (point, nt, response, linear), strict=True):
                                predictions[key].append(value)
            print(json.dumps({'completed_seed': seed, 'horizon': horizon, 'cases': len(metrics)}), flush=True)
    if not all(baseline_checks) or not all(endpoint_checks):
        raise RuntimeError('native history endpoint replay mismatch')
    return {
        'source_commit': source_commit, 'state_count': len(states), 'states': np.vstack(states),
        'state_metadata': metadata, 'metrics': metrics, 'transforms': transforms,
        'predictions': {key: np.vstack(value) for key, value in predictions.items()},
        'integrity': {
            'all_finite': bool(all(np.isfinite(value) for row in metrics for value in row.values() if isinstance(value, float))),
            'exact_baseline_continuation': all(baseline_checks), 'baseline_check_count': len(baseline_checks),
            'exact_endpoint_continuation': all(endpoint_checks), 'endpoint_check_count': len(endpoint_checks),
            'scope': 'all public histories; oriented one-cue chords; interpolants not asserted reachable; precision-limited cells inconclusive',
        },
    }


def main():
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
