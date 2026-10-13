"""Time-resolved frozen public-evidence calibration; early prefixes descriptive."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from development.demian_uncertainty_probe import encode_evidence, run_uncertainty_episode
from development.demian_uncertainty_world import AmbiguousEvidenceWorld, reference_belief
from development.demian_v1_gate_state import V1_CHANNELS
from development.run_demian_uncertainty_similarity import write_artifacts


def fit_probe(features, labels, *, alpha):
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.column_stack(((x-mean)/scale, np.ones(len(x))))
    regularizer = alpha*np.eye(design.shape[1])
    regularizer[-1, -1] = 0.
    weights = np.linalg.solve(design.T@design+regularizer, design.T@y)
    return {'mean': mean, 'scale': scale, 'weights': weights}


def predict_probe(fit, features):
    x = np.asarray(features, dtype=np.float64)
    return np.column_stack(((x-fit['mean'])/fit['scale'], np.ones(len(x))))@fit['weights']


def calibration_metrics(target, prediction):
    y, p = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    total = float(np.sum((y-y.mean())**2))
    slope = float(np.dot(y-y.mean(), p-p.mean())/total) if total > 1e-12 else 0.
    return {'r2': 1.-float(np.sum((y-p)**2))/total if total > 1e-12 else 0.,
            'pearson_r': float(np.corrcoef(y, p)[0, 1]) if y.std()>1e-12 and p.std()>1e-12 else 0.,
            'mae': float(np.mean(np.abs(y-p))), 'calibration_slope': slope,
            'calibration_intercept': float(p.mean()-slope*y.mean()),
            'target_std': float(y.std()), 'prediction_std': float(p.std())}


def collect_rows(protocol):
    train, evaluation = protocol['train_order'], protocol['evaluation_order']
    if len(train) != len(evaluation) or train == evaluation:
        raise ValueError('distinct equal-length reliability orders required')
    checkpoints = protocol['checkpoints']
    length = len(train)
    for point in checkpoints:
        count, delay = int(point['evidence_count']), int(point['delay_steps'])
        if not 1 <= count <= length or delay < 0 or (delay and count != length):
            raise ValueError('invalid evidence/delay checkpoint')
        if sorted(train[:count]) != sorted(evaluation[:count]):
            raise ValueError('checkpoint prefix reliability multiset mismatch')
    max_delay = max(int(point['delay_steps']) for point in checkpoints)
    rows = []
    for seed in protocol['model_seeds']:
        for order_name, schedule in [('train', train), ('heldout', evaluation)]:
            seen = {}
            for symbols in itertools.product((0, 1), repeat=length):
                evidence = tuple(zip(symbols, schedule, strict=True))
                frames = AmbiguousEvidenceWorld(hidden_target=0, evidence=evidence, delay_steps=max_delay).frames()[:length+max_delay]
                trace = run_uncertainty_episode(model_seed=int(seed), hidden_size=int(protocol['hidden_size']), public_frames=frames, checkpoint_phases=('evidence', 'delay'))
                for point in checkpoints:
                    count, delay = int(point['evidence_count']), int(point['delay_steps'])
                    tick = count+delay-1
                    phase = 'delay' if delay else 'evidence'
                    snapshot = trace['checkpoints'][f'{phase}:{tick}']
                    parts = [p.detach().numpy().reshape(-1) for p in snapshot.state]
                    state = np.concatenate(parts)
                    key = (point['id'], symbols[:count])
                    if key in seen:
                        if not np.array_equal(state, seen[key]):
                            raise RuntimeError('identical prefix differs across future suffixes')
                        continue
                    seen[key] = state
                    prefix = frames[:count]
                    rows.append({
                        'model_seed': int(seed), 'order': order_name, 'checkpoint_id': point['id'],
                        'evidence_count': count, 'delay_steps': delay, 'tick': tick,
                        'symbols': symbols[:count], 'reliabilities': schedule[:count],
                        'unique_prefix_count': 2**count, 'channel_widths': [len(p) for p in parts],
                        'target_value': float(reference_belief(prefix)['log_odds']),
                        'current_semantic_log_odds': float(reference_belief([frames[tick]])['log_odds']),
                        'latest_semantic_log_odds': float(reference_belief([frames[count-1]])['log_odds']),
                        'features': {'full_state': state, **dict(zip(V1_CHANNELS, parts, strict=True)),
                            'current_observation': encode_evidence(frames[tick], int(protocol['hidden_size'])).numpy().reshape(-1)[:10],
                            'latest_observation': encode_evidence(frames[count-1], int(protocol['hidden_size'])).numpy().reshape(-1)[:10],
                            'schedule_only': np.asarray([count, delay, *schedule[:count], *([0.]*(length-count))])},
                    })
            print(json.dumps({'collected_seed': int(seed), 'order': order_name, 'unique_rows': len(seen)}), flush=True)
    return rows


def _shuffled_labels(rows, seed):
    labels = np.asarray([r['target_value'] for r in rows])
    rng = np.random.default_rng(seed)
    for symbol in (0, 1):
        indexes = [i for i, r in enumerate(rows) if r['symbols'][-1] == symbol]
        labels[indexes] = rng.permutation(labels[indexes])
    return labels


def _history_index(row):
    return sum(int(symbol) << (len(row['symbols'])-1-position) for position, symbol in enumerate(row['symbols']))


def split_histories(rows, seed):
    groups = defaultdict(list)
    for row in rows:
        label_sign = int(np.sign(row['target_value'])) if abs(row['target_value']) > 1e-12 else 0
        groups[(row['symbols'][-1], label_sign)].append(_history_index(row))
    rng = np.random.default_rng(seed)
    training, evaluation = [], []
    for key in sorted(groups):
        indexes = rng.permutation(sorted(groups[key])).tolist()
        if len(indexes) % 2:
            raise ValueError('history split strata require even support')
        training.extend(indexes[:len(indexes)//2])
        evaluation.extend(indexes[len(indexes)//2:])
    if set(training) & set(evaluation):
        raise RuntimeError('history allocation overlap')
    return {'seed': int(seed), 'train_history_indices': sorted(training),
            'evaluation_history_indices': sorted(evaluation),
            'strata': 'train-order final symbol and public log-odds sign, zero tolerance1e-12; paired allocation fixed across orders, times and parameter seeds'}


def run_protocol(protocol):
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    rows = collect_rows(protocol)
    allocation = None
    if 'history_split_seed' in protocol:
        if any(point['evidence_count'] != len(protocol['train_order']) for point in protocol['checkpoints']):
            raise ValueError('heldout-history matrix requires full-history checkpoints')
        reference_rows = [row for row in rows if row['model_seed']==int(protocol['model_seeds'][0]) and row['order']=='train' and row['checkpoint_id']==protocol['frozen_reference_checkpoint']]
        allocation = split_histories(reference_rows, int(protocol['history_split_seed']))
    metrics, shifts, transforms, prediction_rows = [], [], {}, []
    dimension = int(protocol['projection_dimension'])
    neural = ('full_state', *V1_CHANNELS)
    controls = ('current_observation', 'latest_observation', 'schedule_only')
    for seed in protocol['model_seeds']:
        cells = {(point['id'], order): [r for r in rows if r['model_seed']==int(seed) and r['checkpoint_id']==point['id'] and r['order']==order]
                 for point in protocol['checkpoints'] for order in ('train', 'heldout')}
        def evaluation_rows(cp, order):
            return [r for r in cells[(cp, order)] if allocation is None or _history_index(r) in allocation['evaluation_history_indices']]
        def family(order):
            return ('within_order_heldout_history' if order=='train' else 'cross_order_heldout_history') if allocation is not None else order
        for point in protocol['checkpoints']:
            cp = point['id']
            for order in ('train', 'heldout'):
                evaluate = evaluation_rows(cp, order)
                y = np.asarray([r['target_value'] for r in evaluate])
                for feature in ('current_semantic_log_odds', 'latest_semantic_log_odds'):
                    prediction = np.asarray([r[feature] for r in evaluate])
                    score_index, start = len(metrics), len(prediction_rows)
                    prediction_rows.extend([[score_index, i, float(truth), float(pred)] for i, (truth, pred) in enumerate(zip(y, prediction, strict=True))])
                    metrics.append({'score_index': score_index, 'model_seed': int(seed), 'checkpoint_id': cp,
                        'fit_checkpoint_id': None, 'fit_mode': 'analytic_public_baseline', 'evaluation_order': order,
                        'evaluation_family': family(order),
                        'feature': feature, 'label_kind': 'known_public_semantics', 'projection_seed': None,
                        'projection_dimension_actual': None, 'shuffle_seed': None, 'shuffled_label_changed_count': None,
                        'current_and_latest_identical': int(point['delay_steps']) == 0,
                        'train_unique_prefixes': 0, 'evaluation_unique_prefixes': len(evaluate),
                        'scope': 'primary' if int(point['evidence_count'])==len(protocol['train_order']) else 'descriptive_small_support',
                        'prediction_row_start': start, 'prediction_row_stop': len(prediction_rows),
                        **calibration_metrics(y, prediction)})
            for order in ('train', 'heldout'):
                source = np.vstack([r['features']['full_state'] for r in cells[(cp, 'train')]]).astype(np.float64)
                target = np.vstack([r['features']['full_state'] for r in cells[(cp, order)]]).astype(np.float64)
                reference = np.vstack([r['features']['full_state'] for r in cells[(protocol['frozen_reference_checkpoint'], 'train')]]).astype(np.float64)
                shifts.append({'model_seed': int(seed), 'checkpoint_id': cp, 'order': order,
                    'order_centroid_shift': float(np.linalg.norm(target.mean(axis=0)-source.mean(axis=0))),
                    'frozen_reference_centroid_shift': float(np.linalg.norm(target.mean(axis=0)-reference.mean(axis=0))),
                    'state_norm_mean': float(np.linalg.norm(target, axis=1).mean()),
                    'channel_norm_means': {name: float(np.linalg.norm(np.vstack([r['features'][name] for r in cells[(cp, order)]]).astype(np.float64), axis=1).mean()) for name in V1_CHANNELS}})
            for mode in ('time_refit', 'cue_end_frozen'):
                if mode == 'cue_end_frozen' and int(point['evidence_count']) != len(protocol['train_order']):
                    continue
                fit_cp = cp if mode=='time_refit' else protocol['frozen_reference_checkpoint']
                train_rows = [r for r in cells[(fit_cp, 'train')] if allocation is None or _history_index(r) in allocation['train_history_indices']]
                labels = np.asarray([r['target_value'] for r in train_rows])
                for feature in (*neural, *controls):
                    projection_seeds = protocol['projection_seeds'] if feature in neural else [None]
                    for projection_seed in projection_seeds:
                        x = np.vstack([r['features'][feature] for r in train_rows]).astype(np.float64)
                        basis = None
                        if projection_seed is not None:
                            basis, _ = np.linalg.qr(np.random.default_rng(int(projection_seed)).normal(size=(x.shape[1], min(dimension, x.shape[1]))))
                            x = x@basis
                        kinds = ('true_labels', 'conditional_shuffle') if feature == 'full_state' else ('true_labels',)
                        for label_kind in kinds:
                            shuffle_seed = int(seed)*10000+int(train_rows[0]['tick'])*100+int(projection_seed or 0)
                            fitted_labels = _shuffled_labels(train_rows, shuffle_seed) if label_kind=='conditional_shuffle' else labels
                            fit = fit_probe(x, fitted_labels, alpha=float(protocol['ridge_alpha']))
                            fit_id = f'{seed}_{fit_cp}_{feature}_{projection_seed}_{label_kind}'
                            for name, value in fit.items():
                                transforms[fit_id+'_'+name] = value
                            if basis is not None:
                                transforms[fit_id+'_projection'] = basis
                            for order in (('heldout',) if mode=='time_refit' and allocation is None else ('train', 'heldout')):
                                evaluate = evaluation_rows(cp, order)
                                features = np.vstack([r['features'][feature] for r in evaluate]).astype(np.float64)
                                if basis is not None:
                                    features = features@basis
                                prediction = predict_probe(fit, features)
                                y = np.asarray([r['target_value'] for r in evaluate])
                                score_index = len(metrics)
                                start = len(prediction_rows)
                                prediction_rows.extend([[score_index, i, float(truth), float(pred)] for i, (truth, pred) in enumerate(zip(y, prediction, strict=True))])
                                metrics.append({'score_index': score_index, 'model_seed': int(seed), 'checkpoint_id': cp,
                                    'fit_checkpoint_id': fit_cp, 'fit_mode': mode, 'evaluation_order': order,
                                    'evaluation_family': family(order),
                                    'feature': feature, 'label_kind': label_kind, 'projection_seed': projection_seed,
                                    'projection_dimension_actual': x.shape[1], 'train_unique_prefixes': len(train_rows),
                                    'evaluation_unique_prefixes': len(evaluate), 'shuffle_seed': shuffle_seed if label_kind=='conditional_shuffle' else None,
                                    'shuffled_label_changed_count': int(np.sum(fitted_labels != labels)) if label_kind=='conditional_shuffle' else None,
                                    'scope': 'primary' if int(point['evidence_count'])==len(protocol['train_order']) else 'descriptive_small_support',
                                    'prediction_row_start': start, 'prediction_row_stop': len(prediction_rows),
                                    **calibration_metrics(y, prediction)})
        print(json.dumps({'scored_seed': int(seed), 'scores': len(metrics)}), flush=True)
    metadata = [{k:v for k,v in row.items() if k!='features'} for row in rows]
    return {'source_commit': source_commit,
        'state_count': len(rows), 'states': np.vstack([r['features']['full_state'] for r in rows]),
        'state_metadata': metadata, 'metrics': metrics, 'transforms': transforms,
        'predictions': {'score_and_prefix_target_prediction': np.asarray(prediction_rows)},
        'integrity': {'all_finite': bool(all(np.isfinite(v) for row in metrics for v in row.values() if isinstance(v, float))),
                      **({'history_split': allocation} if allocation is not None else {}),
                      'unique_prefixes_deduplicated': True, 'suffix_invariance_checked': True,
                      'state_shift_rows': shifts, 'scope': 'train-only frozen coefficients and scale; early checkpoints descriptive; no erasure or causal interpretation'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payload = run_protocol(protocol)
    write_artifacts(args.out, protocol, payload)
    print(json.dumps({'out': str(args.out), 'scores': len(payload['metrics'])}))


if __name__ == '__main__':
    main()
