import importlib
import importlib.util

import numpy as np


def localization_module():
    name = 'development.run_demian_uncertainty_localization'
    assert importlib.util.find_spec(name) is not None, 'time-resolved localization runner missing'
    return importlib.import_module(name)


def test_frozen_fit_uses_train_scale_and_retains_calibration():
    module = localization_module()
    fit = module.fit_probe(np.asarray([[-1.], [1.]]), np.asarray([-2., 2.]), alpha=0.)
    predictions = module.predict_probe(fit, np.asarray([[2.], [4.]]))
    np.testing.assert_allclose(predictions, [4., 8.])
    score = module.calibration_metrics(np.asarray([2., 4.]), predictions)
    assert score['calibration_slope'] == 2.
    assert abs(score['calibration_intercept']) < 1e-12


def test_collection_deduplicates_prefix_states_before_probe_fits():
    module = localization_module()
    protocol = {
        'hidden_size': 16, 'model_seeds': [94],
        'train_order': [.95, .75, .5, .75],
        'evaluation_order': [.75, .95, .75, .5],
        'checkpoints': [
            {'id': 'cue2', 'evidence_count': 2, 'delay_steps': 0},
            {'id': 'cue4', 'evidence_count': 4, 'delay_steps': 0},
            {'id': 'delay2', 'evidence_count': 4, 'delay_steps': 2},
        ],
    }
    rows = module.collect_rows(protocol)
    assert len(rows) == 72
    assert sum(r['checkpoint_id'] == 'cue2' for r in rows) == 8
    assert sum(r['checkpoint_id'] == 'cue4' for r in rows) == 32
    assert sum(r['checkpoint_id'] == 'delay2' for r in rows) == 32
    assert all(r['unique_prefix_count'] == 2 ** r['evidence_count'] for r in rows)
    np.testing.assert_allclose(sorted({r['target_value'] for r in rows if r['checkpoint_id']=='cue2'}), sorted({
        -np.log(19.)-np.log(3.), -np.log(19.)+np.log(3.),
        np.log(19.)-np.log(3.), np.log(19.)+np.log(3.),
    }), rtol=1e-14, atol=1e-14)


def test_small_localization_run_has_time_refit_and_cue_end_frozen_controls(tmp_path):
    module = localization_module()
    protocol = {
        'protocol_id': 'test-localization', 'claim_boundary': 'decoding only',
        'hidden_size': 16, 'model_seeds': [94],
        'train_order': [.95, .75], 'evaluation_order': [.75, .95],
        'checkpoints': [{'id': 'cue2', 'evidence_count': 2, 'delay_steps': 0},
                        {'id': 'delay2', 'evidence_count': 2, 'delay_steps': 2}],
        'frozen_reference_checkpoint': 'cue2', 'projection_dimension': 2,
        'projection_seeds': [17], 'ridge_alpha': 1.,
    }
    payload = module.run_protocol(protocol)
    assert payload['state_count'] == 16
    assert {r['fit_mode'] for r in payload['metrics']} == {'time_refit', 'cue_end_frozen', 'analytic_public_baseline'}
    assert {r['evaluation_order'] for r in payload['metrics'] if r['fit_mode']=='cue_end_frozen'} == {'train', 'heldout'}
    assert payload['integrity']['all_finite']
    assert payload['transforms']
    analytic = [r for r in payload['metrics'] if r['fit_mode']=='analytic_public_baseline']
    assert len(analytic) == 8
    assert all(r['fit_checkpoint_id'] is None for r in analytic)
    current_delay = [r for r in analytic if r['checkpoint_id']=='delay2' and r['feature']=='current_semantic_log_odds']
    assert all(r['prediction_std'] == 0. and r['r2'] == 0. for r in current_delay)
    module.write_artifacts(tmp_path/'out', protocol, payload)
    assert (tmp_path/'out'/'predictions.npz').is_file()
