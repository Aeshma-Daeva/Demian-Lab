import importlib
import importlib.util

import numpy as np


def chord_module():
    name = 'development.run_demian_uncertainty_chord'
    assert importlib.util.find_spec(name) is not None, 'natural-history chord runner missing'
    return importlib.import_module(name)


def test_pairs_cover_every_history_once_per_flip_in_both_orientations():
    chord = chord_module()
    pairs = chord.oriented_pairs(3, 0)
    assert pairs == [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)]
    assert chord.oriented_pairs(3, 2) == [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]


def test_quantized_chord_endpoint_is_exact_native_counterfactual():
    chord = chord_module()
    anchor = np.asarray([1e8, 1.], dtype=np.float32)
    endpoint = np.asarray([1e8+16, 3.], dtype=np.float32)
    point, effective = chord.quantized_chord(anchor, endpoint, 1.)
    assert np.array_equal(point, endpoint)
    np.testing.assert_array_equal(effective, [16., 2.])


def test_small_chord_run_preserves_endpoint_replay_and_matched_controls(tmp_path):
    chord = chord_module()
    protocol = {
        'protocol_id': 'test-chord', 'claim_boundary': 'geometry only',
        'hidden_size': 16, 'model_seeds': [94], 'base_delay_steps': 2,
        'reliability_schedule': [.95, .5], 'flip_positions': [0, 1],
        'horizons': [2], 'fractions': [.5, 1.], 'direction_seeds': [17],
        'precision_floor_multiple': 10., 'relative_linearity_threshold': .1,
    }
    payload = chord.run_protocol(protocol)
    assert payload['state_count'] == 4
    assert len(payload['metrics']) == 32
    assert payload['integrity']['exact_endpoint_continuation']
    assert payload['integrity']['endpoint_check_count'] == 8
    assert payload['integrity']['exact_baseline_continuation']
    assert payload['integrity']['all_finite']
    rows = payload['metrics']
    for start in range(0, len(rows), 2):
        natural, random = rows[start:start+2]
        assert natural['condition'] == 'natural_chord'
        assert random['condition'] == 'norm_matched_random'
        assert natural['requested_radius'] == random['requested_radius']
        assert abs(random['effective_radius']/random['requested_radius'] - 1) < .001
    chord.write_artifacts(tmp_path/'result', protocol, payload)
    assert (tmp_path/'result'/'predictions.npz').is_file()
