import numpy as np

from development import run_demian_uncertainty_radius as radius


def test_quantized_perturbation_uses_actual_delta_without_renormalizing():
    anchor = np.asarray([1e8, 1.], dtype=np.float32)
    direction = np.asarray([1., 0.])
    perturbed, effective = radius.quantized_perturbation(anchor, direction, requested_radius=1., sign=1)
    assert np.array_equal(perturbed, anchor)
    assert np.linalg.norm(effective) == 0.


def test_small_reachable_radius_protocol_retains_signs_and_precision_metrics(tmp_path):
    protocol = {
        'protocol_id': 'test-radius', 'claim_boundary': 'geometry only',
        'hidden_size': 16, 'model_seeds': [94], 'base_delay_steps': 2,
        'reliability_schedule': [.5, .75, .95], 'history_indices': [0, 1],
        'horizons': [2], 'direction_seeds': [17], 'radii': [.001, .1],
        'signs': [-1, 1], 'precision_floor_multiple': 10.,
        'relative_linearity_threshold': .1,
    }
    payload = radius.run_protocol(protocol)
    assert payload['state_count'] == 2
    assert len(payload['metrics']) == 8
    assert {row['sign'] for row in payload['metrics']} == {-1, 1}
    assert payload['integrity']['all_finite']
    assert all(row['effective_radius'] > 0 for row in payload['metrics'])
    radius.write_artifacts(tmp_path / 'result', protocol, payload)
    assert (tmp_path / 'result' / 'predictions.npz').is_file()
