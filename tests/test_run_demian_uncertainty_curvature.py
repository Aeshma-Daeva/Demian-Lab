import numpy as np
import torch

from development import run_demian_uncertainty_curvature as curvature


def test_directional_second_order_predicts_quadratic_on_unseen_points():
    def transition(state):
        x, y = state
        return torch.stack((x * x + x * y + 2., y * y - x))
    center = np.array([.2, -.3])
    points = np.array([[.7, .4], [-.1, .8]])
    actual = np.vstack([transition(torch.as_tensor(point)).numpy() for point in points])
    np.testing.assert_allclose(curvature.second_order_predictions(transition, center, points), actual, atol=1e-12)


def test_second_order_handles_zero_curvature_of_affine_transition():
    transition = lambda state: 2. * state + 3.
    points = np.asarray([[1., 2.], [3., 4.]])
    np.testing.assert_allclose(curvature.second_order_predictions(transition, np.zeros(2), points), 2. * points + 3.)


def test_cached_curvature_check_preserves_history_pairing_and_checks_derivative(tmp_path):
    from development import run_demian_uncertainty_jacobian as jacobian
    from development import run_demian_uncertainty_horizon as horizon
    source_protocol = {
        'protocol_id': 'test-source', 'claim_boundary': 'geometry only',
        'hidden_size': 16, 'model_seeds': [94], 'base_delay_steps': 2,
        'train_reliability_schedule': [.5, .75], 'heldout_reliability_schedule': [.75, .5],
        'allocation_seed': 31, 'split_seed': 41, 'subspace_rank': 1,
        'permutation_seed': 51, 'variants': ['base', 'extra_delay'],
    }
    base = tmp_path / 'base'
    jacobian.write_artifacts(base, source_protocol, jacobian.run_protocol(source_protocol))
    future = tmp_path / 'future'
    horizon_protocol = {'protocol_id': 'test-horizons', 'claim_boundary': 'descriptive', 'source_artifact_directory': str(base), 'horizons': [2, 4]}
    horizon.write_artifacts(future, horizon_protocol, horizon.run_protocol(horizon_protocol))
    result = curvature.run_protocol({
        'base_artifact_directory': str(base), 'source_artifact_directory': str(future),
        'horizons': [4], 'finite_difference_epsilon': .01, 'finite_difference_tolerance': .001,
    })
    assert result['integrity']['all_finite']
    assert len(result['metrics']) == 4
    assert result['state_count'] == 6
