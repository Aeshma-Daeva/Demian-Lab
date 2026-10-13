import numpy as np
import torch

from development import run_demian_uncertainty_jacobian as geometry


def test_history_split_is_disjoint_and_balances_terminal_symbol():
    train, heldout = geometry.split_history_indices(history_length=6, seed=41)
    assert len(train) == len(heldout) == 32
    assert set(train).isdisjoint(heldout)
    assert sum(index % 2 for index in train) == 16
    assert sum(index % 2 for index in heldout) == 16


def test_full_state_jvp_predicts_known_affine_transition():
    matrix = torch.tensor([[1., 2.], [0., 3.]], dtype=torch.float64)
    transition = lambda state: state @ matrix + 4.
    center = np.asarray([.5, .25])
    points = np.asarray([[1., 0.], [0., 2.]])
    predictions = geometry.linearized_predictions(transition, center, points)
    np.testing.assert_allclose(predictions, points @ matrix.numpy() + 4.)


def test_small_jacobian_protocol_has_exact_continuation_and_both_axes(tmp_path):
    protocol = {
        'protocol_id': 'test-jacobian', 'claim_boundary': 'geometry only',
        'hidden_size': 16, 'model_seeds': [94], 'base_delay_steps': 2,
        'train_reliability_schedule': [.5, .75, .95],
        'heldout_reliability_schedule': [.95, .5, .75],
        'allocation_seed': 31, 'split_seed': 41, 'subspace_rank': 2,
        'permutation_seed': 51, 'variants': ['base', 'extra_delay'],
    }
    payload = geometry.run_protocol(protocol)
    assert payload['integrity']['exact_float32_continuation']
    assert payload['integrity']['all_finite']
    assert {row['axis'] for row in payload['metrics']} == {'same_order_histories', 'unseen_order'}
    assert len(payload['metrics']) == 12
    geometry.write_artifacts(tmp_path / 'result', protocol, payload)
    assert (tmp_path / 'result' / 'predictions.npz').is_file()
