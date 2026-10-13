import numpy as np

from development import run_demian_uncertainty_similarity as similarity


def test_similarity_recovers_translation_rotation_and_scale_on_new_points():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(24, 3))
    evaluation = rng.normal(size=(10, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    offset = np.asarray([5., -2., 7.])
    target = 1.3 * train @ rotation + offset
    fitted = similarity.fit_similarity(train, target, np.eye(3))
    np.testing.assert_allclose(similarity.predict_similarity(evaluation, fitted), 1.3 * evaluation @ rotation + offset, atol=1e-10)


def test_geometry_residual_separates_shared_offset_from_deformation():
    cloud = np.asarray([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    result = similarity.residual_metrics(cloud, cloud + 10., cloud)
    assert result['centered_residual_relative_to_scatter'] < 1e-10
    assert result['centroid_residual_relative_to_scatter'] > 1.


def test_small_protocol_preserves_paired_states_and_writes_provenance(tmp_path):
    protocol = {
        'protocol_id': 'test-similarity', 'claim_boundary': 'descriptive',
        'hidden_size': 16, 'model_seeds': [94], 'base_delay_steps': 2,
        'train_reliability_schedule': [.5, .75], 'heldout_reliability_schedule': [.75, .5],
        'allocation_seed': 31, 'subspace_rank': 2, 'permutation_seed': 41,
    }
    payload = similarity.run_protocol(protocol)
    assert payload['state_count'] == 48
    assert len(payload['metrics']) == 20
    assert payload['integrity']['all_finite']
    similarity.write_artifacts(tmp_path / 'artifact', protocol, payload)
    assert (tmp_path / 'artifact' / 'states.npz').is_file()
    assert (tmp_path / 'artifact' / 'manifest.json').is_file()
