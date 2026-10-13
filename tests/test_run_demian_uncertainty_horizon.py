import numpy as np
import torch

from development import run_demian_uncertainty_horizon as horizon
from development.run_demian_uncertainty_jacobian import make_transition


def test_horizon_transition_composes_and_repeated_calls_reset_runtime():
    args = dict(seed=94, hidden_size=16, dtype=torch.float64)
    torch.manual_seed(94)
    state = torch.zeros(84, dtype=torch.float64)
    twice = make_transition(**args, step_index=5, steps=2)
    next_twice = make_transition(**args, step_index=7, steps=2)
    four = make_transition(**args, step_index=5, steps=4)
    np.testing.assert_allclose(four(state).detach().numpy(), next_twice(twice(state)).detach().numpy(), atol=1e-12)
    np.testing.assert_array_equal(four(state).detach().numpy(), four(state).detach().numpy())


def test_source_artifact_integrity_rejects_changed_file(tmp_path):
    import hashlib
    import json
    (tmp_path / 'states.npz').write_bytes(b'changed')
    (tmp_path / 'manifest.json').write_text(json.dumps({'file_sha256': {'states.npz': hashlib.sha256(b'original').hexdigest()}}))
    import pytest
    with pytest.raises(ValueError, match='hash mismatch'):
        horizon.verify_source(tmp_path)


def test_cached_horizon_protocol_matches_anchor_and_preserves_horizon_cells(tmp_path):
    from development import run_demian_uncertainty_jacobian as jacobian
    source_protocol = {
        'protocol_id': 'test-source', 'claim_boundary': 'geometry only',
        'hidden_size': 16, 'model_seeds': [94], 'base_delay_steps': 2,
        'train_reliability_schedule': [.5, .75], 'heldout_reliability_schedule': [.75, .5],
        'allocation_seed': 31, 'split_seed': 41, 'subspace_rank': 1,
        'permutation_seed': 51, 'variants': ['base', 'extra_delay'],
    }
    source = tmp_path / 'source'
    jacobian.write_artifacts(source, source_protocol, jacobian.run_protocol(source_protocol))
    payload = horizon.run_protocol({'source_artifact_directory': str(source), 'horizons': [2, 4]})
    assert payload['integrity']['exact_two_step_anchor']
    assert payload['integrity']['all_finite']
    assert len(payload['metrics']) == 12
    assert payload['state_count'] == 12
