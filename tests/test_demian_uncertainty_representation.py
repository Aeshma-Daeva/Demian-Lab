import numpy as np
import pytest

from development import demian_uncertainty_representation


def test_sequence_split_keeps_entire_episode_ids_together() -> None:
    rows = [{"episode_id": value} for value in range(10)]
    train, heldout = demian_uncertainty_representation.split_episode_rows(rows, heldout_fraction=0.3)

    assert {row["episode_id"] for row in train}.isdisjoint({row["episode_id"] for row in heldout})
    assert len(heldout) == 3


def test_ridge_probe_beats_constant_on_linearly_encoded_heldout_signal() -> None:
    x_train = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.asarray([0.0, 1.0, 2.0, 3.0])
    x_eval = np.asarray([[4.0], [5.0]])
    y_eval = np.asarray([4.0, 5.0])

    result = demian_uncertainty_representation.evaluate_ridge_probe(x_train, y_train, x_eval, y_eval)

    assert result["r2"] > 0.99
    assert result["constant_r2"] < 0.0
    assert result["pearson_r"] > 0.99
    assert result["normalized_mae"] < 0.01


def test_ridge_probe_rejects_nonfinite_features() -> None:
    with pytest.raises(ValueError, match="finite"):
        demian_uncertainty_representation.evaluate_ridge_probe(
            np.asarray([[0.0], [np.nan]]),
            np.asarray([0.0, 1.0]),
            np.asarray([[2.0]]),
            np.asarray([2.0]),
        )


def test_fixed_random_projection_is_deterministic_and_has_requested_width() -> None:
    features = np.arange(24, dtype=np.float64).reshape(4, 6)

    first = demian_uncertainty_representation.fixed_random_projection(features, output_dim=3, seed=17)
    second = demian_uncertainty_representation.fixed_random_projection(features, output_dim=3, seed=17)

    assert first.shape == (4, 3)
    assert np.allclose(first, second)
    assert not np.allclose(first, demian_uncertainty_representation.fixed_random_projection(features, output_dim=3, seed=29))
