import numpy as np

from development.analyze_demian_cross_seed_geometry import paired_geometry_score


def test_paired_geometry_outscores_shuffled_history_pairing():
    points = np.asarray([[0., 0.], [1., 0.], [0., 2.], [3., 1.]])
    transformed = points @ np.asarray([[0., -2.], [2., 0.]]) + np.asarray([5., -1.])
    result = paired_geometry_score(points, transformed, shuffle_seed=11, shuffle_count=32)
    assert result["paired_distance_correlation"] > .999
    assert result["paired_minus_shuffled"] > .2
