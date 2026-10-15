import numpy as np

from development.run_demian_final_slot_relation_predictor import _fit_lag_model, _lag_deficit


def test_frozen_lag_model_can_distinguish_a_changed_evaluation_relation() -> None:
    source = np.arange(12, dtype=float).reshape(2, 6, 1)
    target = 2.0 * source + 1.0
    fit = _fit_lag_model(source, target, np.asarray([0, 1]))

    matched = _lag_deficit(fit, source, target, np.asarray([0, 1]))
    changed = _lag_deficit(fit, source, 3.0 * target, np.asarray([0, 1]))

    assert changed > matched
