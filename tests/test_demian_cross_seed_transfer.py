import numpy as np
import pytest

from development.demian_cross_seed_transfer import anchor_features, make_split, fit_readouts, predict_readouts


def test_anchor_calibration_is_rigid_motion_and_scale_invariant():
    x = np.random.default_rng(7).normal(size=(64, 20))
    anchors = make_split(2026101607)["anchors"]
    q = np.linalg.qr(np.random.default_rng(9).normal(size=(20, 20)))[0]
    first, _ = anchor_features(x, anchors)
    second, _ = anchor_features(3 * x @ q + 5, anchors)
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_test_rows_and_labels_cannot_change_fit_or_anchor_scale():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 64, 20))
    y = rng.normal(size=64)
    split = make_split(2026101607)
    fit, _ = fit_readouts(x, y, split, projection_seed=431, shuffle_seed=801)
    changed = x.copy()
    changed[:, split["evaluation"]] += 100
    changed_y = y.copy()
    changed_y[split["evaluation"]] += 99
    other, _ = fit_readouts(changed, changed_y, split, projection_seed=431, shuffle_seed=801)
    for key in fit:
        np.testing.assert_array_equal(fit[key], other[key])
    result = predict_readouts(fit, x[0], split)
    assert result["anchor"].shape == (64,)


def test_split_is_disjoint_exhaustive_balanced_and_full_rank():
    s = make_split(2026101607)
    assert [len(s[k]) for k in ("anchors", "fit", "evaluation")] == [16, 24, 24]
    assert len(set(sum(s.values(), []))) == 64
    h = np.array([list(map(int, f"{i:06b}")) for i in range(64)]) * 2 - 1
    assert np.linalg.matrix_rank(np.c_[h[s["anchors"]], np.ones(16)]) == 7
    assert np.all(h[s["fit"]].sum(axis=0) == 0)


def test_nonfinite_and_collapsed_inputs_are_explicit():
    x = np.zeros((64, 20))
    a = make_split(1)["anchors"]
    features, diagnostics = anchor_features(x, a)
    assert diagnostics["precision_limited"]
    assert not features.any()
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        anchor_features(x, a)


def test_batched_collection_matches_native_serial_and_same_prefix():
    from development.run_demian_cross_seed_transfer import collect_demian
    from development.demian_uncertainty_probe import run_uncertainty_episode
    from development.demian_uncertainty_world import AmbiguousEvidenceWorld
    import torch
    histories = np.array([[0]*6, [1]*6])
    order = [.95, .75, .5, .75, .95, .5]
    states, info = collect_demian(31, histories, order, [0, 2, 8], 16)
    for i, history in enumerate(histories):
        frames = AmbiguousEvidenceWorld(hidden_target=0, evidence=list(zip(history, order)), delay_steps=8).frames()[:14]
        reference = run_uncertainty_episode(model_seed=31, hidden_size=16, public_frames=frames, checkpoint_phases=("evidence", "delay"))
        for ti, key in enumerate(("evidence:5", "delay:7", "delay:13")):
            expected = torch.cat(reference["checkpoints"][key].state, dim=1).numpy()[0]
            np.testing.assert_allclose(states[i, ti], expected, rtol=2e-6, atol=2e-6)
    short, _ = collect_demian(31, histories, order, [0, 2], 16)
    np.testing.assert_array_equal(short, states[:, :2])
    assert info["frozen_weights_verified"]


def test_saved_frozen_readout_reproduces_predictions(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 64, 20))
    s = make_split(2026101607)
    frozen, _ = fit_readouts(x, rng.normal(size=64), s, projection_seed=431, shuffle_seed=801)
    expected = predict_readouts(frozen, x[0], s)
    np.savez_compressed(tmp_path/"fit.npz", **frozen)
    with np.load(tmp_path/"fit.npz") as archive:
        actual = predict_readouts(dict(archive), x[0], s)
    for key in actual:
        np.testing.assert_array_equal(expected[key], actual[key])


def test_roundoff_residual_has_no_reportable_geometry():
    from development.demian_cross_seed_transfer import shape_diagnostics, guarded_geometry
    split = make_split(2026101607)
    x = np.zeros((64, 20))
    x[:, 1] = .25
    shape, residual = shape_diagnostics(x, split)
    assert shape["precision_limited"]
    assert shape["pc1_fraction"] is None
    measured = guarded_geometry(residual, residual, x, x)
    assert measured["precision_limited"]
    assert measured["cka"] is None
