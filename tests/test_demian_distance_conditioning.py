import numpy as np


def test_squared_distance_contrasts_equal_signed_anchor_inner_products():
    from development.run_demian_distance_conditioning import contrast
    rng = np.random.default_rng(9)
    x, a = rng.normal(size=(8, 7)), rng.normal(size=(4, 7))
    d = np.linalg.norm(x[:, None]-a, axis=2)
    expected = -2*x@(a-a.mean(0)).T + np.sum(a*a,axis=1)-np.mean(np.sum(a*a,axis=1))
    np.testing.assert_allclose(contrast(d), expected, atol=2e-14)


def test_whitening_is_fitted_on_training_rows_and_preserves_only_signal():
    from development.run_demian_distance_conditioning import fit_whitener
    x = np.array([[-2.,0,1],[-1,0,1],[1,0,1],[2,0,1]])
    mean, w = fit_whitener(x)
    z = (x-mean)@w
    np.testing.assert_allclose(np.mean(z*z,axis=0),[1,0,0],atol=1e-12)
    np.testing.assert_array_equal(mean,[0,0,1])
    np.testing.assert_array_equal(w[:,1:],np.zeros((3,2)))


def test_frozen_conditioned_probe_replays_without_evaluation_fitting(tmp_path):
    from development.run_demian_distance_conditioning import fit_variant, predict_variant
    rng = np.random.default_rng(15)
    train, evaluate = rng.normal(size=(24,16)), rng.normal(size=(8,16))
    frozen = fit_variant(train, rng.normal(size=24), True)
    snapshot = {key: value.copy() for key,value in frozen.items()}
    expected = predict_variant(frozen, evaluate)
    predict_variant(frozen, evaluate+100)
    for key in frozen:
        np.testing.assert_array_equal(frozen[key],snapshot[key])
    np.savez_compressed(tmp_path/'fit.npz',**frozen)
    with np.load(tmp_path/'fit.npz') as saved:
        np.testing.assert_array_equal(predict_variant(dict(saved),evaluate),expected)
