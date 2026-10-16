import hashlib
import json

import numpy as np
import pytest
import torch

from development.cross_seed_reservoir_reference import collect_reservoir
from development.demian_uncertainty_probe import encode_evidence
from development.demian_uncertainty_world import EvidenceObservation


HISTORIES = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1]], dtype=np.uint8)
RELIABILITIES = [0.5, 0.75, 0.95, 0.75, 0.5, 0.95]


def collect(**overrides):
    arguments = dict(seed=11, histories=HISTORIES, reliabilities=RELIABILITIES,
                     delays=[0, 2, 5], state_width=16, input_width=10)
    arguments.update(overrides)
    return collect_reservoir(**arguments)


def test_reproducible_frozen_model_and_declared_arrays():
    rng_before = torch.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    left, left_meta = collect()
    right, right_meta = collect()
    np.testing.assert_array_equal(left, right)
    assert left.shape == (3, 3, 16)
    assert left.dtype == np.float32
    assert np.isfinite(left).all()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.get_num_threads() == threads_before
    for name in ("weights", "input_projection", "bias"):
        np.testing.assert_array_equal(left_meta[name], right_meta[name])
        assert left_meta[name].dtype == np.float32
        assert left_meta["parameter_hashes"][name] == hashlib.sha256(left_meta[name].tobytes()).hexdigest()
    assert left_meta["parameter_hashes"] == right_meta["parameter_hashes"]
    assert left_meta["parameter_count"] == 16 * 16 + 16 * 10 + 16
    assert left_meta["trainable_parameter_count"] == 0
    assert left_meta["learning"] is False
    assert not np.any(left_meta["bias"])
    radius = np.abs(np.linalg.eigvals(left_meta["weights"].astype(np.float64))).max()
    assert radius == pytest.approx(0.9, abs=1e-7)
    assert left_meta["spectral_radius"] == pytest.approx(radius, abs=1e-12)
    projection = left_meta["input_projection"].astype(np.float64)
    np.testing.assert_allclose(projection.T @ projection, np.eye(10), atol=1e-7)
    assert left_meta["states_sha256"] == hashlib.sha256(left.tobytes()).hexdigest()
    json.dumps({key: value for key, value in left_meta.items()
                if key not in {"weights", "input_projection", "bias"}}, allow_nan=False)
    changed, changed_meta = collect(seed=12)
    assert changed_meta["parameter_hashes"] != left_meta["parameter_hashes"]
    assert not np.array_equal(changed, left)


def test_shared_checkpoints_are_exact_when_horizon_is_extended():
    short, short_meta = collect(delays=[0, 2])
    long, long_meta = collect(delays=[0, 1, 2, 4, 8])
    np.testing.assert_array_equal(short, long[:, [0, 2]])
    assert short_meta["parameter_hashes"] == long_meta["parameter_hashes"]
    cue_only, _ = collect(delays=[0])
    np.testing.assert_array_equal(cue_only[:, 0], short[:, 0])


def test_saved_weights_replay_actual_encoder_and_delay_ticks():
    states, metadata = collect(state_width=32, input_width=16)
    weights = metadata["weights"]
    projection = metadata["input_projection"]
    for row, history in enumerate(HISTORIES):
        state = np.zeros(32, dtype=np.float32)
        checkpoints = []
        for tick, (symbol, reliability) in enumerate(zip(history, RELIABILITIES)):
            encoded = encode_evidence(EvidenceObservation(tick, "evidence", int(symbol), reliability),
                                      16, scale=0.25).numpy()[0]
            state = np.tanh(weights @ state + projection @ encoded)
        checkpoints.append(state.copy())
        for delay in range(1, 6):
            encoded = encode_evidence(EvidenceObservation(5 + delay, "delay", None, None),
                                      16, scale=0.25).numpy()[0]
            state = np.tanh(weights @ state + projection @ encoded)
            if delay in (2, 5):
                checkpoints.append(state.copy())
        np.testing.assert_allclose(states[row], checkpoints, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("delays", [[], [1], [-1, 0], [0, 0], [0, 2, 1], [0, 1.5], [0, np.nan], [False, 1]])
def test_invalid_delays_are_rejected(delays):
    with pytest.raises(ValueError, match="delays"):
        collect(delays=delays)


@pytest.mark.parametrize("histories", [np.zeros((2, 5)), np.empty((0, 6)), HISTORIES[::-1],
                                       HISTORIES[[0, 0]], np.full((1, 6), 2), np.full((1, 6), np.nan),
                                       np.full((1, 6), np.inf)])
def test_invalid_histories_are_rejected(histories):
    with pytest.raises(ValueError, match="histories"):
        collect(histories=histories)


@pytest.mark.parametrize("reliabilities", [[0.75] * 5, [0.75] * 5 + [np.nan],
                                           [0.75] * 5 + [np.inf], [0.75] * 5 + [0.6]])
def test_invalid_reliabilities_are_rejected(reliabilities):
    with pytest.raises(ValueError, match="reliabilities"):
        collect(reliabilities=reliabilities)


@pytest.mark.parametrize("overrides", [{"input_width": 9}, {"state_width": 9}, {"input_width": 17},
                                       {"seed": -1}, {"seed": True}, {"state_width": 16.0}])
def test_invalid_dimensions_and_seed_are_rejected(overrides):
    with pytest.raises(ValueError):
        collect(**overrides)


def test_nonfinite_encoder_output_is_rejected_and_thread_setting_is_restored(monkeypatch):
    monkeypatch.setattr("development.cross_seed_reservoir_reference.encode_evidence",
                        lambda *args, **kwargs: torch.full((1, 10), float("nan")))
    threads_before = torch.get_num_threads()
    with pytest.raises(ValueError, match="nonfinite encoded public input"):
        collect()
    assert torch.get_num_threads() == threads_before
