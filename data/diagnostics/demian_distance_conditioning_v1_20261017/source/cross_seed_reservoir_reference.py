"""Frozen tanh reservoir reference for matched public-history collection.

The default state width matches Demian's concatenated full state, not its
parameter count, dynamics, or channel architecture. No fitting occurs here.
"""

from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np
import torch

from development.demian_uncertainty_probe import encode_evidence
from development.demian_uncertainty_world import EvidenceObservation


_CUE_COUNT = 6
_INPUT_SCALE = 0.25
_SPECTRAL_RADIUS = 0.9


def _array_hash(value: np.ndarray) -> str:
    """SHA256 of C-order array bytes; shape/dtype are declared separately."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate_inputs(seed, histories, reliabilities, delays, state_width, input_width):
    for name, value, minimum in (("seed", seed, 0), ("state_width", state_width, 10),
                                 ("input_width", input_width, 10)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if input_width > state_width:
        raise ValueError("state_width must be >= input_width for an orthonormal input projection")
    values = np.asarray(histories)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != _CUE_COUNT:
        raise ValueError("histories must have shape (N, 6) with N >= 1")
    if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
        raise ValueError("histories must contain finite binary values")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("histories must contain binary values")
    values = values.astype(np.uint8)
    history_ids = values @ (2 ** np.arange(_CUE_COUNT - 1, -1, -1))
    if np.any(np.diff(history_ids) <= 0):
        raise ValueError("histories must be unique and in lexicographic order")
    reliability_values = np.asarray(reliabilities, dtype=np.float64)
    if (reliability_values.shape != (_CUE_COUNT,) or not np.isfinite(reliability_values).all()
            or not np.isin(reliability_values, (0.5, 0.75, 0.95)).all()):
        raise ValueError("reliabilities must contain six values from 0.5, 0.75, 0.95")
    delay_values = list(delays)
    if (not delay_values or any(isinstance(delay, (bool, np.bool_))
                                or not isinstance(delay, Integral) or delay < 0
                                for delay in delay_values)):
        raise ValueError("delays must be nonnegative integers, including zero")
    if delay_values[0] != 0 or any(left >= right for left, right in zip(delay_values, delay_values[1:])):
        raise ValueError("delays must be unique, increasing, and begin with zero")
    return values, reliability_values.tolist(), [int(delay) for delay in delay_values]


def _spectral_radius(weights: torch.Tensor) -> float:
    """Full eigenspectrum in float64, not a power-iteration estimate."""
    radius = float(torch.linalg.eigvals(weights.to(dtype=torch.float64)).abs().max())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("recurrent spectral radius must be finite and positive")
    return radius


@torch.no_grad()
def collect_reservoir(
    seed: int,
    histories: np.ndarray,
    reliabilities: list[float],
    delays: list[int],
    state_width: int = 1312,
    input_width: int = 256,
) -> tuple[np.ndarray, dict]:
    """Collect states after six cues and each declared number of delay ticks.

    ``histories`` contains unique lexicographically ordered binary rows.
    ``delays`` is increasing, unique, and starts at zero (the post-cue6 state).
    Delay ticks use the actual encoder's delay-phase input, not a zero vector.

    A local NumPy ``default_rng(seed)`` draws W then P. W starts Gaussian with
    scale 1/sqrt(state_width), is rescaled using its full float64 eigenspectrum,
    and is rounded to float32 for recurrence. P is the reduced float64 QR of a
    Gaussian draw, with positive diagonal R, then rounded to float32. Both are
    frozen. The update is h <- tanh(W h + P u + b), with zero b and h initially.

    Returns float32 states of shape (N, len(delays), state_width), plus metadata.
    The metadata contains NumPy arrays at ``weights``, ``input_projection``,
    and ``bias`` for separate preservation; all remaining values are JSON-safe.
    Computation uses CPU with one Torch thread, restoring the caller's setting.
    """
    histories, reliabilities, delays = _validate_inputs(
        seed, histories, reliabilities, delays, state_width, input_width,
    )
    seed, state_width, input_width = int(seed), int(state_width), int(input_width)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(seed)
        raw_weights = torch.from_numpy(rng.standard_normal((state_width, state_width)) / np.sqrt(state_width))
        raw_radius = _spectral_radius(raw_weights)
        weights = (raw_weights * (_SPECTRAL_RADIUS / raw_radius)).to(dtype=torch.float32)
        actual_radius = _spectral_radius(weights)
        raw_projection = torch.from_numpy(rng.standard_normal((state_width, input_width)))
        projection, upper = torch.linalg.qr(raw_projection, mode="reduced")
        signs = torch.where(torch.diag(upper) < 0, -1.0, 1.0)
        projection = (projection * signs).to(dtype=torch.float32)
        bias = torch.zeros(state_width, dtype=torch.float32, device="cpu")
        if not all(torch.isfinite(array).all() for array in (weights, projection, bias)):
            raise ValueError("nonfinite reservoir parameters")

        state = torch.zeros((len(histories), state_width), dtype=torch.float32, device="cpu")
        states = np.empty((len(histories), len(delays), state_width), dtype=np.float32)

        def step(inputs: torch.Tensor) -> None:
            nonlocal state
            if not torch.isfinite(inputs).all():
                raise ValueError("nonfinite encoded public input")
            state = torch.tanh(state @ weights.T + inputs @ projection.T + bias)
            if not torch.isfinite(state).all():
                raise ValueError("nonfinite reservoir state")

        for cue_index, reliability in enumerate(reliabilities):
            inputs = torch.cat([
                encode_evidence(EvidenceObservation(cue_index, "evidence", int(symbol), reliability),
                                input_width, scale=_INPUT_SCALE)
                for symbol in histories[:, cue_index]
            ])
            step(inputs)
        states[:, 0] = state.numpy()
        checkpoint_index = 1
        delay_input = encode_evidence(EvidenceObservation(_CUE_COUNT, "delay", None, None),
                                     input_width, scale=_INPUT_SCALE).expand(len(histories), -1)
        for delay in range(1, delays[-1] + 1):
            step(delay_input)
            if delay == delays[checkpoint_index]:
                states[:, checkpoint_index] = state.numpy()
                checkpoint_index += 1

        arrays = {"weights": weights.numpy().copy(), "input_projection": projection.numpy().copy(),
                  "bias": bias.numpy().copy()}
        metadata = {
            "schema_version": 1,
            "model": "frozen_tanh_echo_state_reference",
            "seed": seed,
            "rng": "numpy.random.default_rng(seed); W draw followed by P draw; independent of Torch RNG",
            "state_width": state_width,
            "input_width": input_width,
            "state_shape": list(states.shape),
            "history_count": len(histories),
            "cue_count": _CUE_COUNT,
            "reliabilities": reliabilities,
            "delays": delays,
            "input_encoder": "development.demian_uncertainty_probe.encode_evidence",
            "input_scale": _INPUT_SCALE,
            "delay_input": "encoded delay phase at every delay tick",
            "input_projection_method": "Gaussian reduced QR; positive R diagonal; orthonormal columns before float32 rounding",
            "recurrence": "h_next = tanh(W @ h + P @ u + b)",
            "leak": 1.0,
            "initial_state": "zeros",
            "bias_initialization": "zeros; frozen",
            "recurrent_initialization": "Gaussian N(0, 1 / state_width), rescaled by full float64 eigenspectrum",
            "spectral_radius_target": _SPECTRAL_RADIUS,
            "spectral_radius_raw": raw_radius,
            "spectral_radius": actual_radius,
            "spectral_radius_method": "torch.linalg.eigvals; all eigenvalues; float64; CPU",
            "parameter_count": state_width * state_width + state_width * input_width + state_width,
            "parameter_count_includes": "W, P, and fixed zero bias",
            "trainable_parameter_count": 0,
            "learning": False,
            "device": "cpu",
            "dtype": "float32",
            "initialization_linear_algebra_dtype": "float64",
            "torch_threads": 1,
            "numpy_version": np.__version__,
            "torch_version": str(torch.__version__),
            "hash_algorithm": "sha256; array C-order bytes; shapes/dtypes declared separately",
            "parameter_hashes": {name: _array_hash(array) for name, array in arrays.items()},
            "parameter_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "parameter_dtype": "float32",
            "states_sha256": _array_hash(states),
            "histories_sha256": _array_hash(histories),
            "histories_dtype": "uint8",
            "mismatch_notes": [
                "Default 1312 state coordinates and 256 input coordinates match Demian full-state and encoder widths only.",
                "Parameter count, channel structure, initialization, input injection, and dynamical timescales are not matched.",
                "The reservoir mixes the input through P; Demian adds the encoded input to its fast channel before its step.",
                "A shared seed identifier pairs runs but does not couple the two architectures' random parameters.",
                "Spectral radius 0.9 is a declared setting, not a demonstrated echo-state-property guarantee.",
                "This frozen reference alone cannot establish architectural superiority or a causal mechanism.",
            ],
            **arrays,
        }
        return states, metadata
    finally:
        torch.set_num_threads(previous_threads)
