import importlib
import importlib.util
import hashlib
import json

import numpy as np


def tracking_module():
    name = "development.run_demian_uncertainty_decoder_tracking"
    assert importlib.util.find_spec(name) is not None, "decoder tracking runner missing"
    return importlib.import_module(name)


def test_orthogonal_tracker_recovers_train_only_translation_scale_and_rotation():
    module = tracking_module()
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    target = (source - np.asarray([4.0, -3.0])) @ rotation.T / 3.0

    tracker = module.fit_tracker(source, target, mode="orthogonal_track")

    assert np.allclose(module.apply_tracker(tracker, target), source, atol=1e-10)
    assert tracker["fit_sample_count"] == 4
    assert np.linalg.det(tracker["rotation"]) > 0.0


def test_scale_tracker_does_not_fit_orientation():
    module = tracking_module()
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T

    tracker = module.fit_tracker(source, target, mode="scale_track")

    assert not np.allclose(module.apply_tracker(tracker, target), source)


def test_dynamic_feature_builders_have_declared_small_capacity():
    module = tracking_module()
    projected = np.arange(12, dtype=float).reshape(4, 3)
    checkpoints = ["cue", "delay"]
    rows = [{"checkpoint_id": "cue", "gate_mean": 1.0, "state_norm": 2.0},
            {"checkpoint_id": "delay", "gate_mean": 2.0, "state_norm": 3.0},
            {"checkpoint_id": "cue", "gate_mean": 3.0, "state_norm": 4.0},
            {"checkpoint_id": "delay", "gate_mean": 4.0, "state_norm": 5.0}]

    assert module.time_features(projected, rows, checkpoints).shape == (4, 6)
    assert module.regime_features(projected, rows).shape == (4, 9)


def test_matrix_loader_rejects_nonfinite_states_and_duplicate_histories(tmp_path):
    module = tracking_module()
    artifact = tmp_path / "matrix"
    artifact.mkdir()
    np.savez(artifact / "states.npz", states=np.asarray([[np.nan, 0.0]]))
    (artifact / "state_metadata.jsonl").write_text(json.dumps({"symbols": [0]}) + "\n")
    (artifact / "summary.json").write_text(json.dumps({"integrity": {"history_split": {}}}))
    hashes = {name: hashlib.sha256((artifact / name).read_bytes()).hexdigest()
              for name in ("states.npz", "state_metadata.jsonl", "summary.json")}
    (artifact / "manifest.json").write_text(json.dumps({"file_sha256": hashes}))
    batch = {"artifact_directory": str(artifact),
             "source_manifest_sha256": hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()}

    try:
        module._load_batch(batch)
    except ValueError as error:
        assert "nonfinite" in str(error)
    else:
        raise AssertionError("nonfinite source state accepted")

    states = np.asarray([[0.0], [1.0]])
    rows = [{"model_seed": 1, "order": "train", "checkpoint_id": "cue", "symbols": [0]},
            {"model_seed": 1, "order": "train", "checkpoint_id": "cue", "symbols": [0]}]
    try:
        module._cell(states, rows, 1, "train", "cue")
    except ValueError as error:
        assert "duplicate history" in str(error)
    else:
        raise AssertionError("duplicate history accepted")
