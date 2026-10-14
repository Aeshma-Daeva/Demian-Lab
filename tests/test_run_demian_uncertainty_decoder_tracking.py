import importlib
import importlib.util
import hashlib
import json
import shutil
from pathlib import Path

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


def test_pairing_rejects_missing_history_in_any_order_or_checkpoint():
    module = tracking_module()
    rows = [{"symbols": [0]}, {"symbols": [1]}]
    cells = {("train", "cue"): (np.zeros((2, 1)), rows),
             ("heldout", "cue"): (np.zeros((1, 1)), [{"symbols": [0]}])}
    try:
        module.validate_history_pairing(cells)
    except ValueError as error:
        assert "pairing changed" in str(error)
    else:
        raise AssertionError("mismatched paired histories accepted")


def test_heldout_state_and_label_changes_cannot_change_base_fits_or_predictions(tmp_path):
    """Exercise the frozen fitting boundary against adversarial heldout edits."""
    module = tracking_module()
    root = Path(__file__).resolve().parents[1]
    source = root / "data/diagnostics/demian_uncertainty_tracking_confirmation_matrix_v1_20261014"
    base_protocol = json.loads((root / "docs/DEMIAN_UNCERTAINTY_TRACKING_CONFIRMATION_PROTOCOL.json").read_text())

    def payload_for(name, *, change_states=False, change_labels=False):
        copied = tmp_path / name
        shutil.copytree(source, copied)
        states_path, metadata_path = copied / "states.npz", copied / "state_metadata.jsonl"
        states = np.load(states_path)["states"]
        rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
        heldout = [index for index, row in enumerate(rows) if row["order"] == "heldout"]
        if change_states:
            states[heldout] += 1234.0
            np.savez(states_path, states=states)
        if change_labels:
            for index in heldout:
                rows[index]["target_value"] += 999.0
            metadata_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest_path = copied / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for filename in ("states.npz", "state_metadata.jsonl"):
            manifest["file_sha256"][filename] = hashlib.sha256((copied / filename).read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        protocol = json.loads(json.dumps(base_protocol))
        protocol["batches"][0]["artifact_directory"] = str(copied)
        protocol["batches"][0]["source_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return module.run_protocol(protocol)

    baseline = payload_for("baseline")
    changed_states = payload_for("states", change_states=True)
    changed_labels = payload_for("labels", change_labels=True)

    def base_metrics(payload):
        return [row for row in payload["metrics"] if row["evaluation_order"] == "train"]

    for changed in (changed_states, changed_labels):
        assert baseline["transforms"].keys() == changed["transforms"].keys()
        for key in baseline["transforms"]:
            assert np.array_equal(baseline["transforms"][key], changed["transforms"][key])
        assert base_metrics(baseline) == base_metrics(changed)


def test_pinned_confirmation_replays_saved_metrics_predictions_and_fits():
    module = tracking_module()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/DEMIAN_UNCERTAINTY_TRACKING_CONFIRMATION_PROTOCOL.json").read_text())
    artifact = root / "data/diagnostics/demian_uncertainty_tracking_confirmation_v2_20261014"
    replay = module.run_protocol(protocol)
    saved_metrics = [json.loads(line) for line in (artifact / "metrics.jsonl").read_text().splitlines()]
    saved_predictions = np.load(artifact / "predictions.npz")["score_and_history_target_prediction"]
    saved_transforms = np.load(artifact / "transforms.npz")

    assert replay["metrics"] == saved_metrics
    assert np.array_equal(replay["predictions"], saved_predictions)
    assert set(replay["transforms"]) == set(saved_transforms.files)
    for key, value in replay["transforms"].items():
        assert np.array_equal(value, saved_transforms[key])


def test_channel_transport_fit_is_frozen_to_calibration_rows():
    from development.analyze_demian_uncertainty_transport_descriptors import (
        apply_channel_transport,
        fit_channel_transport,
    )

    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    target = (source - np.asarray([4.0, -3.0])) @ rotation.T / 3.0

    fit = fit_channel_transport(source, target)

    assert np.allclose(apply_channel_transport(fit, target), source, atol=1e-10)
    assert fit["fit_sample_count"] == 4
    assert np.linalg.det(fit["rotation"]) > 0.0


def test_channel_transport_calibration_is_independent_of_heldout_values():
    from development.analyze_demian_uncertainty_transport_descriptors import (
        apply_channel_transport,
        fit_channel_transport,
    )

    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    base_order_target = source + np.asarray([3.0, -2.0])
    heldout_target = source * 1000.0
    fit = fit_channel_transport(source, base_order_target)

    assert np.allclose(apply_channel_transport(fit, base_order_target), source, atol=1e-10)
    assert not np.allclose(apply_channel_transport(fit, heldout_target), source, atol=1e-10)
