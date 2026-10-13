import importlib
import importlib.util
import hashlib
import json

import numpy as np


def representation_module():
    name = "development.analyze_demian_uncertainty_representation"
    assert importlib.util.find_spec(name) is not None, "representation analysis runner missing"
    return importlib.import_module(name)


def test_geometry_decomposition_separates_translation_and_rotation():
    module = representation_module()
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    target = source @ rotation + np.asarray([4.0, -3.0])

    result = module.geometry_decomposition(source, target, rank=2)

    assert result["centroid_shift"] > 0.0
    assert abs(result["scale_ratio"] - 1.0) < 1e-12
    assert result["procrustes_residual"] < 1e-12
    assert abs(result["subspace_overlap"] - 1.0) < 1e-12


def test_procrustes_residual_is_invariant_to_uniform_scale():
    module = representation_module()
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    deformed = source + np.asarray([[0.0, 0.0], [0.2, 0.0], [0.0, 0.0]])
    target = 11.0 * deformed + np.asarray([4.0, -3.0])

    baseline = module.geometry_decomposition(source, deformed, rank=2)["procrustes_residual"]
    scaled = module.geometry_decomposition(source, target, rank=2)["procrustes_residual"]
    assert abs(baseline - scaled) < 1e-12


def test_pairwise_geometry_alignment_is_one_for_equal_distance_patterns():
    module = representation_module()
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    target = 3.0 * source + np.asarray([5.0, 7.0])

    assert abs(module.pairwise_geometry_alignment(source, target) - 1.0) < 1e-12


def test_change_norm_relation_uses_history_wise_channel_changes():
    module = representation_module()
    previous = {
        "fast": np.asarray([[0.0], [0.0], [0.0]]),
        "gate": np.asarray([[0.0], [0.0], [0.0]]),
    }
    current = {
        "fast": np.asarray([[1.0], [2.0], [3.0]]),
        "gate": np.asarray([[2.0], [4.0], [6.0]]),
    }

    result = module.change_norm_relation(previous, current, "fast", "gate")

    assert abs(result["pearson_r"] - 1.0) < 1e-12
    assert result["sample_count"] == 3


def test_protocol_reports_within_time_and_cross_order_metrics(tmp_path):
    module = representation_module()
    artifact = tmp_path / "matrix"
    artifact.mkdir()
    rows, states = [], []
    for checkpoint, offset in (("cue", 0.0), ("delay", 1.0)):
        for order, order_offset in (("train", 0.0), ("heldout", 2.0)):
            for history in range(4):
                states.append(np.asarray([history + offset + order_offset, history, 0.0]))
                rows.append({"model_seed": 1, "order": order, "checkpoint_id": checkpoint,
                             "symbols": [history], "channel_widths": [1, 1, 1, 0, 0, 0]})
    np.savez(artifact / "states.npz", states=np.asarray(states))
    (artifact / "state_metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    source_hashes = {name: hashlib.sha256((artifact / name).read_bytes()).hexdigest()
                     for name in ("states.npz", "state_metadata.jsonl")}
    (artifact / "manifest.json").write_text(json.dumps({"file_sha256": source_hashes}) + "\n")
    manifest_hash = hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()
    payload = module.run_protocol({"subspace_rank": 2, "batches": [{"id": "test", "artifact_directory": str(artifact),
        "source_manifest_sha256": manifest_hash, "model_seeds": [1], "checkpoints": [{"id": "cue"}, {"id": "delay"}]}]})

    assert payload["integrity"]["analysis_only"]
    assert len(payload["metrics"]) == 4
    assert {row["comparison"] for row in payload["metrics"]} == {"within_order_time", "cross_order"}
    assert len(payload["relation_metrics"]) == 10

    module.write_artifacts(tmp_path / "out", {"protocol_id": "test", "claim_boundary": "analysis"}, payload)
    assert (tmp_path / "out" / "metrics.jsonl").is_file()
    assert (tmp_path / "out" / "relation_metrics.jsonl").is_file()


def test_protocol_rejects_unpinned_source_artifact(tmp_path):
    module = representation_module()
    artifact = tmp_path / "matrix"
    artifact.mkdir()
    np.savez(artifact / "states.npz", states=np.zeros((1, 1)))
    (artifact / "state_metadata.jsonl").write_text("")

    try:
        module.run_protocol({"subspace_rank": 1, "batches": [{"id": "test", "artifact_directory": str(artifact),
            "model_seeds": [], "checkpoints": [{"id": "cue"}]}]})
    except ValueError as error:
        assert "source manifest" in str(error)
    else:
        raise AssertionError("expected unpinned source artifact rejection")


def test_protocol_rejects_source_file_not_matching_manifest(tmp_path):
    module = representation_module()
    artifact = tmp_path / "matrix"
    artifact.mkdir()
    np.savez(artifact / "states.npz", states=np.zeros((1, 1)))
    (artifact / "state_metadata.jsonl").write_text("")
    (artifact / "manifest.json").write_text(json.dumps({"file_sha256": {
        "states.npz": "incorrect", "state_metadata.jsonl": hashlib.sha256(b"").hexdigest()}}))
    manifest_hash = hashlib.sha256((artifact / "manifest.json").read_bytes()).hexdigest()

    try:
        module.run_protocol({"subspace_rank": 1, "batches": [{"id": "test", "artifact_directory": str(artifact),
            "source_manifest_sha256": manifest_hash, "model_seeds": [], "checkpoints": [{"id": "cue"}]}]})
    except ValueError as error:
        assert "source file SHA256 mismatch" in str(error)
    else:
        raise AssertionError("expected source file checksum rejection")
