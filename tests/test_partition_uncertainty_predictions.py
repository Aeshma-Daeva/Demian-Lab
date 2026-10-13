import importlib
import importlib.util
import hashlib

import numpy as np


def test_partition_preserves_every_array_row_and_original_hash(tmp_path):
    name = 'development.partition_uncertainty_predictions'
    assert importlib.util.find_spec(name) is not None, 'lossless publication partitioner missing'
    module = importlib.import_module(name)
    source = tmp_path/'predictions.npz'
    x = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.asarray([[7.], [8.], [9.]], dtype=np.float64)
    np.savez_compressed(source, x=x, y=y)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = module.partition_predictions(source, tmp_path/'parts', rows_per_part=2)
    assert manifest['source_sha256'] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert [(p['row_start'], p['row_stop']) for p in manifest['parts']] == [(0, 2), (2, 3)]
    for key, expected in [('x', x), ('y', y)]:
        restored = np.concatenate([np.load(tmp_path/'parts'/p['file'])[key] for p in manifest['parts']])
        assert restored.dtype == expected.dtype
        np.testing.assert_array_equal(restored, expected)
    assert manifest['reconstruction_verified']
