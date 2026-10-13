"""Lossless row partitions for publication; original run artifact is untouched."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def partition_predictions(source: Path, out: Path, *, rows_per_part: int = 1024) -> dict:
    if rows_per_part <= 0:
        raise ValueError('rows_per_part must be positive')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    lengths = {value.shape[0] for value in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError('nonempty arrays must share case axis')
    count = next(iter(lengths))
    out.mkdir(parents=True, exist_ok=False)
    parts = []
    for start in range(0, count, rows_per_part):
        stop = min(start+rows_per_part, count)
        path = out/f'predictions.part-{len(parts):03d}.npz'
        np.savez_compressed(path, **{key: value[start:stop] for key, value in arrays.items()})
        with np.load(path, allow_pickle=False) as restored:
            if any(restored[key].dtype != value.dtype or not np.array_equal(restored[key], value[start:stop]) for key, value in arrays.items()):
                raise RuntimeError('partition reconstruction mismatch')
        parts.append({'file': path.name, 'row_start': start, 'row_stop': stop,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size})
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('original changed during partition')
    manifest = {
        'source_filename': source.name, 'source_sha256': source_hash,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'case_count': count, 'rows_per_part': rows_per_part,
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in arrays.items()},
        'parts': parts, 'reconstruction_verified': True,
        'reconstruction': 'verify part hashes; concatenate each named array in listed row order; NPZ container bytes need not match original',
    }
    with (out/'partition_manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--rows-per-part', type=int, default=1024)
    args = parser.parse_args()
    result = partition_predictions(args.source, args.out, rows_per_part=args.rows_per_part)
    print(json.dumps({'cases': result['case_count'], 'parts': len(result['parts']), 'verified': result['reconstruction_verified']}))


if __name__ == '__main__':
    main()
