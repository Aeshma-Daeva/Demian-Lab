"""Lossless row partitions for publication; original run artifact is untouched."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def validate_reconstruction(source: Path, parts_directory: Path) -> dict:
    """Verify canonical raw array bytes, including signed-zero representation."""
    manifest = json.loads((parts_directory/'partition_manifest.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise RuntimeError('original source hash mismatch')
    hashes = {key: hashlib.sha256() for key in manifest['arrays']}
    next_row = 0
    for part in manifest['parts']:
        path = parts_directory/part['file']
        if part['row_start'] != next_row or hashlib.sha256(path.read_bytes()).hexdigest() != part['sha256']:
            raise RuntimeError('partition ordering or file hash mismatch')
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(hashes):
                raise RuntimeError('partition array names mismatch')
            for key, digest in hashes.items():
                value = saved[key]
                schema = manifest['arrays'][key]
                if list(value.shape) != [part['row_stop']-part['row_start'], *schema['shape'][1:]] or str(value.dtype) != schema['dtype']:
                    raise RuntimeError('partition shape or dtype mismatch')
                digest.update(value.tobytes(order='C'))
        next_row = part['row_stop']
    if next_row != manifest['case_count']:
        raise RuntimeError('incomplete partition case coverage')
    with np.load(source, allow_pickle=False) as original:
        if set(original.files) != set(hashes) or any(hashes[key].hexdigest() != hashlib.sha256(original[key].tobytes(order='C')).hexdigest() for key in hashes):
            raise RuntimeError('reconstructed raw array byte hash mismatch')
    return {'original_sha256': manifest['source_sha256'], 'case_count': next_row,
            'array_sha256': {key: digest.hexdigest() for key, digest in hashes.items()},
            'canonical_array_bytes_verified': True}


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
