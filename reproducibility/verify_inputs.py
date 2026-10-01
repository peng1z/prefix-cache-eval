"""Verify staged inputs before running the benchmark."""
import argparse
import json
from pathlib import Path
from rebuild_data import digest, HERE

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data_dir', type=Path)
    args = ap.parse_args()
    for corpus, row in json.loads((HERE / 'input_manifest.json').read_text()).items():
        path = args.data_dir / (corpus + '.jsonl')
        if digest(path) != row['input_sha256']:
            raise SystemExit(f'Input hash mismatch: {path}')
        print(corpus, 'verified')
