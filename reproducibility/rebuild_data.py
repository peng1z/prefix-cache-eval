"""Reconstruct staged inputs with verified upstream bytes."""
import argparse
import hashlib
import json
import os
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--storage', type=Path, required=True)
    ap.add_argument('--corpora', nargs='+', default=['oasst', 'sharegpt', 'openorca', 'wildchat'])
    ap.add_argument('--download', action='store_true', help='Allow upstream downloads, checked against recorded content hashes')
    args = ap.parse_args()
    report = json.loads((HERE / 'input_manifest.json').read_text())
    upstream = json.loads((HERE / 'source_hashes.json').read_text())
    storage = args.storage.resolve()
    storage.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(storage / '.cache' / 'huggingface')
    module = types.ModuleType('frozen_provision')
    module.__file__ = str(HERE / 'provision.py')
    module.storage_root = storage
    source = Path(module.__file__).read_text().replace(
        'PAY = Path(__file__).resolve().parent', 'PAY = Path(storage_root)')
    # Keep verified WildChat shards for reuse and inspection.
    source = source.replace('p.unlink()', 'None')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    expected = {}
    for corpus, entry in upstream.items():
        if corpus not in report:
            continue
        files = entry.get('shards', [entry])
        for item in files:
            name = item['file'] if corpus == 'wildchat' else report[corpus]['filename']
            expected[(corpus, name)] = item['sha256']
    def checked_download(repo, filename, dest_dir, repo_type='dataset'):
        corpus = Path(dest_dir).name
        path = Path(dest_dir) / filename
        if not path.is_file():
            if not args.download:
                raise FileNotFoundError(f'{path}: supply verified raw files or use --download')
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(repo_id=repo, filename=filename,
                        repo_type=repo_type, local_dir=str(dest_dir)))
        actual = digest(path)
        if actual != expected[(corpus, filename)]:
            raise ValueError(f'Upstream content changed: {path}; SHA-256 {actual}')
        return path
    module.dl = checked_download
    for corpus in args.corpora:
        if corpus not in report:
            raise ValueError(f'Unknown corpus {corpus}')
        getattr(module, 'ingest_' + corpus)()
        out = storage / 'derived' / (corpus + '.jsonl')
        actual = digest(out)
        if actual != report[corpus]['input_sha256']:
            raise ValueError(f'Reconstructed input differs: {out}; SHA-256 {actual}')
        print(corpus, 'verified', actual)

if __name__ == '__main__':
    main()
