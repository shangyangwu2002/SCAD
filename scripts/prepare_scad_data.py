"""Prepare and validate runtime inputs from a read-only SCAD_DATA package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ('train', 'val', 'test')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def safe_image(package, relative):
    path = (package / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(package) or not path.is_file():
        raise ValueError(f'Image is missing or escapes the package: {relative}')
    return path


def inspect_tables(package):
    if (package / 'dataset/ood_test.parquet').exists():
        raise ValueError('Legacy split layout found. Rebuild the benchmark tables with download_data.sh '
                         'before preparing the unified test set; use a fresh SCAD_RUNTIME_ROOT.')
    result = {}; hashes = {}
    frozen = package / 'corpus/image_search_results.json'
    cache = json.loads(frozen.read_text()) if frozen.is_file() else None
    for split in SPLITS:
        rows = pq.read_table(package / 'dataset' / f'{split}.parquet').to_pylist()
        ids = [r['extra_info']['stable_hash'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate examples in {split}')
        hashes[split] = set(ids)
        for row in rows:
            if 'images' in row and split in {'train', 'val'} and cache is None:
                raise ValueError('Missing frozen image-search cache for multimodal training/validation.')
            for image in row.get('images') or []:
                safe_image(package, image['path'])
            if cache is not None and split in {'train', 'val'}:
                extra = row['extra_info']
                if extra['retrieval_mode'] != 'frozen_corpus':
                    raise ValueError(f'{split} must use frozen retrieval')
                if any(key not in cache for key in extra['image_paths']):
                    raise ValueError(f'Missing frozen image-search entry in {split}')
        result[split] = len(rows)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if hashes[a] & hashes[b]:
                raise ValueError(f'Split overlap: {a}, {b}')
    return result


def prepare(package, output, image_base_url=''):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.is_relative_to(package.parent):
        raise ValueError('Runtime output must be outside SCAD_DATA.')
    if image_base_url and not image_base_url.startswith(('https://', 'http://')):
        raise ValueError('Image base URL must be HTTP(S).')
    counts = inspect_tables(package)
    inputs = [package / 'dataset' / f'{split}.parquet' for split in SPLITS]
    mapping_path = package / 'dataset/image_url_map.json'
    if mapping_path.is_file():
        inputs.append(mapping_path)
    signature = {'package_root': str(package), 'input_sha256': {p.name: digest(p) for p in inputs},
                 'image_base_url': image_base_url}
    state_path = output / 'prepare_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['signature'] != signature:
            raise ValueError('Runtime inputs changed; use a new SCAD_RUNTIME_ROOT.')
        for relative, expected in state['outputs'].items():
            if not (output / relative).is_file() or digest(output / relative) != expected:
                raise ValueError(f'Runtime output changed or incomplete: {relative}')
        return counts
    destination = output / 'dataset'
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Unrecognized runtime dataset; choose a new output directory.')
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for split in SPLITS:
        table = pq.read_table(package / 'dataset' / f'{split}.parquet')
        rows = table.to_pylist()
        for row in rows:
            for image in row.get('images') or []:
                image['path'] = str(safe_image(package, image['path']))
        path = destination / f'{split}.parquet'
        temporary = path.with_suffix('.parquet.tmp')
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
        temporary.replace(path)
        written.append(path)
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        for key, value in mapping.items():
            if value.startswith('scad-image://'):
                filename = value.removeprefix('scad-image://')
                safe_image(package, 'dataset/public_images/' + filename)
                if image_base_url:
                    mapping[key] = image_base_url.rstrip('/') + '/' + filename
        path = destination / 'image_url_map.json'
        atomic_json(path, mapping)
        written.append(path)
    atomic_json(state_path, {'signature': signature, 'splits': counts,
                            'outputs': {str(p.relative_to(output)): digest(p) for p in written}})
    return counts


def check_retrieval(package, embedding_root, ann_root):
    from envs.search_server import load_shard_specs
    package, embedding_root, ann_root = map(Path, (package, embedding_root, ann_root))
    specification = json.loads((package / 'corpus/EMBEDDING_SPEC.json').read_text())
    manifest = json.loads((embedding_root / 'manifest.json').read_text())
    expected = specification['rows_by_source']
    specs = load_shard_specs(embedding_root, package / 'corpus/passages', tuple(expected))
    actual = {source: sum(s.rows for s in specs if s.source == source) for source in expected}
    if actual != expected or manifest['dimension'] != specification['dimension']:
        raise ValueError('Embedding corpus or dimension does not match this data package.')
    index = json.loads((ann_root / 'manifest.json').read_text())
    if index.get('embedding_manifest_sha256') != digest(embedding_root / 'manifest.json'):
        raise ValueError('ANN index is not bound to these embeddings; rebuild it with prepare_embeddings.sh.')
    for source, rows in expected.items():
        record = index['sources'].get(source, {})
        if record.get('rows') != rows or not (ann_root / record.get('file', '')).is_file():
            raise ValueError(f'Incomplete ANN index: {source}')
    return actual


def check_search(package, url):
    import httpx
    specification = json.loads((Path(package) / 'corpus/EMBEDDING_SPEC.json').read_text())
    with httpx.Client(timeout=60, trust_env=False) as client:
        response = client.get(url.rstrip('/') + '/health'); response.raise_for_status()
        if response.json()['sources'] != specification['rows_by_source']:
            raise ValueError('Search service is using a different corpus.')
        response = client.post(url.rstrip('/') + '/embed', json={'texts': ['search service readiness'], 'batch_size': 1})
        response.raise_for_status()
        if len(response.json()['vectors'][0]) != specification['dimension']:
            raise ValueError('Unexpected search embedding dimension.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--image-base-url', default='')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-space', action='store_true')
    mode.add_argument('--check-retrieval', action='store_true')
    parser.add_argument('--embedding-root', type=Path)
    parser.add_argument('--ann-root', type=Path)
    mode.add_argument('--check-training', action='store_true')
    parser.add_argument('--search-url')
    args = parser.parse_args()
    if args.check_training and not args.search_url:
        parser.error('--check-training requires --search-url')
    package, output = args.package_root.resolve(), args.output_root.resolve()
    if output.is_relative_to(package.parent):
        parser.error('Runtime output must be outside SCAD_DATA.')
    if args.check_space:
        spec = json.loads((package / 'corpus/EMBEDDING_SPEC.json').read_text())
        dense_bytes = sum(spec['rows_by_source'].values()) * int(spec['dimension']) * 2
        output.mkdir(parents=True, exist_ok=True)
        completed = sum(p.stat().st_size for p in (output / 'embeddings').rglob('*.npy'))
        required = max(0, dense_bytes - completed) + 16 * 1024**3
        if shutil.disk_usage(output).free < required:
            raise ValueError(f'Need approximately {required / 1024**3:.1f} GiB free for remaining embeddings/indexes.')
        print(f'Dense vectors: {dense_bytes / 1024**3:.1f} GiB; sufficient disk available.')
    elif args.check_retrieval:
        print(json.dumps(check_retrieval(package, args.embedding_root or output / 'embeddings', args.ann_root or output / 'indices'), indent=2))
    elif args.check_training:
        state = json.loads((output / 'prepare_state.json').read_text())
        prepare(package, output, state['signature']['image_base_url'])
        if not (os.getenv('JUDGE_API_KEY') or os.getenv('OPENAI_API_KEY')):
            raise ValueError('Set the judge credentials in your local .env.')
        if not os.getenv('JUDGE_MODEL'):
            raise ValueError('Set JUDGE_MODEL explicitly for reproducible grading.')
        check_search(package, args.search_url)
        print('Dataset integrity, frozen-image cache, and retrieval service checks passed.')
    else:
        print(json.dumps(prepare(package, output, args.image_base_url), indent=2))


if __name__ == '__main__':
    main()
