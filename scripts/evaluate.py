"""Evaluate released Hugging Face weights or a validation-selected checkpoint on test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

from scripts.results import split_counts, summarize
from scripts.prepare_scad_data import prepare, check_search

ROOT = Path(__file__).resolve().parents[1]


def selected_checkpoint(output, override=None):
    """Read validation selection; never select a checkpoint from test scores."""
    output = Path(output)
    if override:
        checkpoint = Path(override)
    else:
        for path in (output / 'checkpoints/best_checkpoint.json', output / 'best_checkpoint.json'):
            if path.is_file():
                record = json.loads(path.read_text())
                value = record.get('checkpoint_path') or record.get('checkpoint')
                if not value:
                    raise ValueError(f'Missing checkpoint path in {path}')
                checkpoint = Path(value)
                break
        else:
            raise ValueError('No validation-selected checkpoint. Finish training or set BEST_CHECKPOINT.')
    return checkpoint.resolve()


def check_checkpoint(checkpoint, gpus):
    if not (checkpoint / 'data.pt').is_file():
        raise ValueError(f'Incomplete checkpoint: {checkpoint}')
    for rank in range(gpus):
        if not (checkpoint / 'actor' / f'model_world_size_{gpus}_rank_{rank}.pt').is_file():
            raise ValueError(f'Checkpoint lacks actor shard {rank}; use the training GPU count.')


def check_hf_model(model):
    """Check a local, exported Hugging Face model without loading its weights."""
    model = Path(model).resolve()
    for name in ('config.json', 'tokenizer_config.json'):
        if not (model / name).is_file():
            raise ValueError(f'Incomplete Hugging Face model: missing {model / name}')
    files = []
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (model / name).is_file():
            mapping = json.loads((model / name).read_text()).get('weight_map', {})
            if not mapping:
                raise ValueError(f'Empty model weight index: {model / name}')
            files = sorted(set(mapping.values()))
            break
    if not files:
        files = [name for name in ('model.safetensors', 'pytorch_model.bin') if (model / name).is_file()]
    if not files:
        raise ValueError(f'No Hugging Face model weights found: {model}')
    for name in files:
        path = (model / name).resolve()
        if not path.is_relative_to(model) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or invalid model shard: {name}')


def check_inputs(package, runtime, data, multimodal, search_url):
    import pyarrow.parquet as pq
    state = json.loads((runtime / 'prepare_state.json').read_text())
    prepare(package, runtime, state['signature']['image_base_url'])
    if not (os.getenv('JUDGE_API_KEY') or os.getenv('OPENAI_API_KEY')) or not os.getenv('JUDGE_MODEL'):
        raise ValueError('Configure JUDGE_API_KEY and JUDGE_MODEL in .env.')
    rows = pq.read_table(data).to_pylist()
    modes = [row['extra_info']['retrieval_mode'] for row in rows]
    if any(mode != 'serpapi_live' for mode in modes):
        check_search(package, search_url)
    if 'serpapi_live' in modes:
        from envs.serpapi_search import parse_serpapi_keys
        if not parse_serpapi_keys(os.getenv('SERPAPI_API_KEYS') or os.getenv('SERPAPI_API_KEY')):
            raise ValueError('Live evaluation requires SERPAPI_API_KEYS in .env.')
        if multimodal:
            mapping = json.loads(Path(os.environ['MM_IMAGE_URL_MAP']).read_text())
            for row, mode in zip(rows, modes):
                if mode != 'serpapi_live':
                    continue
                for key in row['extra_info']['image_paths']:
                    url = urlsplit(mapping.get(key, ''))
                    if url.scheme not in {'http', 'https'} or not url.hostname:
                        raise ValueError('Live image search requires publicly accessible HTTP(S) image URLs; configure MM_PUBLIC_IMAGE_BASE_URL as described in README.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['4b', '1_7b', '2b'], required=True)
    parser.add_argument('--split', choices=['test'], default='test')
    parser.add_argument('--model', type=Path, help='Local Hugging Face model directory; evaluate without a training checkpoint.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = Path(os.environ['OUTPUT_DIR']).resolve()
    runtime = Path(os.environ['RUNTIME_ROOT']).resolve()
    package = Path(os.environ['PACKAGE_ROOT']).resolve()
    data = runtime / 'dataset' / f'{args.split}.parquet'
    model = args.model.resolve() if args.model else None
    if model and os.getenv('BEST_CHECKPOINT'):
        parser.error('Use either --model or BEST_CHECKPOINT, not both.')
    if model:
        checkpoint = None
    elif args.dry_run and not os.getenv('BEST_CHECKPOINT'):
        checkpoint = output / 'checkpoints/global_step_10'
    else:
        checkpoint = selected_checkpoint(output, os.getenv('BEST_CHECKPOINT'))
    result_dir = Path(os.getenv('EVAL_OUTPUT_DIR') or output / f'test_{(model or checkpoint).name}').resolve()
    if result_dir == output or result_dir.is_relative_to(output / 'checkpoints') or result_dir == runtime or result_dir.is_relative_to(package.parent):
        raise ValueError('Evaluation must use a separate output directory outside the data package.')
    if not args.dry_run:
        if model:
            check_hf_model(model)
        else:
            check_checkpoint(checkpoint, int(os.environ['N_GPUS']))
        check_inputs(package, runtime, data, args.profile == '2b', os.environ['LOCAL_SEARCH_URL'])
        if result_dir.exists() and any(result_dir.iterdir()):
            raise ValueError('Evaluation output already contains files; choose a new EVAL_OUTPUT_DIR.')
    if args.profile == '2b':
        from scripts.mm_train_config import settings_from_env, train_command
        from scripts.mm_runtime import run_env, validate_models
        settings = settings_from_env()
        if model:
            settings['MODEL_PATH'] = str(model)
        if not args.dry_run:
            validate_models(settings['MODEL_PATH'])
        command = train_command(settings, split=args.split, checkpoint=checkpoint,
                                output_dir=result_dir, dry_run=args.dry_run,
                                model_path=str(model) if model else None)
        env = run_env(settings, result_dir, evaluation=True)
    else:
        command = ['bash', str(ROOT / 'scripts/_launch_text.sh')]
        env = dict(os.environ)
        env.update({'OUTPUT_DIR': str(result_dir), 'VAL_FILE': str(data), 'VAL_ONLY': 'True',
                    'VAL_BEFORE_TRAIN': 'True', 'VAL_MAX_SAMPLES': '-1', 'STRATIFIED_SOURCE_SAMPLING': 'False',
                    'ADV_ESTIMATOR': 'scad', 'SCAD_ENABLE': 'False',
                    'SCAD_NODE_MERGE': 'False', 'SCAD_TREE_CREDIT': 'False',
                    'TRACK_BEST_CHECKPOINT': 'False', 'KEEP_BEST_AND_LATEST': 'False',
                    'RESUME_MODE': 'disable' if model else 'resume_path',
                    'CHECKPOINT_SAVE_CONTENTS': '[model,extra]',
                    'METRICS_FILE': str(result_dir / 'test_metrics.jsonl'),
                    'VALIDATION_DIR': str(result_dir / 'test_predictions'),
                    'EXPERIMENT_NAME': f'scad_{args.profile}_{args.split}',
                    'DRY_RUN': '1' if args.dry_run else '0'})
        if model:
            env['MODEL_PATH'] = str(model)
            env.pop('RESUME_FROM_PATH', None)
        else:
            env['RESUME_FROM_PATH'] = str(checkpoint)
    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.dry_run:
        summarize(result_dir / 'test_metrics.jsonl', result_dir, split_counts(data), args.split)


if __name__ == '__main__':
    main()
