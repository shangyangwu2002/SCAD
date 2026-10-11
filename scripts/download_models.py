"""Download public reference models to the release layout (explicit invocation only)."""
import argparse
import os
from pathlib import Path

RELEASE_MODEL = 'shangyangwu2002/SCAD'
RELEASE_REVISION = '981b29e65047c833810aa1fdff686b8fe5b9053d'


def model_repositories(task):
    names = ['Qwen/Qwen3-Embedding-4B']
    if task in {'text', 'all'}:
        names += ['Qwen/Qwen3-4B', 'Qwen/Qwen3-1.7B', 'Qwen/Qwen3-4B-Instruct-2507']
    if task in {'multimodal', 'all'}:
        names += ['Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct']
    if task in {'released', 'all'}:
        names.append(RELEASE_MODEL)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['released', 'text', 'multimodal', 'all'], default='text')
    parser.add_argument('--output-root', type=Path, default=Path(os.getenv('MODELS_ROOT', 'models')))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for name in model_repositories(args.task):
        revision = RELEASE_REVISION if name == RELEASE_MODEL else None
        snapshot_download(repo_id=name, revision=revision, local_dir=args.output_root / name)


if __name__ == '__main__':
    main()
