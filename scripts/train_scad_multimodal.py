"""Train against an explicitly started multimodal search server."""
import argparse
import json
from pathlib import Path
import subprocess

from scripts.mm_train_config import ROOT, settings_from_env, train_command
from scripts.mm_runtime import run_env, check_hardware, validate_output_directory, validate_models, validate_endpoints
from scripts.results import select_checkpoint, split_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    settings = settings_from_env()
    output = Path(settings['OUTPUT_DIR'])
    if not args.dry_run:
        validate_endpoints(settings)
        validate_output_directory(output, settings['RESUME_MODE'])
        output.mkdir(parents=True, exist_ok=True)
        check_hardware(settings)
        validate_models(settings['MODEL_PATH'], settings['TEACHER_MODEL_PATH'])
        (output / 'mm_run.json').write_text(json.dumps({'schema': 'mm_scad_v1', 'settings': settings}, indent=2) + '\n')
    subprocess.run(train_command(settings, dry_run=args.dry_run), cwd=ROOT,
                   env=run_env(settings, output), check=True)
    if not args.dry_run:
        select_checkpoint(output / 'metrics.jsonl', output / 'checkpoints', output / 'best_checkpoint.json',
                          split_counts(Path(settings['DATA_DIR']) / 'val.parquet'))


if __name__ == '__main__':
    main()
