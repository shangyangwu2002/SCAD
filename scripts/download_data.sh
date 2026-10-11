#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/download_data.sh [all|text|multimodal] [--dry-run|--offline]'
  exit 0
fi
source "$(dirname "${BASH_SOURCE[0]}")/_scad_env.sh"
modality="${1:-all}"
[[ $# -eq 0 ]] || shift
exec "${PYTHON_BIN}" -m scripts.download_data --modality "${modality}" "$@"
