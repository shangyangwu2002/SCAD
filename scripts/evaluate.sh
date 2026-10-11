#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || $# -lt 1 ]]; then
  echo 'Usage: bash scripts/evaluate.sh {4b|1_7b|2b} [test] [--model LOCAL_HF_MODEL] [--dry-run]'
  exit 0
fi
profile="$1"
shift
# The optional literal `test` is retained for existing evaluation commands.
if [[ "${1:-}" == test ]]; then shift; fi
export SCAD_EVAL_SPLIT=test
case "${profile}" in
  4b|1_7b|2b) exec bash "$(dirname "${BASH_SOURCE[0]}")/train_${profile}.sh" "$@" ;;
  *) echo 'Expected 4b, 1_7b, or 2b.' >&2; exit 2 ;;
esac
