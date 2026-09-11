#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${1:-$project_root/checkpoints/silma-saudi/model_last.pt}"
shift || true
if [[ "$#" -eq 0 ]]; then
  set -- --role resumable
fi

cd "$project_root"
export PYTHONPATH="$project_root"
exec "$project_root/.venv-runpod/bin/python" -m src.training.durable_storage \
  upload "$checkpoint" "$@"
