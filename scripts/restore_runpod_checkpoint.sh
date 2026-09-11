#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-$project_root/checkpoints/silma-saudi/model_last.pt}"
shift || true

cd "$project_root"
export PYTHONPATH="$project_root"
exec "$project_root/.venv-runpod/bin/python" -m src.training.durable_storage \
  restore "$destination" "$@"
