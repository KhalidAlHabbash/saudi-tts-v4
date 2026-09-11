#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root"
exec .venv-runpod/bin/python -m src.training.train --config configs/train_runpod.yaml --train "$@"
