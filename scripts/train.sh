#!/usr/bin/env bash
# This is the sole project launcher for a meaningful fine-tuning run.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root"
exec .venv/bin/python -m src.training.train --config configs/train.yaml --train "$@"
