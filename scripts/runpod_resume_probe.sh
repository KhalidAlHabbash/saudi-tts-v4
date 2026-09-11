#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_checkpoint="${1:-$project_root/checkpoints/silma-saudi/model_last.pt}"
output_root="${2:-$project_root/outputs/runpod-resume-probe}"
python_bin="$project_root/.venv-runpod/bin/python"

cd "$project_root"
export PYTHONPATH="$project_root"

"$python_bin" -m src.training.resume_probe \
  --config configs/train_runpod.yaml \
  --source-checkpoint "$source_checkpoint" \
  --output-dir "$output_root/batch1" \
  --profile batch1 \
  --optimizer-updates 2 \
  --memory-limit-gib 21.5

"$python_bin" -m src.training.resume_probe \
  --config configs/train_runpod.yaml \
  --source-checkpoint "$source_checkpoint" \
  --output-dir "$output_root/batch2" \
  --profile batch2 \
  --optimizer-updates 2 \
  --memory-limit-gib 21.5
