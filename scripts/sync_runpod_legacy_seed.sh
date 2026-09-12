#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint="${1:-$project_root/checkpoints/silma-saudi/model_last.pt}"
if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [LEGACY_UPDATE100_CHECKPOINT]" >&2
  exit 2
fi

source "$project_root/scripts/validate_runpod_runtime_env.sh"
validate_runpod_runtime_env live

lock_file="/workspace/.saudi-tts-durable-storage.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another durable storage operation holds $lock_file" >&2
  exit 1
fi

cd "$project_root"
exec "$project_root/.venv-runpod/bin/python" \
  -m scripts.runpod_payload_storage upload-legacy-seed "$checkpoint"
