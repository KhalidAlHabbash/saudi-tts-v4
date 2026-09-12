#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_env_mode="live"
command_args=(upload-payload --project-root "$project_root")
if [[ "${1:-}" == "--dry-run" ]]; then
  runtime_env_mode="dry-run"
  command_args=(plan-payload --project-root "$project_root")
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
source "$project_root/scripts/validate_runpod_runtime_env.sh"
validate_runpod_runtime_env "$runtime_env_mode"

if [[ "$runtime_env_mode" == "live" ]]; then
  : "${RUNPOD_NETWORK_VOLUME_ID:?Set RUNPOD_NETWORK_VOLUME_ID to the network volume id}"
  : "${RUNPOD_S3_DATACENTER:?Set RUNPOD_S3_DATACENTER to the volume datacenter}"
  : "${AWS_ACCESS_KEY_ID:?RunPod must inject AWS_ACCESS_KEY_ID}"
  : "${AWS_SECRET_ACCESS_KEY:?RunPod must inject AWS_SECRET_ACCESS_KEY}"
fi

lock_file="/workspace/.saudi-tts-durable-storage.lock"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another durable storage operation holds $lock_file" >&2
  exit 1
fi

cd "$project_root"
exec "$project_root/.venv-runpod/bin/python" \
  -m scripts.runpod_payload_storage "${command_args[@]}"
