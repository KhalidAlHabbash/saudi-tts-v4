#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="/workspace/saudi-tts-training.pid"
log_file="/workspace/saudi-tts-training.log"
lock_file="/workspace/saudi-tts-training.launch.lock"

source "$project_root/scripts/validate_runpod_runtime_env.sh"
validate_runpod_runtime_env live

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another launcher holds $lock_file" >&2
  exit 1
fi
if [[ -f "$pid_file" ]]; then
  old_pid="$(tr -d '[:space:]' < "$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Training is already running as PID $old_pid" >&2
    exit 1
  fi
fi

cd "$project_root"
setsid "$project_root/scripts/train_runpod.sh" "$@" >"$log_file" 2>&1 </dev/null &
training_pid=$!
pid_tmp="${pid_file}.tmp"
printf '%s\n' "$training_pid" >"$pid_tmp"
mv "$pid_tmp" "$pid_file"
echo "Launched PID $training_pid"
echo "Log: $log_file"
echo "PID: $pid_file"
