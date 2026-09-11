#!/usr/bin/env bash
set -euo pipefail

pid_file="/workspace/saudi-tts-training.pid"
log_file="/workspace/saudi-tts-training.log"
if [[ ! -f "$pid_file" ]]; then
  echo "No PID file at $pid_file" >&2
  exit 1
fi
training_pid="$(tr -d '[:space:]' < "$pid_file")"
if [[ ! "$training_pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid PID file: $pid_file" >&2
  exit 1
fi
if ! kill -0 "$training_pid" 2>/dev/null; then
  echo "PID $training_pid is not running; removing stale PID file"
  rm -f "$pid_file"
  exit 0
fi

kill -TERM "$training_pid"
echo "Requested a safe stop for PID $training_pid. The trainer will finish an optimizer update and save model_last."
for _ in $(seq 1 120); do
  if ! kill -0 "$training_pid" 2>/dev/null; then
    rm -f "$pid_file"
    echo "Stopped. Confirm the final checkpoint line in $log_file"
    exit 0
  fi
  sleep 1
done
echo "PID $training_pid is still shutting down; do not send SIGKILL while a .tmp checkpoint exists." >&2
exit 1
