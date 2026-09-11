#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_NETWORK_VOLUME_ID:?Set RUNPOD_NETWORK_VOLUME_ID to the 100 GB network volume id}"
: "${RUNPOD_S3_DATACENTER:?Set RUNPOD_S3_DATACENTER to the volume datacenter}"
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID to the RunPod user id}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY to the RunPod S3 API key}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
endpoint="https://s3api-${RUNPOD_S3_DATACENTER,,}.runpod.io/"
destination="s3://${RUNPOD_NETWORK_VOLUME_ID}/saudi-tts-finetune"
aws_args=(--region "$RUNPOD_S3_DATACENTER" --endpoint-url "$endpoint")
lock_file="/workspace/.saudi-tts-durable-storage.lock"
dry_run_args=()
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run_args=(--dryrun)
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "Another durable storage operation holds $lock_file" >&2
  exit 1
fi

for directory in configs docs reports scripts src tests data/processed data/splits models/base; do
  aws s3 sync "$project_root/$directory/" "$destination/$directory/" \
    --only-show-errors "${dry_run_args[@]}" "${aws_args[@]}"
done
for filename in README.md THIRD_PARTY_NOTICES.md pyproject.toml requirements-runpod.txt requirements.txt; do
  aws s3 cp "$project_root/$filename" "$destination/$filename" \
    --only-show-errors "${dry_run_args[@]}" "${aws_args[@]}"
done

echo "RunPod payload sync completed. Checkpoints are intentionally handled by sync_runpod_checkpoint.sh."
