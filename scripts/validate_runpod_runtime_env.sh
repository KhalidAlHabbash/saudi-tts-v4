#!/usr/bin/env bash
# Source this file, then call validate_runpod_runtime_env. Credentials must be
# injected by RunPod; this script never reads a credential file.

_validate_runpod_runtime_env() {
  local mode="$1"
  local aws_binary="$2"
  local forbidden_plaintext_file="$3"
  local name value

  if [[ "$mode" != "live" && "$mode" != "dry-run" ]]; then
    echo "Invalid RunPod environment validation mode: $mode" >&2
    return 2
  fi
  if [[ -e "$forbidden_plaintext_file" || -L "$forbidden_plaintext_file" ]]; then
    echo "Refusing persistent plaintext credential file; delete: $forbidden_plaintext_file" >&2
    return 1
  fi

  case ":${PATH:-}:" in
    *:/workspace/.local/bin:*) ;;
    *) PATH="/workspace/.local/bin:${PATH:-/usr/local/bin:/usr/bin:/bin}" ;;
  esac
  export PATH

  # Dry-run entrypoints retain their previous ability to use test-provided
  # values or no credentials when the underlying dry-run performs no request.
  if [[ "$mode" == "dry-run" ]]; then
    return 0
  fi

  for name in RUNPOD_NETWORK_VOLUME_ID RUNPOD_S3_DATACENTER AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
    value="${!name:-}"
    if [[ -z "$value" ]]; then
      echo "RunPod did not inject required environment variable: $name" >&2
      return 1
    fi
    case "$value" in
      *'{{'*|*'}}'*|*RUNPOD_SECRET_*)
        echo "RunPod managed-secret reference was not resolved for: $name" >&2
        return 1
        ;;
    esac
    export "$name"
  done
  if [[ ! -x "$aws_binary" ]]; then
    echo "AWS CLI is missing or not executable at $aws_binary" >&2
    return 1
  fi
}

validate_runpod_runtime_env() {
  local mode="${1:-live}"
  _validate_runpod_runtime_env \
    "$mode" \
    "/workspace/.local/bin/aws" \
    "/workspace/.saudi-tts/runtime.env"
}
