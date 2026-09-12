#!/usr/bin/env python3
"""Archive-oriented RunPod S3 storage for payloads and the legacy update-100 seed.

The large payload is streamed twice: once to establish its deterministic digest
and size, and once directly into AWS CLI's multipart upload. No archive-sized
temporary file is written to the 70 GB working volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.training.checkpoints import inspect_checkpoint, sha256_file
from src.training.durable_storage import AwsCli, verify_remote_object

LEGACY_UPDATE = 100
LEGACY_EPOCH = 0
LEGACY_NEXT_BATCH = 800
LEGACY_SHA256 = "7b54ac7c31462d7c24b76e16ed679a95754cd00f7bdc5ab10d86961fe418e82e"
PAYLOAD_LATEST_KEY = "payload/LATEST.json"
RUNPOD_S3_TRANSFER_CONFIG = """[default]
s3 =
    max_concurrent_requests = 2
    max_queue_size = 4
    multipart_threshold = 64MB
    multipart_chunksize = 128MB
"""
PAYLOAD_COMPONENTS: dict[str, tuple[str, ...]] = {
    "assets": ("data/processed", "data/splits", "models/base"),
    "project": (
        "configs",
        "docs",
        "reports",
        "scripts",
        "src",
        "tests",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "requirements-runpod.txt",
        "requirements.txt",
    ),
}


def payload_object_key(component: str, digest: str) -> str:
    if component not in PAYLOAD_COMPONENTS or len(digest) != 64:
        raise ValueError("Invalid payload component or SHA-256")
    return f"payload/objects/{component}/sha256-{digest}.tar"


def legacy_seed_record(*, size: int) -> dict[str, Any]:
    prefix = f"legacy-seeds/update-{LEGACY_UPDATE:09d}"
    key = f"{prefix}/sha256-{LEGACY_SHA256}.pt"
    return {
        "format_version": 1,
        "kind": "legacy-mac-to-cuda-migration-seed",
        "update": LEGACY_UPDATE,
        "epoch": LEGACY_EPOCH,
        "next_batch": LEGACY_NEXT_BATCH,
        "sha256": LEGACY_SHA256,
        "size": size,
        "object_key": key,
        "sha256_sidecar_key": f"{key}.sha256",
        "strict_latest_eligible": False,
        "resumable_after_migration_only": True,
    }


def _require_client() -> AwsCli:
    bucket = os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")
    datacenter = os.environ.get("RUNPOD_S3_DATACENTER", "")
    if not bucket or not datacenter:
        raise RuntimeError("RunPod network-volume environment is missing")
    return AwsCli(bucket=bucket, datacenter=datacenter)


def _tar_command(paths: Sequence[str]) -> list[str]:
    return [
        "tar", "--sort=name", "--format=posix", "--mtime=@0", "--owner=0",
        "--group=0", "--numeric-owner", "--pax-option=delete=atime,delete=ctime",
        "-cf", "-", "--", *paths,
    ]


def _check_sources(root: Path, paths: Iterable[str]) -> None:
    missing = [item for item in paths if not (root / item).exists()]
    if missing:
        raise RuntimeError(f"Payload sources are missing: {missing}")


def _check_gnu_tar() -> None:
    result = subprocess.run(["tar", "--version"], text=True, capture_output=True, check=True)
    if "GNU tar" not in result.stdout:
        raise RuntimeError("Deterministic payload archives require GNU tar")


def _digest_tar(root: Path, paths: Sequence[str]) -> tuple[str, int]:
    process = subprocess.Popen(
        _tar_command(paths), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"tar failed while hashing payload: {stderr.strip()}")
    return digest.hexdigest(), size


def _stream_tar_upload(
    root: Path, paths: Sequence[str], client: AwsCli, key: str, expected: tuple[str, int]
) -> None:
    expected_digest, expected_size = expected
    command = client.command(
        "s3", "cp", "-", f"s3://{client.bucket}/{key}", "--only-show-errors",
        "--expected-size", str(expected_size), "--metadata", f"sha256={expected_digest}",
        "--cli-connect-timeout", "60", "--cli-read-timeout", "7200",
    )
    # RunPod caps each multipart part at 500 MB. AWS CLI defaults to 8 MB and
    # ten concurrent requests; that produced thousands of parts and an
    # InvalidPart completion failure for the 29 GB assets stream. Keep the
    # transfer bounded to 128 MB parts and two concurrent requests using an
    # ephemeral, non-secret AWS config file.
    with tempfile.TemporaryDirectory(prefix="saudi-tts-aws-config-") as temporary:
        config_path = Path(temporary) / "config"
        config_path.write_text(RUNPOD_S3_TRANSFER_CONFIG, encoding="ascii")
        aws_environment = os.environ.copy()
        aws_environment["AWS_CONFIG_FILE"] = str(config_path)
        aws_environment["AWS_RETRY_MODE"] = "standard"
        aws_environment["AWS_MAX_ATTEMPTS"] = "15"
        aws_environment["AWS_PAGER"] = ""

        tar = subprocess.Popen(
            _tar_command(paths), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        aws = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=aws_environment,
        )
        assert tar.stdout is not None and aws.stdin is not None
        digest = hashlib.sha256()
        size = 0
        stream_error: BrokenPipeError | None = None
        try:
            for chunk in iter(lambda: tar.stdout.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                aws.stdin.write(chunk)
        except BrokenPipeError as exc:
            stream_error = exc
            tar.terminate()
        finally:
            try:
                aws.stdin.close()
            except BrokenPipeError:
                pass
        tar_stderr = tar.stderr.read().decode("utf-8", errors="replace") if tar.stderr else ""
        aws_stdout = aws.stdout.read() if aws.stdout else b""
        aws_stderr = aws.stderr.read().decode("utf-8", errors="replace") if aws.stderr else ""
        tar_code = tar.wait()
        aws_code = aws.wait()
    actual = (digest.hexdigest(), size)
    if stream_error is not None or tar_code != 0 or aws_code != 0 or actual != expected:
        client.delete(key)
        raise RuntimeError(
            "Payload changed or upload failed; the unverified object was removed. "
            f"tar={tar_code}, aws={aws_code}, expected={expected}, actual={actual}, "
            f"tar_error={tar_stderr.strip()}, aws_error={aws_stderr.strip()}, "
            f"aws_output={aws_stdout.decode(errors='replace').strip()}"
        )


def _verify_object(client: AwsCli, key: str, digest: str, size: int) -> None:
    verify_remote_object(
        client,
        key=key,
        expected_size=size,
        expected_sha256=digest,
    )


def _publish_sidecar(client: AwsCli, key: str, digest: str) -> str:
    sidecar_key = f"{key}.sha256"
    with tempfile.TemporaryDirectory(prefix="saudi-tts-payload-") as temporary:
        path = Path(temporary) / "object.sha256"
        content = f"{digest}  {Path(key).name}\n"
        path.write_text(content, encoding="ascii")
        if client.head(sidecar_key) is None:
            client.upload(path, sidecar_key)
        head = client.head(sidecar_key)
        if head is None or int(head.get("ContentLength", -1)) != path.stat().st_size:
            raise RuntimeError(f"Sidecar size verification failed for {key}")
        if client.download(sidecar_key, "-") != content:
            raise RuntimeError(f"Sidecar content verification failed for {key}")
    return sidecar_key


def upload_payload(root: Path, client: AwsCli) -> dict[str, Any]:
    _check_gnu_tar()
    records: list[dict[str, Any]] = []
    for component, paths in PAYLOAD_COMPONENTS.items():
        _check_sources(root, paths)
        digest, size = _digest_tar(root, paths)
        key = payload_object_key(component, digest)
        if client.head(key) is None:
            _stream_tar_upload(root, paths, client, key, (digest, size))
        _verify_object(client, key, digest, size)
        sidecar_key = _publish_sidecar(client, key, digest)
        records.append({
            "component": component, "paths": list(paths), "sha256": digest, "size": size,
            "object_key": key, "sha256_sidecar_key": sidecar_key,
        })
    manifest = {
        "format_version": 1,
        "kind": "complete-runpod-payload-snapshot",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "archives": records,
    }
    with tempfile.TemporaryDirectory(prefix="saudi-tts-payload-") as temporary:
        path = Path(temporary) / "LATEST.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        client.upload(path, PAYLOAD_LATEST_KEY)
        head = client.head(PAYLOAD_LATEST_KEY)
        if head is None or int(head.get("ContentLength", -1)) != path.stat().st_size:
            raise RuntimeError("Payload LATEST.json verification failed")
    return manifest


def upload_legacy_seed(checkpoint: Path, client: AwsCli) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    metadata = inspect_checkpoint(checkpoint, require_sampler_fingerprint=False)
    expected_metadata = {
        "update": LEGACY_UPDATE, "epoch": LEGACY_EPOCH,
        "next_batch": LEGACY_NEXT_BATCH, "sampler_fingerprint": None,
    }
    if metadata != expected_metadata:
        raise RuntimeError(f"Not the exact fingerprintless legacy update-100 seed: {metadata}")
    digest = sha256_file(checkpoint)
    if digest != LEGACY_SHA256:
        raise RuntimeError(f"Legacy seed SHA-256 mismatch: expected {LEGACY_SHA256}, got {digest}")
    record = legacy_seed_record(size=checkpoint.stat().st_size)
    if client.head(record["object_key"]) is None:
        client.upload(checkpoint, record["object_key"], sha256=digest)
    _verify_object(client, record["object_key"], digest, record["size"])
    _publish_sidecar(client, record["object_key"], digest)
    metadata_key = record["object_key"].removesuffix(".pt") + ".json"
    record["metadata_key"] = metadata_key
    with tempfile.TemporaryDirectory(prefix="saudi-tts-legacy-") as temporary:
        path = Path(temporary) / "metadata.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if client.head(metadata_key) is None:
            client.upload(path, metadata_key)
        head = client.head(metadata_key)
        if head is None or int(head.get("ContentLength", -1)) != path.stat().st_size:
            raise RuntimeError("Legacy seed metadata verification failed")
        if client.download(metadata_key, "-") != path.read_text(encoding="utf-8"):
            raise RuntimeError("Legacy seed metadata content verification failed")
    return record


def plan_payload(root: Path) -> None:
    for component, paths in PAYLOAD_COMPONENTS.items():
        _check_sources(root, paths)
        print(json.dumps({"component": component, "paths": list(paths), "operation": "hash-then-stream"}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan-payload", "upload-payload"):
        child = commands.add_parser(name)
        child.add_argument("--project-root", type=Path, required=True)
    legacy = commands.add_parser("upload-legacy-seed")
    legacy.add_argument("checkpoint", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan-payload":
        plan_payload(args.project_root.resolve())
    elif args.command == "upload-payload":
        print(json.dumps(upload_payload(args.project_root.resolve(), _require_client()), indent=2, sort_keys=True))
    else:
        print(json.dumps(upload_legacy_seed(args.checkpoint, _require_client()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
