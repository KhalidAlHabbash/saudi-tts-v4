"""Durable RunPod network-volume checkpoint sync through its S3-compatible API."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.training.checkpoints import inspect_checkpoint, sha256_file

LATEST_KEY = "checkpoints/LATEST.json"
PERMANENT_ROLES = {"stage", "milestone", "best"}
VALID_ROLES = {"resumable", *PERMANENT_ROLES}
OBJECT_PATTERN = re.compile(
    r"^checkpoints/objects/update-(?P<update>\d{9})/sha256-(?P<sha>[0-9a-f]{64})\.pt$"
)


def endpoint_for(datacenter: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9-]+", datacenter):
        raise ValueError(f"Invalid RunPod datacenter: {datacenter!r}")
    return f"https://s3api-{datacenter.lower()}.runpod.io/"


def aws_global_args(datacenter: str) -> list[str]:
    return ["--region", datacenter, "--endpoint-url", endpoint_for(datacenter)]


def checkpoint_object_key(update: int, sha256: str) -> str:
    if update < 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Invalid checkpoint update or SHA-256")
    return f"checkpoints/objects/update-{update:09d}/sha256-{sha256}.pt"


def new_record(*, update: int, sha256: str, size: int, roles: Sequence[str]) -> dict[str, Any]:
    role_set = sorted(set(roles))
    invalid = sorted(set(role_set).difference(VALID_ROLES))
    if not role_set or invalid:
        raise ValueError(f"Checkpoint roles must be from {sorted(VALID_ROLES)}; invalid={invalid}")
    key = checkpoint_object_key(update, sha256)
    return {
        "update": int(update),
        "sha256": sha256,
        "size": int(size),
        "object_key": key,
        "sha256_sidecar_key": f"{key}.sha256",
        "roles": role_set,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }


def empty_manifest() -> dict[str, Any]:
    return {"format_version": 1, "resumable": [], "permanent": []}


def merge_manifest(
    existing: dict[str, Any] | None,
    record: dict[str, Any],
    *,
    keep_resumable: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if keep_resumable != 3:
        raise ValueError("Durable policy is fixed at the latest three resumable checkpoints")
    manifest = dict(existing or empty_manifest())
    if manifest.get("format_version") != 1:
        raise RuntimeError("Unsupported durable manifest format")
    all_existing = [*manifest.get("resumable", []), *manifest.get("permanent", [])]
    for old in all_existing:
        if old["update"] == record["update"] and old["sha256"] != record["sha256"]:
            raise RuntimeError(
                f"Update {record['update']} already points at a different immutable checkpoint hash"
            )

    by_key = {item["object_key"]: dict(item) for item in all_existing}
    combined = by_key.get(record["object_key"], dict(record))
    combined["roles"] = sorted(set(combined.get("roles", [])).union(record["roles"]))
    by_key[record["object_key"]] = combined

    resumable_before = {item["object_key"]: item for item in manifest.get("resumable", [])}
    resumable = sorted(
        (item for item in by_key.values() if "resumable" in item["roles"]),
        key=lambda item: (item["update"], item["sha256"]),
    )
    kept_resumable = resumable[-keep_resumable:]
    permanent = sorted(
        (item for item in by_key.values() if PERMANENT_ROLES.intersection(item["roles"])),
        key=lambda item: (item["update"], item["sha256"]),
    )
    live_keys = {item["object_key"] for item in [*kept_resumable, *permanent]}
    retired = [item for key, item in resumable_before.items() if key not in live_keys]
    updated = {
        "format_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "resumable": kept_resumable,
        "permanent": permanent,
    }
    return updated, retired


class AwsCli:
    def __init__(self, *, bucket: str, datacenter: str, dry_run: bool = False) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", bucket):
            raise ValueError("RUNPOD_NETWORK_VOLUME_ID contains invalid characters")
        self.bucket = bucket
        self.datacenter = datacenter
        self.global_args = aws_global_args(datacenter)
        self.dry_run = dry_run

    def command(self, *args: str) -> list[str]:
        return ["aws", *args, *self.global_args]

    def run(self, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        if self.dry_run:
            print(json.dumps({"dry_run_command": list(args)}))
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.run(args, check=check, text=True, capture_output=True)

    def head(self, key: str) -> dict[str, Any] | None:
        result = self.run(
            self.command("s3api", "head-object", "--bucket", self.bucket, "--key", key, "--output", "json"),
            check=False,
        )
        if self.dry_run:
            return None
        if result.returncode != 0:
            combined = f"{result.stdout}\n{result.stderr}".lower()
            if any(token in combined for token in ("404", "not found", "nosuchkey")):
                return None
            raise RuntimeError(f"HeadObject failed for {key}: {result.stderr.strip()}")
        return json.loads(result.stdout)

    def upload(self, source: Path, key: str, *, sha256: str | None = None) -> None:
        args = self.command(
            "s3",
            "cp",
            str(source),
            f"s3://{self.bucket}/{key}",
            "--only-show-errors",
            *( ["--metadata", f"sha256={sha256}"] if sha256 else [] ),
        )
        self.run(args)

    def download(self, key: str, destination: str | Path) -> str:
        result = self.run(
            self.command(
                "s3",
                "cp",
                f"s3://{self.bucket}/{key}",
                str(destination),
                "--only-show-errors",
            )
        )
        return result.stdout

    def delete(self, key: str) -> None:
        self.run(
            self.command("s3", "rm", f"s3://{self.bucket}/{key}", "--only-show-errors"),
            check=False,
        )


def _require_environment(*, live: bool) -> tuple[str, str]:
    bucket = os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    datacenter = os.environ.get("RUNPOD_S3_DATACENTER")
    missing = [
        name
        for name, value in (
            ("RUNPOD_NETWORK_VOLUME_ID", bucket),
            ("RUNPOD_S3_DATACENTER", datacenter),
        )
        if not value
    ]
    if live:
        missing.extend(
            name
            for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
            if not os.environ.get(name)
        )
    if missing:
        raise RuntimeError(f"Missing required environment variables: {sorted(set(missing))}")
    return str(bucket), str(datacenter)


@contextmanager
def single_flight_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another durable storage operation holds {path}") from exc
        yield


def _verify_head(
    client: AwsCli,
    *,
    key: str,
    expected_size: int,
    expected_sha256: str | None = None,
) -> None:
    if client.dry_run:
        return
    head = client.head(key)
    if head is None or int(head.get("ContentLength", -1)) != expected_size:
        raise RuntimeError(f"HeadObject size verification failed for {key}")
    if expected_sha256 is not None:
        metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
        if metadata.get("sha256") != expected_sha256:
            raise RuntimeError(f"HeadObject SHA-256 metadata verification failed for {key}")


def _load_remote_manifest(client: AwsCli) -> dict[str, Any] | None:
    if client.dry_run or client.head(LATEST_KEY) is None:
        return None
    payload = client.download(LATEST_KEY, "-")
    manifest = json.loads(payload)
    if manifest.get("format_version") != 1:
        raise RuntimeError("Remote LATEST.json has an unsupported format")
    return manifest


def upload_checkpoint(
    checkpoint: Path,
    *,
    roles: Sequence[str],
    dry_run: bool = False,
    lock_path: Path = Path("/workspace/.saudi-tts-durable-storage.lock"),
) -> dict[str, Any]:
    bucket, datacenter = _require_environment(live=not dry_run)
    client = AwsCli(bucket=bucket, datacenter=datacenter, dry_run=dry_run)
    checkpoint = checkpoint.resolve()
    metadata = inspect_checkpoint(checkpoint, require_sampler_fingerprint=True)
    digest = sha256_file(checkpoint)
    record = new_record(update=metadata["update"], sha256=digest, size=checkpoint.stat().st_size, roles=roles)

    with single_flight_lock(lock_path):
        existing_manifest = _load_remote_manifest(client)
        existing_head = client.head(record["object_key"])
        if existing_head is None:
            client.upload(checkpoint, record["object_key"], sha256=digest)
        _verify_head(
            client,
            key=record["object_key"],
            expected_size=record["size"],
            expected_sha256=digest,
        )

        with tempfile.TemporaryDirectory(prefix="saudi-tts-s3-") as temporary:
            temporary_dir = Path(temporary)
            sidecar = temporary_dir / "checkpoint.sha256"
            sidecar.write_text(f"{digest}  {Path(record['object_key']).name}\n", encoding="ascii")
            sidecar_head = client.head(record["sha256_sidecar_key"])
            if sidecar_head is None:
                client.upload(sidecar, record["sha256_sidecar_key"])
            _verify_head(
                client,
                key=record["sha256_sidecar_key"],
                expected_size=sidecar.stat().st_size,
            )
            if not client.dry_run:
                remote_sidecar = client.download(record["sha256_sidecar_key"], "-")
                if remote_sidecar != sidecar.read_text(encoding="ascii"):
                    raise RuntimeError("Remote SHA-256 sidecar content verification failed")

            manifest, retired = merge_manifest(existing_manifest, record)
            manifest_path = temporary_dir / "LATEST.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            client.upload(manifest_path, LATEST_KEY)
            _verify_head(client, key=LATEST_KEY, expected_size=manifest_path.stat().st_size)

        # Retention is intentionally after publishing LATEST. Stage/milestone/best
        # objects are referenced in `permanent` and can never enter this list.
        for old in retired:
            client.delete(old["sha256_sidecar_key"])
            client.delete(old["object_key"])

    print(json.dumps({"uploaded": record, "latest": manifest}, indent=2, sort_keys=True))
    return manifest


def _record_for_key(manifest: dict[str, Any], object_key: str) -> dict[str, Any]:
    for record in [*manifest.get("resumable", []), *manifest.get("permanent", [])]:
        if record["object_key"] == object_key:
            return record
    raise RuntimeError(f"{object_key} is not referenced by LATEST.json")


def restore_checkpoint(
    destination: Path,
    *,
    object_key: str | None = None,
    replace: bool = False,
    dry_run: bool = False,
    lock_path: Path = Path("/workspace/.saudi-tts-durable-storage.lock"),
) -> dict[str, Any]:
    bucket, datacenter = _require_environment(live=not dry_run)
    client = AwsCli(bucket=bucket, datacenter=datacenter, dry_run=dry_run)
    destination = destination.resolve()
    if destination.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --replace explicitly")

    with single_flight_lock(lock_path):
        manifest = _load_remote_manifest(client)
        if dry_run and manifest is None:
            if object_key is None:
                raise ValueError("Dry-run restore requires --object-key")
            match = OBJECT_PATTERN.fullmatch(object_key)
            if not match:
                raise ValueError("Object key is not an immutable update/hash checkpoint name")
            record = {
                "update": int(match.group("update")),
                "sha256": match.group("sha"),
                "object_key": object_key,
                "sha256_sidecar_key": f"{object_key}.sha256",
            }
            client.download(object_key, destination.with_suffix(destination.suffix + ".part"))
            return record
        if manifest is None:
            raise RuntimeError("No durable LATEST.json exists")
        if object_key is None:
            resumable = manifest.get("resumable", [])
            if not resumable:
                raise RuntimeError("LATEST.json contains no resumable checkpoint")
            record = resumable[-1]
        else:
            record = _record_for_key(manifest, object_key)

        head = client.head(record["object_key"])
        if head is None or int(head.get("ContentLength", -1)) != int(record["size"]):
            raise RuntimeError("Remote checkpoint size disagrees with LATEST.json")
        if client.head(record["sha256_sidecar_key"]) is None:
            raise RuntimeError("Remote checkpoint SHA-256 sidecar is missing")
        sidecar_digest = client.download(record["sha256_sidecar_key"], "-").strip().split()[0]
        if sidecar_digest != record["sha256"]:
            raise RuntimeError("Remote SHA-256 sidecar disagrees with LATEST.json")
        part = destination.with_suffix(destination.suffix + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.unlink(missing_ok=True)
        try:
            client.download(record["object_key"], part)
            if part.stat().st_size != int(record["size"]):
                raise RuntimeError("Downloaded checkpoint size mismatch")
            digest = sha256_file(part)
            if digest != record["sha256"]:
                raise RuntimeError("Downloaded checkpoint SHA-256 mismatch")
            metadata = inspect_checkpoint(part, require_sampler_fingerprint=True)
            if metadata["update"] != int(record["update"]):
                raise RuntimeError("Strict checkpoint update disagrees with LATEST.json")
            os.replace(part, destination)
        finally:
            part.unlink(missing_ok=True)
    result = {"restored": str(destination), "record": record, "checkpoint": metadata}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    upload = subparsers.add_parser("upload")
    upload.add_argument("checkpoint", type=Path)
    upload.add_argument("--role", action="append", choices=sorted(VALID_ROLES), required=True)
    upload.add_argument("--dry-run", action="store_true")
    upload.add_argument("--lock-path", type=Path, default=Path("/workspace/.saudi-tts-durable-storage.lock"))
    restore = subparsers.add_parser("restore")
    restore.add_argument("destination", type=Path)
    restore.add_argument("--object-key")
    restore.add_argument("--replace", action="store_true")
    restore.add_argument("--dry-run", action="store_true")
    restore.add_argument("--lock-path", type=Path, default=Path("/workspace/.saudi-tts-durable-storage.lock"))
    args = parser.parse_args()
    if args.command == "upload":
        upload_checkpoint(
            args.checkpoint,
            roles=args.role,
            dry_run=args.dry_run,
            lock_path=args.lock_path,
        )
    else:
        restore_checkpoint(
            args.destination,
            object_key=args.object_key,
            replace=args.replace,
            dry_run=args.dry_run,
            lock_path=args.lock_path,
        )


if __name__ == "__main__":
    main()
