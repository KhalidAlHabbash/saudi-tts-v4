from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import runpod_payload_storage as storage


class FakeClient:
    bucket = "volume-test"
    dry_run = False

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.upload_order: list[str] = []

    def head(self, key: str):
        item = self.objects.get(key)
        if item is None:
            return None
        body, metadata = item
        return {"ContentLength": len(body), "Metadata": metadata}

    def upload(self, source: Path, key: str, *, sha256: str | None = None) -> None:
        self.objects[key] = (Path(source).read_bytes(), {"sha256": sha256} if sha256 else {})
        self.upload_order.append(key)

    def download(self, key: str, destination: str) -> str:
        assert destination == "-"
        return self.objects[key][0].decode("utf-8")

    def stream_sha256(self, key: str) -> tuple[str, int]:
        body = self.objects[key][0]
        return hashlib.sha256(body).hexdigest(), len(body)


def test_payload_keys_are_immutable_and_split_large_assets_from_project() -> None:
    digest = "a" * 64
    assert storage.payload_object_key("assets", digest) == (
        f"payload/objects/assets/sha256-{digest}.tar"
    )
    assert storage.PAYLOAD_COMPONENTS["assets"] == (
        "data/processed",
        "data/splits",
        "models/base",
    )
    assert set(storage.PAYLOAD_COMPONENTS) == {"assets", "project"}


def test_runpod_s3_transfer_config_limits_part_count_and_concurrency() -> None:
    config = storage.RUNPOD_S3_TRANSFER_CONFIG
    assert "multipart_chunksize = 128MB" in config
    assert "max_concurrent_requests = 2" in config
    assert "max_queue_size = 4" in config


def test_payload_latest_is_published_only_after_both_verified_archives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for paths in storage.PAYLOAD_COMPONENTS.values():
        for relative in paths:
            path = tmp_path / relative
            if Path(relative).suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)

    client = FakeClient()
    monkeypatch.setattr(storage, "_check_gnu_tar", lambda: None)
    monkeypatch.setattr(
        storage,
        "_digest_tar",
        lambda root, paths: (hashlib.sha256(b"x" * 123).hexdigest(), 123),
    )

    def fake_stream(root, paths, target, key, expected):
        target.objects[key] = (b"x" * expected[1], {"sha256": expected[0]})
        target.upload_order.append(key)

    monkeypatch.setattr(storage, "_stream_tar_upload", fake_stream)
    manifest = storage.upload_payload(tmp_path, client)

    assert [item["component"] for item in manifest["archives"]] == ["assets", "project"]
    assert client.upload_order[-1] == storage.PAYLOAD_LATEST_KEY
    assert len([key for key in client.objects if key.endswith(".tar")]) == 2


def test_legacy_seed_record_is_integrity_only_and_outside_strict_latest() -> None:
    record = storage.legacy_seed_record(size=2_603_000_000)
    assert record["sha256"] == storage.LEGACY_SHA256
    assert record["strict_latest_eligible"] is False
    assert record["resumable_after_migration_only"] is True
    assert record["object_key"].startswith("legacy-seeds/update-000000100/")
    assert "LATEST" not in record["object_key"]


def test_legacy_seed_upload_checks_metadata_and_never_publishes_latest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "model_last.pt"
    checkpoint.write_bytes(b"legacy")
    client = FakeClient()
    digest = hashlib.sha256(b"legacy").hexdigest()
    monkeypatch.setattr(storage, "LEGACY_SHA256", digest)
    monkeypatch.setattr(
        storage,
        "inspect_checkpoint",
        lambda path, require_sampler_fingerprint: {
            "update": 100,
            "epoch": 0,
            "next_batch": 800,
            "sampler_fingerprint": None,
        },
    )
    monkeypatch.setattr(storage, "sha256_file", lambda path: digest)

    record = storage.upload_legacy_seed(checkpoint, client)

    assert record["strict_latest_eligible"] is False
    assert all("LATEST" not in key for key in client.objects)
    assert record["object_key"] in client.objects
    assert record["metadata_key"] in client.objects


def test_legacy_seed_rejects_any_fingerprinted_or_wrong_position_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"wrong")
    monkeypatch.setattr(
        storage,
        "inspect_checkpoint",
        lambda path, require_sampler_fingerprint: {
            "update": 100,
            "epoch": 0,
            "next_batch": 0,
            "sampler_fingerprint": {"digest": "already-production"},
        },
    )
    with pytest.raises(RuntimeError, match="exact fingerprintless legacy"):
        storage.upload_legacy_seed(checkpoint, FakeClient())
