from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch

import src.training.checkpoints as checkpointing
from src.training.checkpoints import GIB, ensure_free_space
from src.training.config import load_config, validate_config
from src.training.durable_storage import (
    AwsCli,
    checkpoint_object_key,
    endpoint_for,
    merge_manifest,
    new_record,
    single_flight_lock,
)
from src.training.resume_probe import probe_profile_config, validate_probe_paths
from src.training.train import (
    configure_benchmark_determinism,
    durable_upload_roles,
    immutable_checkpoint_artifact,
    publish_durable_checkpoint,
    resolve_training_stop,
    tensor_to_device,
)


def _record(update: int, *, roles: tuple[str, ...] = ("resumable",)):
    return new_record(update=update, sha256=f"{update:064x}", size=1000 + update, roles=roles)


def test_runpod_s3_commands_lowercase_endpoint_and_use_immutable_key():
    assert endpoint_for("EU-RO-1") == "https://s3api-eu-ro-1.runpod.io/"
    key = checkpoint_object_key(13122, "a" * 64)
    assert key == f"checkpoints/objects/update-000013122/sha256-{'a' * 64}.pt"
    command = AwsCli(bucket="volume_123", datacenter="EU-RO-1", dry_run=True).command(
        "s3api", "head-object", "--bucket", "volume_123", "--key", key
    )
    assert command[-4:] == [
        "--region",
        "EU-RO-1",
        "--endpoint-url",
        "https://s3api-eu-ro-1.runpod.io/",
    ]


def test_durable_manifest_keeps_three_resumable_and_all_permanent():
    manifest = None
    retired = []
    for update in (100, 5000, 10000, 13122):
        manifest, retired = merge_manifest(manifest, _record(update))
    assert [item["update"] for item in manifest["resumable"]] == [5000, 10000, 13122]
    assert [item["update"] for item in retired] == [100]

    manifest, retired = merge_manifest(manifest, _record(13122, roles=("resumable", "stage")))
    assert retired == []
    assert [item["update"] for item in manifest["permanent"]] == [13122]
    assert manifest["permanent"][0]["object_key"] == manifest["resumable"][-1]["object_key"]


def test_single_flight_lock_rejects_concurrent_operation(tmp_path):
    lock = tmp_path / "sync.lock"
    with single_flight_lock(lock):
        with pytest.raises(RuntimeError, match="Another durable storage operation"):
            with single_flight_lock(lock):
                pass


def test_free_space_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(100 * GIB, 90 * GIB, 10 * GIB))
    with pytest.raises(RuntimeError, match="Insufficient free space"):
        ensure_free_space(tmp_path / "future" / "checkpoints", minimum_free_gib=20)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(100 * GIB, 70 * GIB, 30 * GIB))
    assert ensure_free_space(tmp_path, minimum_free_gib=20) == 30 * GIB


def test_durable_replace_fsyncs_file_then_directory(monkeypatch, tmp_path):
    source = tmp_path / "checkpoint.pt.tmp"
    destination = tmp_path / "checkpoint.pt"
    source.write_bytes(b"complete-checkpoint")
    calls = []
    monkeypatch.setattr(checkpointing, "fsync_file", lambda path: calls.append(("file", path)))
    monkeypatch.setattr(checkpointing, "fsync_directory", lambda path: calls.append(("directory", path)))

    checkpointing.durable_replace(source, destination)

    assert destination.read_bytes() == b"complete-checkpoint"
    assert not source.exists()
    assert calls == [("file", source), ("directory", tmp_path)]


def test_checkpoint_alias_uses_durable_publication(monkeypatch, tmp_path):
    source = tmp_path / "model_5000.pt"
    destination = tmp_path / "model_last.pt"
    source.write_bytes(b"checkpoint")
    calls = []
    monkeypatch.setattr(checkpointing, "fsync_file", lambda path: calls.append(("file", path.name)))
    monkeypatch.setattr(checkpointing, "fsync_directory", lambda path: calls.append(("directory", path)))

    checkpointing.link_checkpoint(source, destination)

    assert destination.read_bytes() == b"checkpoint"
    assert calls == [("file", "model_last.pt.link.tmp"), ("directory", tmp_path)]


def test_benchmark_stop_is_bounded_deterministic_and_not_a_stage(monkeypatch):
    cfg = load_config("configs/train_runpod.yaml")
    assert resolve_training_stop(cfg, benchmark_stop_update=200) == (200, "benchmark")
    assert resolve_training_stop(cfg, benchmark_stop_update=300) == (300, "benchmark")
    assert resolve_training_stop(cfg, stage_stop_update=13122) == (13122, "stage")
    with pytest.raises(ValueError, match="benchmark stop"):
        resolve_training_stop(cfg, benchmark_stop_update=400)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    configure_benchmark_determinism(cfg, stop_kind="benchmark", smoke_one_step=False)
    assert cfg["runtime"]["deterministic"] is True
    assert cfg["optimization"]["max_updates"] == 120000
    assert cfg["optimization"]["stage_targets"] == [13122, 26144, 52188, 104276, 120000]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    directory = Path("checkpoints/silma-saudi")
    assert immutable_checkpoint_artifact(
        directory,
        update=300,
        stop_target=300,
        stop_kind="benchmark",
        save_every_updates=5000,
    ) == (directory / "benchmark_300.pt", "benchmark")
    assert immutable_checkpoint_artifact(
        directory,
        update=13122,
        stop_target=13122,
        stop_kind="stage",
        save_every_updates=5000,
    ) == (directory / "stage_13122.pt", "stage")
    assert immutable_checkpoint_artifact(
        directory,
        update=5000,
        stop_target=13122,
        stop_kind="stage",
        save_every_updates=5000,
    ) == (directory / "model_5000.pt", "routine")


def test_automatic_durable_upload_roles_and_fail_closed_mock(tmp_path):
    cfg = load_config("configs/train_runpod.yaml")
    checkpointing_cfg = cfg["checkpointing"]
    assert durable_upload_roles(checkpointing_cfg, artifact_kind="routine") == ("resumable",)
    assert durable_upload_roles(checkpointing_cfg, artifact_kind="stage") == ("resumable", "stage")
    assert durable_upload_roles(checkpointing_cfg, artifact_kind="benchmark") is None

    checkpoint = tmp_path / "model_5000.pt"
    checkpoint.touch()
    calls = []

    def successful_upload(path, *, roles, lock_path):
        calls.append((path, tuple(roles), lock_path))

    publish_durable_checkpoint(
        checkpoint,
        roles=("resumable",),
        durable_config=checkpointing_cfg["durable_upload"],
        uploader=successful_upload,
    )
    assert calls == [
        (
            checkpoint,
            ("resumable",),
            Path("/workspace/.saudi-tts-durable-storage.lock"),
        )
    ]

    def failed_upload(*_args, **_kwargs):
        raise OSError("mock S3 failure")

    with pytest.raises(RuntimeError, match="local checkpoint is preserved"):
        publish_durable_checkpoint(
            checkpoint,
            roles=("resumable",),
            durable_config=checkpointing_cfg["durable_upload"],
            uploader=failed_upload,
        )


def test_durable_upload_config_is_cuda_only_fail_closed_and_has_no_credentials():
    cfg = load_config("configs/train_runpod.yaml")
    assert set(cfg["checkpointing"]["durable_upload"]) == {"enabled", "fail_closed", "lock_path"}

    cfg["checkpointing"]["durable_upload"]["fail_closed"] = False
    with pytest.raises(ValueError, match="fail_closed"):
        validate_config(cfg, require_assets=False)

    cfg = load_config("configs/train_runpod.yaml")
    cfg["checkpointing"]["durable_upload"]["AWS_SECRET_ACCESS_KEY"] = "forbidden"
    with pytest.raises(ValueError, match="environment-independent"):
        validate_config(cfg, require_assets=False)

    local = load_config("configs/train.yaml")
    local["checkpointing"]["durable_upload"] = {
        "enabled": True,
        "fail_closed": True,
        "lock_path": "/workspace/.saudi-tts-durable-storage.lock",
    }
    with pytest.raises(ValueError, match="CUDA profile"):
        validate_config(local, require_assets=False)


def test_probe_profiles_and_output_isolation(tmp_path):
    cfg = load_config("configs/train_runpod.yaml")
    batch1 = probe_profile_config(cfg, "batch1")
    batch2 = probe_profile_config(cfg, "batch2")
    assert (batch1["data"]["max_samples_per_batch"], batch1["optimization"]["gradient_accumulation_steps"]) == (1, 8)
    assert (batch2["data"]["max_samples_per_batch"], batch2["optimization"]["gradient_accumulation_steps"]) == (2, 4)
    source_dir = tmp_path / "checkpoints" / "silma-saudi"
    source_dir.mkdir(parents=True)
    source = source_dir / "model_last.pt"
    source.touch()
    output = tmp_path / "outputs" / "runpod-resume-probe"
    assert validate_probe_paths(source, output) == (source.resolve(), output.resolve())
    with pytest.raises(ValueError, match="authoritative checkpoint"):
        validate_probe_paths(source, source_dir / "probe")


class _TransferRecorder:
    def __init__(self):
        self.calls = []

    def to(self, device, *, non_blocking):
        self.calls.append((device.type, non_blocking))
        return self


@pytest.mark.parametrize(
    ("device_type", "pin_memory", "expected_non_blocking"),
    [("cuda", True, True), ("cuda", False, False), ("mps", True, False), ("cpu", True, False)],
)
def test_tensor_transfer_is_nonblocking_only_for_pinned_cuda(device_type, pin_memory, expected_non_blocking):
    tensor = _TransferRecorder()
    assert tensor_to_device(tensor, torch.device(device_type), pin_memory=pin_memory) is tensor
    assert tensor.calls == [(device_type, expected_non_blocking)]
