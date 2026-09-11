"""Full-state checkpoints, sampler compatibility, and storage safety guards."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from ema_pytorch import EMA

GIB = 1024**3
_REQUIRED_CHECKPOINT_KEYS = {
    "model_state_dict",
    "optimizer_state_dict",
    "ema_model_state_dict",
    "scheduler_state_dict",
    "update",
    "trainer_state",
}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sampler_fingerprint(
    *,
    train_manifest: Path,
    eligible_row_count: int,
    seed: int,
    f5_tts_version: str,
    frames_threshold: int,
    max_samples: int,
    gradient_accumulation: int,
) -> dict[str, Any]:
    """Return a stable description and digest of the resume-critical data order."""
    fields: dict[str, Any] = {
        "format_version": 1,
        "train_manifest_sha256": sha256_file(train_manifest),
        "eligible_row_count": int(eligible_row_count),
        "seed": int(seed),
        "f5_tts_version": str(f5_tts_version),
        "frames_threshold": int(frames_threshold),
        "max_samples": int(max_samples),
        "gradient_accumulation": int(gradient_accumulation),
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**fields, "digest": hashlib.sha256(canonical).hexdigest()}


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(path)
        candidate = candidate.parent
    return candidate


def ensure_free_space(path: Path, *, minimum_free_gib: float) -> int:
    """Fail before a run/save when the target filesystem lacks the safety reserve."""
    if minimum_free_gib < 0:
        raise ValueError("minimum_free_gib cannot be negative")
    target = _existing_ancestor(path)
    free = shutil.disk_usage(target).free
    required = int(minimum_free_gib * GIB)
    if free < required:
        raise RuntimeError(
            f"Insufficient free space on {target}: {free / GIB:.2f} GiB available; "
            f"at least {minimum_free_gib:.2f} GiB is required"
        )
    return free


def fsync_file(path: Path) -> None:
    """Flush a completed file payload to its backing device."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Flush directory metadata after an atomic publication rename."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_replace(source: Path, destination: Path) -> None:
    """Durably publish a completed temp file with an atomic replacement."""
    fsync_file(source)
    os.replace(source, destination)
    fsync_directory(destination.parent)


def _validate_structure(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    missing = sorted(_REQUIRED_CHECKPOINT_KEYS.difference(checkpoint))
    if missing:
        raise RuntimeError(f"{path} is missing checkpoint keys: {missing}")
    trainer_state = checkpoint.get("trainer_state")
    if not trainer_state or trainer_state.get("format_version") != 1:
        raise RuntimeError(f"{path} is not an exact-resume project checkpoint")
    for key in ("epoch", "next_batch", "rng"):
        if key not in trainer_state:
            raise RuntimeError(f"{path} trainer_state is missing {key!r}")
    return trainer_state


def _validate_sampler_resume(
    *,
    checkpoint: dict[str, Any],
    trainer_state: dict[str, Any],
    expected: dict[str, Any] | None,
    legacy_migration: dict[str, Any] | None,
    path: Path,
) -> bool:
    """Validate sampler identity and report whether the one-time reset applies."""
    if expected is None:
        return False
    saved = trainer_state.get("sampler_fingerprint")
    if saved is not None:
        if saved != expected:
            changed = sorted(
                key
                for key in set(saved).union(expected)
                if saved.get(key) != expected.get(key)
            )
            raise RuntimeError(
                f"Sampler fingerprint mismatch for {path}; changed fields: {changed}. "
                "Refusing to reuse next_batch with a different data order."
            )
        return False

    policy = legacy_migration or {}
    allowed_update = policy.get("allowed_update")
    allowed_epoch = policy.get("allowed_epoch")
    allowed_next_batch = policy.get("allowed_next_batch")
    if (
        not policy.get("enabled", False)
        or allowed_update is None
        or int(checkpoint["update"]) != int(allowed_update)
        or allowed_epoch is None
        or int(trainer_state["epoch"]) != int(allowed_epoch)
        or allowed_next_batch is None
        or int(trainer_state["next_batch"]) != int(allowed_next_batch)
        or not policy.get("reset_next_batch", False)
    ):
        raise RuntimeError(
            f"{path} has no sampler fingerprint and is not the explicitly allowed "
            "legacy Mac update checkpoint"
        )
    return True


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
    }
    if device.type == "mps" and hasattr(torch, "mps"):
        state["torch_mps"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    torch.random.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    if device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    if device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    update: int,
    epoch: int,
    next_batch: int,
    device: torch.device,
    sampler_fingerprint: dict[str, Any] | None = None,
    minimum_free_gib: float = 0.0,
) -> None:
    ensure_free_space(path.parent, minimum_free_gib=minimum_free_gib)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_model_state_dict": ema.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "update": update,
        "trainer_state": {
            "format_version": 1,
            "epoch": epoch,
            "next_batch": next_batch,
            "rng": _rng_state(device),
            "sampler_fingerprint": sampler_fingerprint,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def link_checkpoint(source: Path, destination: Path, *, minimum_free_gib: float = 0.0) -> None:
    """Atomically publish a same-filesystem checkpoint alias without reserializing it."""
    ensure_free_space(destination.parent, minimum_free_gib=minimum_free_gib)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".link.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        # Cross-filesystem aliases are not expected in the project layout, but a
        # copy is safer than losing model_last if that invariant ever changes.
        shutil.copy2(source, temporary)
    durable_replace(temporary, destination)


def load_resume(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: EMA,
    device: torch.device,
    expected_sampler_fingerprint: dict[str, Any] | None = None,
    legacy_migration: dict[str, Any] | None = None,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    trainer_state = _validate_structure(checkpoint, path)
    reset_position = _validate_sampler_resume(
        checkpoint=checkpoint,
        trainer_state=trainer_state,
        expected=expected_sampler_fingerprint,
        legacy_migration=legacy_migration,
        path=path,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ema.load_state_dict(checkpoint["ema_model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    _restore_rng(trainer_state["rng"], device)
    next_batch = 0 if reset_position else int(trainer_state["next_batch"])
    return int(checkpoint["update"]), int(trainer_state["epoch"]), next_batch


def inspect_checkpoint(path: Path, *, require_sampler_fingerprint: bool = True) -> dict[str, Any]:
    """Strictly inspect a downloaded checkpoint without constructing the model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    trainer_state = _validate_structure(checkpoint, path)
    fingerprint = trainer_state.get("sampler_fingerprint")
    if require_sampler_fingerprint and not fingerprint:
        raise RuntimeError(f"{path} lacks the required sampler fingerprint")
    return {
        "update": int(checkpoint["update"]),
        "epoch": int(trainer_state["epoch"]),
        "next_batch": int(trainer_state["next_batch"]),
        "sampler_fingerprint": fingerprint,
    }


def rotate_checkpoints(directory: Path, keep_last_n: int) -> None:
    if keep_last_n < 0:
        return
    numbered = []
    for path in directory.glob("model_*.pt"):
        try:
            numbered.append((int(path.stem.split("_")[1]), path))
        except (IndexError, ValueError):
            continue
    numbered.sort()
    for _, path in numbered[:-keep_last_n] if keep_last_n else numbered:
        path.unlink()
