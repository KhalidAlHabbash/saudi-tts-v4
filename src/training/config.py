"""Configuration loading and cross-checks for the pinned SILMA architecture."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path = "configs/train.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any], *, require_assets: bool = True) -> None:
    if version("f5-tts") != cfg["upstream"]["f5_tts_version"]:
        raise RuntimeError(
            f"Installed f5-tts {version('f5-tts')} does not match {cfg['upstream']['f5_tts_version']}"
        )
    if cfg["model"]["backbone"] != "DiT":
        raise ValueError("Pinned SILMA v1 Small requires the F5-TTS DiT backbone")
    arch = cfg["model"]["architecture"]
    expected_arch = {"dim": 768, "depth": 18, "heads": 12, "ff_mult": 2, "text_dim": 512, "conv_layers": 4}
    for key, value in expected_arch.items():
        if arch.get(key) != value:
            raise ValueError(f"SILMA architecture mismatch: {key} must be {value!r}")
    if arch.get("attn_backend") != "torch":
        raise ValueError("The validated SILMA path requires attn_backend=torch")
    mel = cfg["model"]["mel"]
    expected_mel = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024,
        "mel_spec_type": "vocos",
    }
    for key, value in expected_mel.items():
        if mel.get(key) != value:
            raise ValueError(f"SILMA mel/Vocos mismatch: {key} must be {value!r}")
    opt = cfg["optimization"]
    precision = opt.get("mixed_precision")
    if precision not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be 'no' or 'bf16'")
    if precision == "no" and not cfg["runtime"].get("fp32"):
        raise ValueError("FP32 must be enabled when mixed precision is disabled")
    if precision == "bf16" and cfg["runtime"].get("device_preference") != "cuda_then_mps_then_cpu":
        raise ValueError("BF16 is enabled only for the validated CUDA configuration")
    if opt.get("freeze") != "none":
        raise ValueError("No unvalidated freezing policy is allowed for the baseline")
    if cfg["data"]["max_samples_per_batch"] < 1:
        raise ValueError("max_samples_per_batch must be positive")
    if cfg["data"]["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    if cfg["data"].get("expected_eligible_rows", 1) < 1:
        raise ValueError("expected_eligible_rows must be positive when configured")

    max_updates = int(opt["max_updates"])
    stage_targets = [int(value) for value in opt.get("stage_targets", [max_updates])]
    if not stage_targets or stage_targets != sorted(set(stage_targets)):
        raise ValueError("stage_targets must be a non-empty, strictly increasing list")
    if stage_targets[-1] != max_updates or any(target <= 0 for target in stage_targets):
        raise ValueError("stage_targets must be positive and end at optimization.max_updates")
    stage_stop = int(opt.get("stage_stop_update", max_updates))
    if stage_stop not in stage_targets:
        raise ValueError("stage_stop_update must be one of optimization.stage_targets")

    benchmark = cfg.get("benchmark")
    if benchmark is not None:
        if cfg["runtime"].get("device_preference") != "cuda_then_mps_then_cpu":
            raise ValueError("benchmark controls are valid only in the CUDA profile")
        allowed_benchmark_stops = [int(value) for value in benchmark.get("allowed_stop_updates", [])]
        if int(benchmark.get("source_update", -1)) != 100 or allowed_benchmark_stops != [200, 300]:
            raise ValueError("benchmark must start at update 100 and allow only stop updates 200 and 300")
        if allowed_benchmark_stops[-1] >= stage_targets[0]:
            raise ValueError("benchmark stops must remain below the first formal stage")

    checkpointing = cfg["checkpointing"]
    for key in ("save_every_updates", "save_last_every_updates"):
        if int(checkpointing[key]) <= 0:
            raise ValueError(f"checkpointing.{key} must be positive")
    migration = checkpointing.get("legacy_mac_to_cuda_migration")
    if migration is not None:
        if cfg["runtime"].get("device_preference") != "cuda_then_mps_then_cpu":
            raise ValueError("legacy Mac-to-CUDA migration is valid only in the CUDA profile")
        expected = {
            "enabled": True,
            "allowed_update": 100,
            "allowed_epoch": 0,
            "allowed_next_batch": 800,
            "reset_next_batch": True,
        }
        if any(migration.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "legacy_mac_to_cuda_migration must allow only update 100 / epoch 0 / "
                "next_batch 800 and reset next_batch"
            )
    durable = checkpointing.get("durable_upload")
    if durable is not None:
        allowed_durable_keys = {"enabled", "fail_closed", "lock_path"}
        unexpected = sorted(set(durable).difference(allowed_durable_keys))
        if unexpected:
            raise ValueError(
                f"durable_upload accepts only environment-independent controls; unexpected={unexpected}"
            )
        if durable.get("enabled", False):
            if cfg["runtime"].get("device_preference") != "cuda_then_mps_then_cpu":
                raise ValueError("automatic durable upload is valid only in the CUDA profile")
            if durable.get("fail_closed") is not True:
                raise ValueError("automatic durable upload must be fail_closed")
            if int(checkpointing["save_every_updates"]) != 5000:
                raise ValueError("automatic durable upload requires immutable saves every 5000 updates")
            lock_path = Path(str(durable.get("lock_path", "")))
            if not lock_path.is_absolute() or lock_path.parent != Path("/workspace"):
                raise ValueError("durable_upload.lock_path must be an absolute file under /workspace")
    minimum_free = float(cfg["runtime"].get("minimum_free_disk_gib", 0))
    if minimum_free < 0:
        raise ValueError("runtime.minimum_free_disk_gib cannot be negative")

    expected_batches = cfg["data"].get("expected_microbatches_per_epoch")
    expected_updates = cfg["data"].get("expected_optimizer_updates_per_epoch")
    if expected_batches is not None or expected_updates is not None:
        if expected_batches is None or expected_updates is None:
            raise ValueError("expected microbatch and optimizer-update counts must be set together")
        if int(expected_batches) % int(opt["gradient_accumulation_steps"]):
            raise ValueError("expected microbatches must divide evenly by gradient accumulation")
        if int(expected_batches) // int(opt["gradient_accumulation_steps"]) != int(expected_updates):
            raise ValueError("expected optimizer updates per epoch disagrees with accumulation")
    if require_assets:
        for key in ("checkpoint", "model_config", "vocab"):
            if not resolve_path(cfg["upstream"][key]).is_file():
                raise FileNotFoundError(resolve_path(cfg["upstream"][key]))
        vocos = resolve_path(cfg["upstream"]["vocos_dir"])
        for filename in ("config.yaml", "pytorch_model.bin"):
            if not (vocos / filename).is_file():
                raise FileNotFoundError(vocos / filename)

    # Cross-check the downloaded vendor config so local settings cannot silently drift.
    if require_assets:
        with resolve_path(cfg["upstream"]["model_config"]).open(encoding="utf-8") as handle:
            vendor = yaml.safe_load(handle)
        for key, value in expected_arch.items():
            if vendor["model"]["arch"].get(key) != value:
                raise RuntimeError(f"Downloaded SILMA config disagrees on model.arch.{key}")
        for key, value in expected_mel.items():
            if vendor["model"]["mel_spec"].get(key) != value:
                raise RuntimeError(f"Downloaded SILMA config disagrees on model.mel_spec.{key}")
