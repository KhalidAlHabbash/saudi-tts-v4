"""Bounded, production-shaped CUDA resume probes for the copied update-100 state."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.training.checkpoints import build_sampler_fingerprint, load_resume, save_checkpoint
from src.training.config import load_config, resolve_path
from src.training.data import build_dataloader, build_dataset
from src.training.model import build_model, make_ema
from src.training.train import (
    _autocast,
    _prepare_rows,
    _schedule,
    clip_gradients,
    require_finite_loss,
    tensor_to_device,
)
from src.utils.device import select_device

GIB = 1024**3
PROFILES = {
    "batch1": {"frames_per_batch": 750, "max_samples_per_batch": 1, "gradient_accumulation_steps": 8},
    "batch2": {"frames_per_batch": 1500, "max_samples_per_batch": 2, "gradient_accumulation_steps": 4},
}


def probe_profile_config(cfg: dict[str, Any], profile: str) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown probe profile {profile!r}; choose from {sorted(PROFILES)}")
    result = copy.deepcopy(cfg)
    values = PROFILES[profile]
    result["data"]["frames_per_batch"] = values["frames_per_batch"]
    result["data"]["max_samples_per_batch"] = values["max_samples_per_batch"]
    result["optimization"]["gradient_accumulation_steps"] = values["gradient_accumulation_steps"]
    for key in (
        "expected_eligible_rows",
        "expected_microbatches_per_epoch",
        "expected_optimizer_updates_per_epoch",
    ):
        result["data"].pop(key, None)
    return result


def validate_probe_paths(source: Path, output_dir: Path) -> tuple[Path, Path]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    authoritative_dir = source.parent
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir == authoritative_dir or authoritative_dir in output_dir.parents:
        raise ValueError("Probe output must not be inside the authoritative checkpoint directory")
    return source, output_dir


def _assert_finite_state(state: dict[str, Any], *, label: str) -> None:
    stack: list[Any] = [state]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"Non-finite tensor detected in {label}")


def run_probe(
    cfg: dict[str, Any],
    *,
    source_checkpoint: Path,
    output_dir: Path,
    profile: str,
    optimizer_updates: int,
    memory_limit_gib: float,
) -> dict[str, Any]:
    if optimizer_updates < 2:
        raise ValueError("The resume probe requires at least two optimizer updates")
    if memory_limit_gib <= 0:
        raise ValueError("memory_limit_gib must be positive")
    source_checkpoint, output_dir = validate_probe_paths(source_checkpoint, output_dir)
    run_cfg = probe_profile_config(cfg, profile)

    choice = select_device(preference=run_cfg["runtime"]["device_preference"])
    if choice.device.type != "cuda":
        raise RuntimeError("The resume probe is RunPod/CUDA-only")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16")
    device = choice.device
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    all_rows, stats, unknown = _prepare_rows(run_cfg, "train")
    if unknown:
        raise ValueError("Probe manifest unexpectedly contains out-of-vocabulary text")
    rows_needed = (
        optimizer_updates
        * int(run_cfg["optimization"]["gradient_accumulation_steps"])
        * int(run_cfg["data"]["max_samples_per_batch"])
    )
    worst_case_rows = sorted(all_rows, key=lambda row: float(row["duration"]), reverse=True)[:rows_needed]
    if len(worst_case_rows) < rows_needed or float(worst_case_rows[-1]["duration"]) < 7.5:
        raise RuntimeError("Not enough >=7.5-second eligible rows for the worst-case resume probe")

    loader = build_dataloader(
        build_dataset(worst_case_rows, run_cfg["model"]["mel"]),
        frames_per_batch=run_cfg["data"]["frames_per_batch"],
        max_samples=run_cfg["data"]["max_samples_per_batch"],
        seed=run_cfg["optimization"]["seed"],
        num_workers=run_cfg["data"]["num_workers"],
        pin_memory=True,
        prefetch_factor=run_cfg["data"].get("prefetch_factor", 2),
    )
    required_microbatches = optimizer_updates * run_cfg["optimization"]["gradient_accumulation_steps"]
    if len(loader) < required_microbatches:
        raise RuntimeError(
            f"Probe loader produced {len(loader)} microbatches; {required_microbatches} are required"
        )

    fingerprint = build_sampler_fingerprint(
        train_manifest=resolve_path(run_cfg["data"]["train_manifest"]),
        eligible_row_count=len(all_rows),
        seed=run_cfg["optimization"]["seed"],
        f5_tts_version=run_cfg["upstream"]["f5_tts_version"],
        frames_threshold=run_cfg["data"]["frames_per_batch"],
        max_samples=run_cfg["data"]["max_samples_per_batch"],
        gradient_accumulation=run_cfg["optimization"]["gradient_accumulation_steps"],
    )

    model = build_model(run_cfg, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=run_cfg["optimization"]["learning_rate"],
        weight_decay=run_cfg["optimization"]["weight_decay"],
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_schedule(
            run_cfg["optimization"]["max_updates"],
            run_cfg["optimization"]["warmup_updates"],
        ),
    )
    ema = make_ema(model).to(device)
    update, epoch, next_batch = load_resume(
        source_checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        device=device,
        expected_sampler_fingerprint=fingerprint,
        legacy_migration=run_cfg["checkpointing"]["legacy_mac_to_cuda_migration"],
    )
    if (update, epoch, next_batch) != (100, 0, 0):
        raise RuntimeError(
            f"Probe source must be the legacy update-100/epoch-0 checkpoint; got "
            f"update={update}, epoch={epoch}, next_batch={next_batch}"
        )
    _assert_finite_state(model.state_dict(), label="loaded model")
    _assert_finite_state(ema.state_dict(), label="loaded EMA")
    _assert_finite_state(optimizer.state_dict(), label="loaded optimizer")

    optimizer.zero_grad(set_to_none=True)
    micro_count = 0
    completed = 0
    last_batch_index = -1
    last_loss = 0.0
    for batch_index, batch in enumerate(loader):
        last_batch_index = batch_index
        mel = tensor_to_device(batch["mel"].permute(0, 2, 1), device, pin_memory=True)
        lengths = tensor_to_device(batch["mel_lengths"], device, pin_memory=True)
        with _autocast(device, "bf16"):
            loss, _, _ = model(mel, text=batch["text"], lens=lengths)
        require_finite_loss(loss, context=f"probe {profile} batch {batch_index}")
        last_loss = float(loss.item())
        (loss / run_cfg["optimization"]["gradient_accumulation_steps"]).backward()
        micro_count += 1
        if micro_count < run_cfg["optimization"]["gradient_accumulation_steps"]:
            continue
        clip_gradients(
            model.parameters(),
            max_norm=run_cfg["optimization"]["max_grad_norm"],
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update()
        micro_count = 0
        completed += 1
        update += 1
        if completed >= optimizer_updates:
            break
    if completed != optimizer_updates:
        raise RuntimeError(f"Probe completed {completed}/{optimizer_updates} optimizer updates")

    _assert_finite_state(model.state_dict(), label="updated model")
    _assert_finite_state(ema.state_dict(), label="updated EMA")
    _assert_finite_state(optimizer.state_dict(), label="updated optimizer")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"probe_{profile}_update_{update}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=update,
        epoch=0,
        next_batch=last_batch_index + 1,
        device=device,
        sampler_fingerprint=fingerprint,
        minimum_free_gib=run_cfg["runtime"].get("minimum_free_disk_gib", 20),
    )
    reloaded = load_resume(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        device=device,
        expected_sampler_fingerprint=fingerprint,
    )
    if reloaded != (update, 0, last_batch_index + 1):
        raise RuntimeError(f"Probe checkpoint position failed strict reload: {reloaded}")
    _assert_finite_state(model.state_dict(), label="reloaded model")

    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / GIB
    result = {
        "status": "GO" if peak_reserved_gib <= memory_limit_gib else "NO-GO",
        "profile": profile,
        "source_checkpoint": str(source_checkpoint),
        "probe_checkpoint": str(checkpoint_path),
        "source_update": 100,
        "final_update": update,
        "optimizer_updates": completed,
        "gradient_accumulation_steps": run_cfg["optimization"]["gradient_accumulation_steps"],
        "max_samples_per_batch": run_cfg["data"]["max_samples_per_batch"],
        "frames_per_batch": run_cfg["data"]["frames_per_batch"],
        "minimum_probe_duration_seconds": min(float(row["duration"]) for row in worst_case_rows),
        "last_loss": last_loss,
        "sampler_fingerprint": fingerprint,
        "cuda_peak_reserved_gib": peak_reserved_gib,
        "memory_limit_gib": memory_limit_gib,
        "manifest_stats": stats,
    }
    report_path = output_dir / f"probe_{profile}_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "GO":
        raise RuntimeError(
            f"NO-GO: CUDA peak reserved memory {peak_reserved_gib:.2f} GiB exceeds "
            f"{memory_limit_gib:.2f} GiB"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_runpod.yaml")
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--optimizer-updates", type=int, default=2)
    parser.add_argument("--memory-limit-gib", type=float, default=21.5)
    args = parser.parse_args()
    run_probe(
        load_config(args.config),
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        profile=args.profile,
        optimizer_updates=args.optimizer_updates,
        memory_limit_gib=args.memory_limit_gib,
    )


if __name__ == "__main__":
    main()
