"""Train, smoke-test, or dry-run the pinned SILMA/F5-TTS Saudi integration."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import signal
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.training.assets import materialize_assets
from src.training.checkpoints import (
    build_sampler_fingerprint,
    ensure_free_space,
    link_checkpoint,
    load_resume,
    rotate_checkpoints,
    save_checkpoint,
)
from src.training.config import load_config, resolve_path
from src.training.data import (
    build_dataloader,
    build_dataset,
    load_vocab,
    read_manifest,
    validate_audio_headers,
    vocabulary_audit,
)
from src.training.model import build_model, load_pretrained, make_ema
from src.training import durable_storage
from src.utils.device import select_device


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    strict_determinism: bool = False,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=not strict_determinism)


def _autocast(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "bf16"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    """Use asynchronous host-to-CUDA copies only for pinned-memory loaders."""
    non_blocking = bool(pin_memory and device.type == "cuda")
    return tensor.to(device, non_blocking=non_blocking)


def require_finite_loss(loss: torch.Tensor, *, context: str) -> None:
    if loss.numel() != 1 or not bool(torch.isfinite(loss).item()):
        value = loss.detach().float().cpu().tolist()
        raise FloatingPointError(f"Non-finite scalar loss at {context}: {value!r}")


def clip_gradients(parameters, *, max_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        error_if_nonfinite=True,
    )


def resolve_stage_stop(optimization: dict[str, Any], override: int | None = None) -> int:
    max_updates = int(optimization["max_updates"])
    targets = {int(value) for value in optimization.get("stage_targets", [max_updates])}
    target = int(optimization.get("stage_stop_update", max_updates) if override is None else override)
    if target not in targets:
        raise ValueError(f"stage stop {target} is not one of {sorted(targets)}")
    if target > max_updates:
        raise ValueError("stage stop cannot exceed the scheduler max_updates horizon")
    return target


def normalize_resume_position(
    epoch: int,
    next_batch: int,
    batches_per_epoch: int,
) -> tuple[int, int]:
    """Canonicalize an exact end-of-epoch checkpoint to the next epoch.

    DataLoader materializes a batch before the training loop can skip it. Leaving
    an exact boundary as ``next_batch == len(loader)`` would therefore reload and
    discard a complete epoch on resume. Positions within an epoch are preserved.
    """
    if epoch < 0:
        raise ValueError("Checkpoint epoch cannot be negative")
    if batches_per_epoch <= 0:
        raise ValueError("Training loader must contain at least one batch")
    if next_batch < 0 or next_batch > batches_per_epoch:
        raise ValueError(
            f"Checkpoint next_batch {next_batch} is outside loader range "
            f"0..{batches_per_epoch}"
        )
    if next_batch == batches_per_epoch:
        return epoch + 1, 0
    return epoch, next_batch


def resolve_training_stop(
    cfg: dict[str, Any],
    *,
    stage_stop_update: int | None = None,
    benchmark_stop_update: int | None = None,
) -> tuple[int, str]:
    if stage_stop_update is not None and benchmark_stop_update is not None:
        raise ValueError("Stage and benchmark stop modes are mutually exclusive")
    if benchmark_stop_update is not None:
        allowed = {int(value) for value in cfg.get("benchmark", {}).get("allowed_stop_updates", [])}
        if benchmark_stop_update not in allowed:
            raise ValueError(f"benchmark stop {benchmark_stop_update} is not one of {sorted(allowed)}")
        if benchmark_stop_update >= min(int(value) for value in cfg["optimization"]["stage_targets"]):
            raise ValueError("A benchmark stop must remain below the first formal stage")
        return benchmark_stop_update, "benchmark"
    return resolve_stage_stop(cfg["optimization"], stage_stop_update), "stage"


def durable_upload_roles(checkpointing: dict[str, Any], *, artifact_kind: str) -> tuple[str, ...] | None:
    if not checkpointing.get("durable_upload", {}).get("enabled", False):
        return None
    if artifact_kind == "routine":
        return ("resumable",)
    if artifact_kind == "stage":
        return ("resumable", "stage")
    return None


def immutable_checkpoint_artifact(
    checkpoint_dir: Path,
    *,
    update: int,
    stop_target: int,
    stop_kind: str,
    save_every_updates: int,
) -> tuple[Path | None, str | None]:
    if update == stop_target and stop_kind == "stage":
        return checkpoint_dir / f"stage_{update}.pt", "stage"
    if update == stop_target and stop_kind == "benchmark":
        return checkpoint_dir / f"benchmark_{update}.pt", "benchmark"
    if update % save_every_updates == 0:
        return checkpoint_dir / f"model_{update}.pt", "routine"
    return None, None


def publish_durable_checkpoint(
    path: Path,
    *,
    roles: Sequence[str],
    durable_config: dict[str, Any],
    uploader: Callable[..., Any] | None = None,
) -> None:
    """Upload synchronously and convert every failure into a fail-closed stop."""
    upload = uploader or durable_storage.upload_checkpoint
    try:
        upload(
            path,
            roles=roles,
            lock_path=Path(durable_config["lock_path"]),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Durable upload failed for {path}; local checkpoint is preserved and "
            "training is stopping before another optimizer update"
        ) from exc


def configure_benchmark_determinism(
    run_cfg: dict[str, Any],
    *,
    stop_kind: str,
    smoke_one_step: bool,
) -> None:
    if stop_kind != "benchmark":
        return
    if smoke_one_step or run_cfg["runtime"]["device_preference"] != "cuda_then_mps_then_cpu":
        raise ValueError("The bounded benchmark is available only for the production CUDA profile")
    run_cfg["runtime"]["deterministic"] = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _schedule(total_updates: int, warmup_updates: int):
    floor = 1e-8

    def factor(update: int) -> float:
        if warmup_updates > 0 and update < warmup_updates:
            return floor + (1.0 - floor) * update / warmup_updates
        decay_span = max(1, total_updates - warmup_updates)
        return max(floor, 1.0 - (update - warmup_updates) / decay_span)

    return factor


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, ensure_ascii=False, sort_keys=True) + "\n")


def _prepare_rows(cfg: dict[str, Any], split: str, *, limit: int | None = None, max_duration: float | None = None):
    data_cfg = cfg["data"]
    mel_cfg = cfg["model"]["mel"]
    rows, stats = read_manifest(
        data_cfg[f"{split}_manifest"],
        min_duration=data_cfg["min_duration_seconds"],
        max_duration=max_duration or data_cfg["max_duration_seconds"],
        max_text_characters=data_cfg["max_text_characters"],
        target_sample_rate=mel_cfg["target_sample_rate"],
        hop_length=mel_cfg["hop_length"],
        limit=limit,
        verify_paths=True,
    )
    vocab, _ = load_vocab(cfg["upstream"]["vocab"])
    unknown = vocabulary_audit(rows, vocab)
    if unknown and data_cfg["strict_vocabulary"]:
        display = ", ".join(f"{char!r}:{count}" for char, count in unknown.most_common(20))
        raise ValueError(f"{split} manifest contains characters outside pinned SILMA vocab: {display}")
    return rows, stats, unknown


def dry_run(cfg: dict[str, Any], *, check_model: bool, forward_only: bool) -> None:
    materialize_assets(download=False)
    train_rows, train_stats, train_unknown = _prepare_rows(cfg, "train")
    validation_rows, validation_stats, validation_unknown = _prepare_rows(cfg, "validation")
    validate_audio_headers(train_rows, sample_rate=cfg["model"]["mel"]["target_sample_rate"])
    validate_audio_headers(validation_rows, sample_rate=cfg["model"]["mel"]["target_sample_rate"])

    preview_rows = train_rows[: cfg["smoke"]["manifest_rows"]]
    dataset = build_dataset(preview_rows, cfg["model"]["mel"])
    loader = build_dataloader(
        dataset,
        frames_per_batch=cfg["data"]["frames_per_batch"],
        max_samples=cfg["data"]["max_samples_per_batch"],
        seed=cfg["optimization"]["seed"],
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    summary = {
        "status": "ok",
        "train": train_stats,
        "validation": validation_stats,
        "train_oov_types": len(train_unknown),
        "validation_oov_types": len(validation_unknown),
        "batch_mel_shape": list(batch["mel"].shape),
        "batch_mel_lengths": batch["mel_lengths"].tolist(),
        "batch_text": batch["text"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if check_model or forward_only:
        selection = select_device(preference=cfg["runtime"]["device_preference"])
        print(f"device={selection.device} ({selection.reason})")
        model = build_model(cfg, device=selection.device)
        load_pretrained(
            model,
            cfg["upstream"]["checkpoint"],
            weights=cfg["upstream"]["pretrained_weights"],
        )
        print(f"loaded_parameters={sum(parameter.numel() for parameter in model.parameters())}")
        if forward_only:
            model.eval()
            mel = batch["mel"].permute(0, 2, 1).to(selection.device)
            lengths = batch["mel_lengths"].to(selection.device)
            with torch.inference_mode():
                seed_everything(cfg["validation"]["seed"])
                with _autocast(selection.device, cfg["optimization"]["mixed_precision"]):
                    loss, _, _ = model(mel, text=batch["text"], lens=lengths)
            print(f"forward_loss={loss.item():.8f}")


@torch.no_grad()
def validate(
    model,
    loader,
    *,
    device: torch.device,
    max_batches: int,
    seed: int,
    precision: str = "no",
    pin_memory: bool = False,
) -> float:
    cpu_rng = torch.random.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    python_rng = random.getstate()
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        mel = tensor_to_device(batch["mel"].permute(0, 2, 1), device, pin_memory=pin_memory)
        lengths = tensor_to_device(batch["mel_lengths"], device, pin_memory=pin_memory)
        with _autocast(device, precision):
            loss, _, _ = model(mel, text=batch["text"], lens=lengths)
        require_finite_loss(loss, context=f"validation batch {batch_index}")
        losses.append(float(loss.item()))
    model.train()
    torch.random.set_rng_state(cpu_rng)
    random.setstate(python_rng)
    if mps_rng is not None:
        torch.mps.set_rng_state(mps_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return sum(losses) / len(losses)


def train(
    cfg: dict[str, Any],
    *,
    smoke_one_step: bool = False,
    stage_stop_update: int | None = None,
    benchmark_stop_update: int | None = None,
) -> None:
    materialize_assets(download=False)
    run_cfg = copy.deepcopy(cfg)
    if smoke_one_step:
        run_cfg["optimization"]["max_updates"] = 1
        run_cfg["optimization"]["stage_targets"] = [1]
        run_cfg["optimization"]["stage_stop_update"] = 1
        run_cfg["optimization"]["epochs"] = 1
        run_cfg["optimization"]["gradient_accumulation_steps"] = 1
        run_cfg["data"]["max_duration_seconds"] = run_cfg["smoke"]["one_step_max_duration_seconds"]
        run_cfg["checkpointing"]["output_dir"] = run_cfg["smoke"]["output_dir"]
        run_cfg["checkpointing"]["resume"] = "never"
        run_cfg["checkpointing"]["durable_upload"] = {"enabled": False}
        run_cfg["logging"]["jsonl"] = run_cfg["smoke"]["metric_jsonl"]
        run_cfg["data"].pop("expected_eligible_rows", None)
        run_cfg["data"].pop("expected_microbatches_per_epoch", None)
        run_cfg["data"].pop("expected_optimizer_updates_per_epoch", None)
        row_limit = run_cfg["smoke"]["manifest_rows"]
    else:
        row_limit = None

    stop_target, stop_kind = resolve_training_stop(
        run_cfg,
        stage_stop_update=stage_stop_update,
        benchmark_stop_update=benchmark_stop_update,
    )
    configure_benchmark_determinism(
        run_cfg,
        stop_kind=stop_kind,
        smoke_one_step=smoke_one_step,
    )

    seed = int(run_cfg["optimization"]["seed"])
    seed_everything(
        seed,
        deterministic=run_cfg["runtime"].get("deterministic", True),
        strict_determinism=stop_kind == "benchmark",
    )
    selection = select_device(preference=run_cfg["runtime"]["device_preference"])
    if selection.device.type == "cpu" and not run_cfg["runtime"]["allow_cpu_fallback"]:
        raise RuntimeError(selection.reason)
    device = selection.device
    print(f"device={device} ({selection.reason})")
    precision = run_cfg["optimization"]["mixed_precision"]
    if precision == "bf16" and device.type != "cuda":
        raise RuntimeError("BF16 training requires CUDA; refusing an unsafe fallback")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(run_cfg["runtime"].get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(run_cfg["runtime"].get("allow_tf32", True))
        torch.backends.cudnn.deterministic = bool(run_cfg["runtime"].get("deterministic", False))
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
    print(f"precision={precision} distributed=false device={device.type}")

    train_rows, train_stats, _ = _prepare_rows(
        run_cfg, "train", limit=row_limit, max_duration=run_cfg["data"]["max_duration_seconds"]
    )
    validation_rows, validation_stats, _ = _prepare_rows(
        run_cfg, "validation", limit=row_limit, max_duration=run_cfg["data"]["max_duration_seconds"]
    )
    print(f"train_rows={len(train_rows)} validation_rows={len(validation_rows)}")
    print(f"train_filter={train_stats} validation_filter={validation_stats}")

    data_cfg = run_cfg["data"]
    train_loader = build_dataloader(
        build_dataset(train_rows, run_cfg["model"]["mel"]),
        frames_per_batch=data_cfg["frames_per_batch"],
        max_samples=data_cfg["max_samples_per_batch"],
        seed=seed,
        num_workers=data_cfg["num_workers"],
        pin_memory=run_cfg["runtime"]["pin_memory"],
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
    )
    validation_loader = build_dataloader(
        build_dataset(validation_rows, run_cfg["model"]["mel"]),
        frames_per_batch=data_cfg["frames_per_batch"],
        max_samples=data_cfg["max_samples_per_batch"],
        seed=run_cfg["validation"]["seed"],
        num_workers=0,
        pin_memory=False,
    )

    expected_rows = data_cfg.get("expected_eligible_rows")
    if expected_rows is not None and len(train_rows) != int(expected_rows):
        raise RuntimeError(
            f"Eligible train row count changed: expected {expected_rows}, got {len(train_rows)}"
        )
    expected_batches = data_cfg.get("expected_microbatches_per_epoch")
    if expected_batches is not None and len(train_loader) != int(expected_batches):
        raise RuntimeError(
            f"Training sampler changed: expected {expected_batches} microbatches/epoch, "
            f"got {len(train_loader)}"
        )

    sampler_fingerprint = build_sampler_fingerprint(
        train_manifest=resolve_path(data_cfg["train_manifest"]),
        eligible_row_count=len(train_rows),
        seed=seed,
        f5_tts_version=run_cfg["upstream"]["f5_tts_version"],
        frames_threshold=data_cfg["frames_per_batch"],
        max_samples=data_cfg["max_samples_per_batch"],
        gradient_accumulation=run_cfg["optimization"]["gradient_accumulation_steps"],
    )
    print(f"sampler_fingerprint={sampler_fingerprint['digest']}")

    model = build_model(run_cfg, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=run_cfg["optimization"]["learning_rate"],
        weight_decay=run_cfg["optimization"]["weight_decay"],
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_schedule(
            run_cfg["optimization"]["max_updates"], run_cfg["optimization"]["warmup_updates"]
        ),
    )
    checkpoint_dir = resolve_path(run_cfg["checkpointing"]["output_dir"])
    minimum_free_gib = float(run_cfg["runtime"].get("minimum_free_disk_gib", 20))
    free_bytes = ensure_free_space(checkpoint_dir, minimum_free_gib=minimum_free_gib)
    print(f"checkpoint_free_gib={free_bytes / (1024**3):.2f} minimum_required_gib={minimum_free_gib:.2f}")
    last_path = checkpoint_dir / "model_last.pt"
    resume = run_cfg["checkpointing"]["resume"]
    if resume == "auto" and last_path.exists():
        ema = make_ema(model).to(device)
        global_update, saved_epoch, saved_next_batch = load_resume(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            expected_sampler_fingerprint=(
                sampler_fingerprint
                if run_cfg["runtime"]["device_preference"] == "cuda_then_mps_then_cpu"
                else None
            ),
            legacy_migration=run_cfg["checkpointing"].get("legacy_mac_to_cuda_migration"),
        )
        start_epoch, next_batch = normalize_resume_position(
            saved_epoch,
            saved_next_batch,
            len(train_loader),
        )
        if (start_epoch, next_batch) != (saved_epoch, saved_next_batch):
            print(
                "resume_boundary_normalized="
                f"epoch:{saved_epoch}->{start_epoch},next_batch:{saved_next_batch}->0"
            )
        print(f"resumed={last_path} update={global_update} epoch={start_epoch} next_batch={next_batch}")
    else:
        load_pretrained(
            model,
            run_cfg["upstream"]["checkpoint"],
            weights=run_cfg["upstream"]["pretrained_weights"],
        )
        # EMA is created after loading so it reflects the pretrained initialization.
        ema = make_ema(model).to(device)
        global_update, start_epoch, next_batch = 0, 0, 0
        print(f"initialized_from={run_cfg['upstream']['checkpoint']} weights={run_cfg['upstream']['pretrained_weights']}")

    benchmark_source = int(run_cfg.get("benchmark", {}).get("source_update", -1))
    if stop_kind == "benchmark" and global_update != benchmark_source:
        raise RuntimeError(
            f"The deterministic benchmark must resume the authoritative update-{benchmark_source} state; "
            f"found update {global_update}"
        )

    grad_acc = run_cfg["optimization"]["gradient_accumulation_steps"]
    metric_path = resolve_path(run_cfg["logging"]["jsonl"])
    max_updates = run_cfg["optimization"]["max_updates"]
    if global_update >= stop_target:
        print(
            f"{stop_kind}_complete update={global_update} target={stop_target} "
            f"scheduler_horizon={max_updates} checkpoint={last_path}"
        )
        return
    print(f"stop_kind={stop_kind} stop_update={stop_target} scheduler_horizon={max_updates}")

    stop_signal: dict[str, int | None] = {"number": None}

    def request_safe_stop(signum, _frame) -> None:
        stop_signal["number"] = int(signum)
        print(f"stop_requested signal={signum}; checkpointing after the next complete optimizer update")

    signal.signal(signal.SIGTERM, request_safe_stop)
    signal.signal(signal.SIGINT, request_safe_stop)
    optimizer.zero_grad(set_to_none=True)
    stop = False
    started = time.monotonic()

    for epoch in range(start_epoch, run_cfg["optimization"]["epochs"]):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        micro_count = 0
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < next_batch:
                continue
            model.train()
            mel = tensor_to_device(
                batch["mel"].permute(0, 2, 1),
                device,
                pin_memory=run_cfg["runtime"]["pin_memory"],
            )
            lengths = tensor_to_device(
                batch["mel_lengths"],
                device,
                pin_memory=run_cfg["runtime"]["pin_memory"],
            )
            with _autocast(device, precision):
                loss, _, _ = model(mel, text=batch["text"], lens=lengths)
            require_finite_loss(loss, context=f"epoch {epoch} batch {batch_index}")
            (loss / grad_acc).backward()
            micro_count += 1
            if micro_count < grad_acc:
                continue

            grad_norm = clip_gradients(
                model.parameters(),
                max_norm=run_cfg["optimization"]["max_grad_norm"],
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update()
            micro_count = 0
            global_update += 1

            metric = {
                "type": "train",
                "update": global_update,
                "epoch": epoch,
                "batch": batch_index,
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm.item()),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.monotonic() - started,
                "device": device.type,
            }
            _append_metric(metric_path, metric)
            if global_update == 1 or global_update % run_cfg["logging"]["log_every_updates"] == 0:
                print(json.dumps(metric, sort_keys=True))

            if global_update % run_cfg["validation"]["every_updates"] == 0:
                val_loss = validate(
                    model,
                    validation_loader,
                    device=device,
                    max_batches=run_cfg["validation"]["max_batches"],
                    seed=run_cfg["validation"]["seed"],
                    precision=precision,
                    pin_memory=False,
                )
                val_metric = {"type": "validation", "update": global_update, "loss": val_loss}
                _append_metric(metric_path, val_metric)
                print(json.dumps(val_metric, sort_keys=True))

            checkpoint_epoch, checkpoint_next_batch = normalize_resume_position(
                epoch,
                batch_index + 1,
                len(train_loader),
            )
            save_args = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "ema": ema,
                "update": global_update,
                "epoch": checkpoint_epoch,
                "next_batch": checkpoint_next_batch,
                "device": device,
                "sampler_fingerprint": sampler_fingerprint,
                "minimum_free_gib": minimum_free_gib,
            }
            reached_target = global_update == stop_target
            safe_stop = stop_signal["number"] is not None
            immutable_path, artifact_kind = immutable_checkpoint_artifact(
                checkpoint_dir,
                update=global_update,
                stop_target=stop_target,
                stop_kind=stop_kind,
                save_every_updates=run_cfg["checkpointing"]["save_every_updates"],
            )

            if immutable_path is not None:
                save_checkpoint(immutable_path, **save_args)
            if (
                global_update % run_cfg["checkpointing"]["save_last_every_updates"] == 0
                or reached_target
                or safe_stop
            ):
                if immutable_path is not None:
                    link_checkpoint(immutable_path, last_path, minimum_free_gib=minimum_free_gib)
                else:
                    save_checkpoint(last_path, **save_args)
            durable_roles = durable_upload_roles(
                run_cfg["checkpointing"],
                artifact_kind=artifact_kind or "none",
            )
            if immutable_path is not None and durable_roles is not None:
                publish_durable_checkpoint(
                    immutable_path,
                    roles=durable_roles,
                    durable_config=run_cfg["checkpointing"]["durable_upload"],
                )
            if artifact_kind == "routine":
                rotate_checkpoints(checkpoint_dir, run_cfg["checkpointing"]["keep_last_n"])

            if reached_target or safe_stop:
                stop = True
                break

        if micro_count:
            optimizer.zero_grad(set_to_none=True)
            print(f"discarded_partial_accumulation={micro_count} epoch={epoch}")
        next_batch = 0
        if stop:
            break

    if not stop:
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            update=global_update,
            epoch=run_cfg["optimization"]["epochs"],
            next_batch=0,
            device=device,
            sampler_fingerprint=sampler_fingerprint,
            minimum_free_gib=minimum_free_gib,
        )
    reason = "signal" if stop_signal["number"] is not None else stop_kind
    print(
        f"complete update={global_update} checkpoint={last_path} reason={reason} "
        f"scheduler_horizon={max_updates}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Begin the configured full fine-tuning run")
    mode.add_argument("--smoke-one-step", action="store_true", help="Run exactly one optimizer update")
    parser.add_argument("--check-model", action="store_true", help="During dry-run, construct and load the model")
    parser.add_argument("--forward-only", action="store_true", help="During dry-run, execute one no-grad CFM loss")
    stop_mode = parser.add_mutually_exclusive_group()
    stop_mode.add_argument(
        "--stage-stop-update",
        type=int,
        help="Stop at a configured stage gate without changing the 120k LR schedule",
    )
    stop_mode.add_argument(
        "--benchmark-stop-update",
        type=int,
        choices=(200, 300),
        help="Deterministically stop at update 200 or 300 without changing the LR schedule",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.train or args.smoke_one_step:
        train(
            cfg,
            smoke_one_step=args.smoke_one_step,
            stage_stop_update=args.stage_stop_update,
            benchmark_stop_update=args.benchmark_stop_update,
        )
    else:
        dry_run(cfg, check_model=args.check_model or args.forward_only, forward_only=args.forward_only)


if __name__ == "__main__":
    main()
