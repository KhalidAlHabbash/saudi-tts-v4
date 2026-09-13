from __future__ import annotations

import json
import random

import numpy as np
import pytest
import soundfile as sf
import torch
from ema_pytorch import EMA
from safetensors.torch import save_file as save_safetensors
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.evaluation.phrases import audit_phrases
from src.training.checkpoints import build_sampler_fingerprint, load_resume, save_checkpoint
from src.training.config import load_config, validate_config
from src.training.data import (
    build_dataloader,
    build_dataset,
    load_vocab,
    read_manifest,
    vocabulary_audit,
)
from src.training.train import (
    clip_gradients,
    normalize_resume_position,
    require_finite_loss,
    resolve_stage_stop,
)
from src.training.model import load_inference_weights


def test_inference_loader_accepts_ema_only_safetensors(tmp_path):
    model = torch.nn.Linear(2, 2)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "model.safetensors"
    save_safetensors(expected, path)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    load_inference_weights(model, path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
    with pytest.raises(ValueError, match="EMA weights only"):
        load_inference_weights(model, path, use_ema=False)


def test_pinned_config_and_local_assets_are_consistent():
    cfg = load_config("configs/train.yaml")
    assert cfg["model"]["architecture"]["attn_backend"] == "torch"
    assert cfg["model"]["mel"]["mel_spec_type"] == "vocos"
    assert cfg["data"]["max_samples_per_batch"] == 1
    assert cfg["optimization"]["mixed_precision"] == "no"
    assert cfg["smoke"]["metric_jsonl"] != cfg["logging"]["jsonl"]


def test_runpod_config_preserves_effective_batch_and_requires_cuda():
    local = load_config("configs/train.yaml")
    cloud = load_config("configs/train_runpod.yaml")
    local_effective = local["data"]["max_samples_per_batch"] * local["optimization"]["gradient_accumulation_steps"]
    cloud_effective = cloud["data"]["max_samples_per_batch"] * cloud["optimization"]["gradient_accumulation_steps"]
    assert cloud_effective == local_effective == 8
    assert cloud["optimization"]["mixed_precision"] == "bf16"
    assert cloud["runtime"]["device_preference"] == "cuda_then_mps_then_cpu"
    assert cloud["runtime"]["allow_cpu_fallback"] is False
    assert cloud["checkpointing"]["legacy_mac_to_cuda_migration"] == {
        "enabled": True,
        "allowed_update": 100,
        "allowed_epoch": 0,
        "allowed_next_batch": 800,
        "reset_next_batch": True,
    }
    assert cloud["data"]["expected_eligible_rows"] == 104176
    assert cloud["data"]["expected_microbatches_per_epoch"] == 52088
    assert cloud["data"]["expected_optimizer_updates_per_epoch"] == 13022
    assert cloud["benchmark"] == {"source_update": 100, "allowed_stop_updates": [200, 300]}
    assert cloud["checkpointing"]["durable_upload"]["enabled"] is True
    assert "durable_upload" not in local["checkpointing"]


def test_resume_position_normalizes_exact_epoch_boundary():
    assert normalize_resume_position(0, 52088, 52088) == (1, 0)
    assert normalize_resume_position(3, 52088, 52088) == (4, 0)


def test_resume_position_preserves_in_epoch_progress():
    assert normalize_resume_position(2, 0, 52088) == (2, 0)
    assert normalize_resume_position(2, 1234, 52088) == (2, 1234)


@pytest.mark.parametrize(
    ("epoch", "next_batch", "batches_per_epoch"),
    [(-1, 0, 1), (0, -1, 1), (0, 2, 1), (0, 0, 0)],
)
def test_resume_position_rejects_invalid_state(epoch, next_batch, batches_per_epoch):
    with pytest.raises(ValueError):
        normalize_resume_position(epoch, next_batch, batches_per_epoch)


def test_manifest_dataset_produces_f5_padded_mel(tmp_path):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.zeros(24000, dtype=np.float32), 24000, subtype="PCM_16")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": str(audio_path), "text": "هلا", "duration": 1.0}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows, stats = read_manifest(
        manifest,
        min_duration=1.0,
        max_duration=2.0,
        max_text_characters=20,
    )
    vocab, _ = load_vocab("models/base/silma-tts-v1-ac81834c/vocab.txt")
    assert not vocabulary_audit(rows, vocab)
    mel = load_config("configs/train.yaml")["model"]["mel"]
    loader = build_dataloader(
        build_dataset(rows, mel),
        frames_per_batch=188,
        max_samples=1,
        seed=7,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    assert stats["accepted_rows"] == 1
    assert batch["mel"].shape[:2] == (1, 100)
    assert batch["mel_lengths"].item() > 0
    assert batch["text"] == ["هلا"]


def test_checkpoint_restores_optimizer_scheduler_ema_and_position(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = EMA(model, include_online_model=False)
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    scheduler.last_epoch = 3
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    fingerprint = build_sampler_fingerprint(
        train_manifest=manifest,
        eligible_row_count=1,
        seed=7,
        f5_tts_version="1.1.7",
        frames_threshold=750,
        max_samples=1,
        gradient_accumulation=8,
    )
    path = tmp_path / "model_last.pt"
    random.seed(19)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=3,
        epoch=1,
        next_batch=9,
        device=torch.device("cpu"),
        sampler_fingerprint=fingerprint,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert load_resume(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        device=torch.device("cpu"),
        expected_sampler_fingerprint=fingerprint,
    ) == (3, 1, 9)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_sampler_mismatch_fails_before_state_is_applied(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = EMA(model, include_online_model=False)
    path = tmp_path / "model_last.pt"
    fingerprint = {"format_version": 1, "digest": "a"}
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=10,
        epoch=1,
        next_batch=7,
        device=torch.device("cpu"),
        sampler_fingerprint=fingerprint,
    )
    with torch.no_grad():
        model.weight.fill_(123)
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="Sampler fingerprint mismatch"):
        load_resume(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=torch.device("cpu"),
            expected_sampler_fingerprint={"format_version": 1, "digest": "b"},
        )
    assert torch.equal(model.weight, before)


def test_only_legacy_update100_epoch0_resets_position_once(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = EMA(model, include_online_model=False)
    path = tmp_path / "model_last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=100,
        epoch=0,
        next_batch=800,
        device=torch.device("cpu"),
    )
    policy = {
        "enabled": True,
        "allowed_update": 100,
        "allowed_epoch": 0,
        "allowed_next_batch": 800,
        "reset_next_batch": True,
    }
    assert load_resume(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        device=torch.device("cpu"),
        expected_sampler_fingerprint={"format_version": 1, "digest": "runpod"},
        legacy_migration=policy,
    ) == (100, 0, 0)

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=101,
        epoch=0,
        next_batch=4,
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="no sampler fingerprint"):
        load_resume(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=torch.device("cpu"),
            expected_sampler_fingerprint={"format_version": 1, "digest": "runpod"},
            legacy_migration=policy,
        )


def test_finite_loss_and_gradient_guards_reject_nonfinite_values():
    require_finite_loss(torch.tensor(1.0), context="test")
    with pytest.raises(FloatingPointError, match="Non-finite"):
        require_finite_loss(torch.tensor(float("nan")), context="test")
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_gradients([parameter], max_norm=1.0)


def test_stage_stop_is_separate_from_scheduler_horizon():
    cfg = load_config("configs/train_runpod.yaml")
    optimization = cfg["optimization"]
    assert optimization["max_updates"] == 120000
    assert resolve_stage_stop(optimization) == 13122
    assert resolve_stage_stop(optimization, 26144) == 26144
    with pytest.raises(ValueError, match="not one of"):
        resolve_stage_stop(optimization, 999)


def test_config_rejects_broadened_legacy_migration():
    cfg = load_config("configs/train_runpod.yaml")
    cfg["checkpointing"]["legacy_mac_to_cuda_migration"]["allowed_update"] = 101
    with pytest.raises(ValueError, match="update 100"):
        validate_config(cfg, require_assets=False)


def test_held_out_phrases_have_required_coverage_and_no_exact_overlap():
    result = audit_phrases(
        phrases_path="configs/saudi_eval_phrases.jsonl",
        manifest_paths=["data/splits/train.jsonl", "data/splits/validation.jsonl", "data/splits/test.jsonl"],
        vocab_path="models/base/silma-tts-v1-ac81834c/vocab.txt",
    )
    assert result["phrase_count"] >= 8
    assert result["exact_overlap_count"] == 0
    assert result["oov_characters"] == []
