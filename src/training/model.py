"""Pinned SILMA model construction and checkpoint compatibility helpers."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from ema_pytorch import EMA
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer

from .config import resolve_path

_MEL_BUFFER_KEYS = (
    "mel_spec.mel_stft.mel_scale.fb",
    "mel_spec.mel_stft.spectrogram.window",
)


def build_model(cfg: dict[str, Any], *, device: torch.device | str = "cpu") -> CFM:
    vocab_path = resolve_path(cfg["upstream"]["vocab"])
    vocab_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    mel = dict(cfg["model"]["mel"])
    transformer = DiT(
        **cfg["model"]["architecture"],
        text_num_embeds=vocab_size,
        mel_dim=mel["n_mel_channels"],
    )
    model = CFM(transformer=transformer, mel_spec_kwargs=mel, vocab_char_map=vocab_map)
    return model.to(device=device, dtype=torch.float32)


def _online_state(checkpoint: dict[str, Any], *, use_ema: bool) -> dict[str, torch.Tensor]:
    if use_ema:
        if "ema_model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint has no ema_model_state_dict")
        state = {
            key.removeprefix("ema_model."): value
            for key, value in checkpoint["ema_model_state_dict"].items()
            if key not in {"initted", "step"} and key.startswith("ema_model.")
        }
    else:
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint has no model_state_dict")
        state = dict(checkpoint["model_state_dict"])
    for key in _MEL_BUFFER_KEYS:
        state.pop(key, None)
    return state


def load_pretrained(model: CFM, checkpoint_path: str | Path, *, weights: str = "ema") -> None:
    """Initialize weights only; intentionally start a new optimizer and update count."""
    if weights not in {"ema", "online"}:
        raise ValueError("pretrained weights must be 'ema' or 'online'")
    path = resolve_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = _online_state(checkpoint, use_ema=weights == "ema")
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    gc.collect()


def load_inference_weights(model: CFM, checkpoint_path: str | Path, *, use_ema: bool = True) -> None:
    path = resolve_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = _online_state(checkpoint, use_ema=use_ema)
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    gc.collect()


def make_ema(model: CFM) -> EMA:
    """Use the exact ema-pytorch defaults used by F5-TTS 1.1.7 Trainer."""
    return EMA(model, include_online_model=False)

