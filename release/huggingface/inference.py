"""Reference-conditioned local inference for Saudi TTS V4."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audio", required=True, type=Path)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--nfe-steps", type=int)
    args = parser.parse_args()
    if not args.reference_audio.is_file():
        raise FileNotFoundError(args.reference_audio)

    with (ROOT / "inference_config.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    device = select_device(args.device)
    print(f"device={device}")
    seed_everything(int(cfg["inference"]["seed"]))

    vocab_map, vocab_size = get_tokenizer(str(ROOT / "vocab.txt"), "custom")
    unknown = sorted({char for char in args.reference_text + args.text if char not in vocab_map})
    if unknown:
        raise ValueError(f"Input contains characters outside the model vocabulary: {unknown!r}")
    mel = cfg["model"]["mel"]
    transformer = DiT(
        **cfg["model"]["architecture"],
        text_num_embeds=vocab_size,
        mel_dim=mel["n_mel_channels"],
    )
    model = CFM(transformer=transformer, mel_spec_kwargs=mel, vocab_char_map=vocab_map)
    model.load_state_dict(load_file(ROOT / "model.safetensors", device="cpu"), strict=True)
    model = model.to(device=device, dtype=torch.float32).eval()

    vocos_cfg = cfg["vocos"]
    vocos_dir = snapshot_download(
        repo_id=vocos_cfg["repo_id"],
        revision=vocos_cfg["revision"],
        allow_patterns=("config.yaml", "pytorch_model.bin"),
    )
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=vocos_dir,
        device=str(device),
    )
    reference_audio, reference_text = preprocess_ref_audio_text(
        str(args.reference_audio), args.reference_text
    )
    infer_cfg = cfg["inference"]
    waveform, sample_rate, _ = infer_process(
        reference_audio,
        reference_text,
        args.text,
        model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=args.nfe_steps or int(infer_cfg["nfe_steps"]),
        cfg_strength=float(infer_cfg["cfg_strength"]),
        sway_sampling_coef=float(infer_cfg["sway_sampling_coefficient"]),
        speed=float(infer_cfg["speed"]),
        cross_fade_duration=float(infer_cfg["cross_fade_seconds"]),
        device=str(device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, sample_rate, subtype="PCM_16")
    print(f"wrote {args.output} ({sample_rate} Hz)")


if __name__ == "__main__":
    main()
