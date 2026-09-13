"""Synthesize Arabic speech from a project or base SILMA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import yaml
from f5_tts.infer.utils_infer import (
    infer_process,
    load_vocoder,
    preprocess_ref_audio_text,
)

from src.evaluation.phrases import load_phrases
from src.training.config import load_config, resolve_path
from src.training.data import load_vocab
from src.training.model import build_model, load_inference_weights
from src.training.train import seed_everything
from src.utils.device import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--text", help="Text to synthesize; omit with --phrase-id")
    parser.add_argument("--phrase-id", help="ID from the held-out phrase suite")
    parser.add_argument(
        "--reference-audio",
        required=True,
        help="User-supplied consented reference WAV; no reference voice is bundled",
    )
    parser.add_argument(
        "--reference-text",
        required=True,
        help="Exact transcript of --reference-audio",
    )
    parser.add_argument("--output")
    parser.add_argument("--nfe-steps", type=int, help="Override ODE steps for a bounded synthesis smoke test")
    parser.add_argument("--online", action="store_true", help="Use online rather than EMA checkpoint weights")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--device-preference",
        choices=("cuda_then_mps_then_cpu", "mps_then_cpu", "cpu"),
        help="Override the configured device order (useful for RunPod CUDA evaluation)",
    )
    args = parser.parse_args()
    if args.cpu and args.device_preference:
        parser.error("--cpu and --device-preference cannot be used together")

    with resolve_path(args.config).open(encoding="utf-8") as handle:
        eval_cfg = yaml.safe_load(handle)
    train_cfg = load_config("configs/train.yaml")
    checkpoint = resolve_path(args.checkpoint or eval_cfg["checkpoint"])
    reference_audio = resolve_path(args.reference_audio)
    reference_text = args.reference_text
    if args.phrase_id:
        matches = [row for row in load_phrases(eval_cfg["phrases"]) if row["id"] == args.phrase_id]
        if not matches:
            raise ValueError(f"Unknown phrase id: {args.phrase_id}")
        text = matches[0]["text"]
    else:
        text = args.text
    if not text:
        raise ValueError("Provide --text or --phrase-id")
    if not checkpoint.is_file() or not reference_audio.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else reference_audio)

    vocab, _ = load_vocab(eval_cfg["vocab"])
    unknown = sorted({char for char in reference_text + text if char not in vocab})
    if unknown:
        raise ValueError(f"Text contains characters outside the pinned SILMA vocabulary: {unknown!r}")
    device_preference = (
        "cpu"
        if args.cpu
        else args.device_preference or eval_cfg.get("device_preference", "mps_then_cpu")
    )
    selection = select_device(preference=device_preference)
    device = selection.device
    print(f"device={device} ({selection.reason})")
    seed_everything(int(eval_cfg["seed"]))
    model = build_model(train_cfg, device=device)
    load_inference_weights(model, checkpoint, use_ema=not args.online and eval_cfg["use_ema"])
    vocoder_dir = resolve_path(eval_cfg["vocos_dir"])
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=str(vocoder_dir), device=str(device))
    processed_ref, processed_text = preprocess_ref_audio_text(str(reference_audio), reference_text)
    waveform, sample_rate, _ = infer_process(
        processed_ref,
        processed_text,
        text,
        model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=args.nfe_steps or eval_cfg["nfe_steps"],
        cfg_strength=eval_cfg["cfg_strength"],
        sway_sampling_coef=eval_cfg["sway_sampling_coefficient"],
        speed=eval_cfg["speed"],
        cross_fade_duration=eval_cfg["cross_fade_seconds"],
        device=str(device),
    )
    output = resolve_path(args.output or Path(eval_cfg["output_dir"]) / "synthesis.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, waveform, sample_rate, subtype="PCM_16")
    print(json.dumps({"output": str(output), "sample_rate": sample_rate, "text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
