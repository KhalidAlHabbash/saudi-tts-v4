"""Synthesize a reproducible checkpoint-evaluation suite with one model load."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Keep CUDA sampling reproducible across checkpoint stages. This must be set
# before the first CUDA matrix multiplication in the imported TTS stack.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import soundfile as sf
import yaml
from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text

from src.training.config import load_config, resolve_path
from src.training.data import load_vocab
from src.training.model import build_model, load_inference_weights
from src.training.train import seed_everything
from src.utils.device import select_device


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_suite(path: Path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            required = ("id", "complexity", "category", "text")
            if not all(isinstance(row.get(key), str) and row[key] for key in required):
                raise ValueError(f"Invalid prompt at {path}:{line_number}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate prompt id {row['id']!r}")
            seen_ids.add(row["id"])
            prompts.append({key: row[key] for key in required})
    if len(prompts) != 10:
        raise ValueError(f"Checkpoint evaluation suite must contain exactly 10 prompts; got {len(prompts)}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/checkpoint_eval.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--nfe-steps", type=int)
    parser.add_argument("--online", action="store_true", help="Use online rather than EMA weights")
    parser.add_argument(
        "--device-preference",
        choices=("cuda_then_mps_then_cpu", "mps_then_cpu", "cpu"),
        default="cuda_then_mps_then_cpu",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    checkpoint = resolve_path(args.checkpoint)
    reference_audio = resolve_path(args.reference_audio)
    prompt_path = resolve_path(cfg["prompts"])
    output_dir = resolve_path(args.output_dir)
    reference_text = args.reference_text
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not reference_audio.is_file():
        raise FileNotFoundError(reference_audio)

    prompts = load_prompt_suite(prompt_path)
    vocab, _ = load_vocab(cfg["vocab"])
    unknown = sorted(
        {
            character
            for text in [reference_text, *(row["text"] for row in prompts)]
            for character in text
            if character not in vocab
        }
    )
    if unknown:
        raise ValueError(f"Reference or prompt text contains out-of-vocabulary characters: {unknown!r}")

    selection = select_device(preference=args.device_preference)
    device = selection.device
    print(f"device={device} ({selection.reason})")
    train_cfg = load_config(cfg["train_config"])
    model = build_model(train_cfg, device=device)
    use_ema = bool(cfg["use_ema"] and not args.online)
    load_inference_weights(model, checkpoint, use_ema=use_ema)
    vocoder_dir = resolve_path(cfg["vocos_dir"])
    vocoder = load_vocoder(
        vocoder_name="vocos",
        is_local=True,
        local_path=str(vocoder_dir),
        device=str(device),
    )
    processed_ref, processed_text = preprocess_ref_audio_text(str(reference_audio), reference_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    nfe_steps = int(args.nfe_steps or cfg["nfe_steps"])
    base_seed = int(cfg["seed"])
    for index, prompt in enumerate(prompts):
        seed = base_seed + index
        seed_everything(seed)
        waveform, sample_rate, _ = infer_process(
            processed_ref,
            processed_text,
            prompt["text"],
            model,
            vocoder,
            mel_spec_type="vocos",
            nfe_step=nfe_steps,
            cfg_strength=float(cfg["cfg_strength"]),
            sway_sampling_coef=float(cfg["sway_sampling_coefficient"]),
            speed=float(cfg["speed"]),
            cross_fade_duration=float(cfg["cross_fade_seconds"]),
            device=str(device),
        )
        output = output_dir / f"{prompt['id']}.wav"
        sf.write(output, waveform, sample_rate, subtype="PCM_16")
        result = {
            **prompt,
            "seed": seed,
            "output": str(output),
            "sample_rate": int(sample_rate),
            "duration_seconds": len(waveform) / float(sample_rate),
            "sha256": sha256_file(output),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    manifest = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights": "online" if args.online else "ema",
        "reference_audio": str(reference_audio),
        "reference_audio_sha256": sha256_file(reference_audio),
        "reference_text": reference_text,
        "prompt_suite": str(prompt_path),
        "prompt_suite_sha256": sha256_file(prompt_path),
        "device": str(device),
        "nfe_steps": nfe_steps,
        "results": results,
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"complete": str(manifest_path), "clips": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
