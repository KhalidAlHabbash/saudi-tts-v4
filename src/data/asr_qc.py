"""Diagnostic-only, deterministic Whisper transcript QC for processed SADA."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .preprocess import _load_settings

MODEL_ID = "openai/whisper-small"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
QC_SEED = 20260911
QC_DURATION_RANGE = (2.0, 8.0)
SPLITS = ("train", "validation", "test")


def normalized_metric_text(text: str) -> str:
    """Comparison-only normalization; never used to change a corpus transcript."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn" and not unicodedata.category(char).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for index, item in enumerate(reference, 1):
        current = [index]
        for other_index, other in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1,
                               previous[other_index - 1] + (item != other)))
        previous = current
    return previous[-1]


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_normalized = normalized_metric_text(reference)
    hypothesis_normalized = normalized_metric_text(hypothesis)
    char_distance = levenshtein(list(reference_normalized), list(hypothesis_normalized))
    word_distance = levenshtein(reference_normalized.split(), hypothesis_normalized.split())
    return {
        "reference_metric_text": reference_normalized,
        "asr_metric_text": hypothesis_normalized,
        "character_error_rate": char_distance / max(1, len(reference_normalized)),
        "word_error_rate": word_distance / max(1, len(reference_normalized.split())),
        "character_similarity": 1 - char_distance / max(1, max(len(reference_normalized), len(hypothesis_normalized))),
        "character_edit_distance": char_distance,
        "word_edit_distance": word_distance,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_sample(rows: Iterable[dict[str, Any]], *, per_stratum: int = 2) -> list[dict[str, Any]]:
    """Choose the lexicographically smallest seeded ranks in each final stratum."""
    selected: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = collections.defaultdict(list)
    for row in rows:
        if not QC_DURATION_RANGE[0] <= float(row["duration"]) <= QC_DURATION_RANGE[1]:
            continue
        key = (row["dialect"], row["gender"], row["split"])
        rank = hashlib.sha256(f"{QC_SEED}:{row['record_key']}".encode()).hexdigest()
        selected[key].append((rank, row))
        selected[key] = sorted(selected[key], key=lambda item: item[0])[:per_stratum]
    return [row for key in sorted(selected) for _, row in sorted(selected[key], key=lambda item: item[0])]


def _resample_to_16k(path: Path) -> np.ndarray:
    waveform, rate = sf.read(path, dtype="float32", always_2d=False)
    if waveform.ndim != 1:
        waveform = waveform.mean(axis=1)
    if rate == 16000:
        return waveform
    divisor = int(np.gcd(rate, 16000))
    return np.asarray(resample_poly(waveform, 16000 // divisor, rate // divisor), dtype=np.float32)


def model_provenance(snapshot: Path) -> dict[str, Any]:
    files = []
    for name in ("model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = snapshot / name
        if path.exists():
            files.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": str(snapshot), "files": files}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in results:
        groups["overall"].append(row)
        groups[f"dialect={row['dialect']}"].append(row)
        groups[f"gender={row['gender']}"] .append(row)
        groups[f"split={row['split']}"] .append(row)
        groups[f"{row['dialect']}|{row['gender']}|{row['split']}"] .append(row)
    output: dict[str, Any] = {}
    for key, values in sorted(groups.items()):
        output[key] = {
            "samples": len(values),
            "mean_character_error_rate": round(float(np.mean([item["character_error_rate"] for item in values])), 6),
            "mean_word_error_rate": round(float(np.mean([item["word_error_rate"] for item in values])), 6),
            "mean_character_similarity": round(float(np.mean([item["character_similarity"] for item in values])), 6),
        }
    return output


def run(config: Path, *, per_stratum: int = 2) -> dict[str, Any]:
    settings = _load_settings(config.resolve())
    rows = [json.loads(line) for line in (settings.processed_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = select_sample(rows, per_stratum=per_stratum)
    missing = {(dialect, gender, split) for dialect in ("Najdi", "Hijazi", "Khaliji") for gender in ("Female", "Male") for split in SPLITS}
    missing.difference_update((row["dialect"], row["gender"], row["split"]) for row in sample)
    if missing:
        raise RuntimeError(f"sample lacks expected final strata: {sorted(missing)}")
    # Do not ask the Hub to validate unrelated optional framework files.  The
    # explicitly pinned inference files were fetched before this offline run.
    from huggingface_hub.constants import HF_HUB_CACHE
    snapshot = Path(HF_HUB_CACHE) / "models--openai--whisper-small" / "snapshots" / MODEL_REVISION
    if not (snapshot / "model.safetensors").exists():
        raise RuntimeError(f"Pinned Whisper checkpoint is unavailable at {snapshot}")
    provenance = model_provenance(snapshot)
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    model = AutoModelForSpeechSeq2Seq.from_pretrained(snapshot, local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    recognizer = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                          feature_extractor=processor.feature_extractor, device=-1)
    results: list[dict[str, Any]] = []
    for row in sample:
        output = recognizer({"raw": _resample_to_16k(settings.root / row["audio_path"]), "sampling_rate": 16000},
                            generate_kwargs={"language": "arabic", "task": "transcribe"})
        asr_text = str(output["text"]).strip()
        results.append({
            "record_key": row["record_key"], "audio_path": row["audio_path"], "split": row["split"],
            "dialect": row["dialect"], "gender": row["gender"], "duration": row["duration"],
            "reference_text": row["text"], "asr_text": asr_text, **score_pair(row["text"], asr_text),
        })
    payload = {
        "purpose": "diagnostic_asr_qc_only_not_transcript_correction_or_filtering",
        "seed": QC_SEED, "selection": {"per_dialect_gender_split": per_stratum,
        "duration_seconds": list(QC_DURATION_RANGE), "strata": 18}, "model": provenance,
        "metric_policy": "NFC, punctuation, spacing, and combining-diacritic insensitive comparison only; dialectal ASR errors are not ground truth",
        "summary": summarize(results), "samples": results,
    }
    output_path = settings.root / "reports" / "asr_quality_sample.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()
    payload = run(args.config, per_stratum=args.per_stratum)
    print(json.dumps({"samples": len(payload["samples"]), "summary": payload["summary"]["overall"], "model": payload["model"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
