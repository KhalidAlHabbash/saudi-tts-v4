"""JSONL manifest loading with F5-compatible mel collation and duration batches."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf
from f5_tts.model.dataset import CustomDataset, DynamicBatchSampler, collate_fn
from torch.utils.data import DataLoader, SequentialSampler

from .config import resolve_path


def load_vocab(path: str | Path) -> tuple[dict[str, int], list[str]]:
    vocab_path = resolve_path(path)
    tokens = vocab_path.read_text(encoding="utf-8").splitlines()
    if not tokens or tokens[0] != " ":
        raise ValueError("Pinned F5 character vocabulary must contain space at index zero")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate entries in vocabulary: {vocab_path}")
    return {token: index for index, token in enumerate(tokens)}, tokens


def read_manifest(
    path: str | Path,
    *,
    min_duration: float,
    max_duration: float,
    max_text_characters: int,
    target_sample_rate: int | None = None,
    hop_length: int | None = None,
    limit: int | None = None,
    verify_paths: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest_path = resolve_path(path)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            stats["manifest_rows"] += 1
            missing = {key for key in ("audio_path", "text", "duration") if key not in row}
            if missing:
                raise ValueError(f"{manifest_path}:{line_number}: missing {sorted(missing)}")
            duration = float(row["duration"])
            if not min_duration <= duration <= max_duration:
                stats["duration_filtered"] += 1
                continue
            if not row["text"] or len(row["text"]) > max_text_characters:
                stats["text_length_filtered"] += 1
                continue
            if target_sample_rate and hop_length:
                mel_frames = int(duration * target_sample_rate / hop_length)
                if len(row["text"]) > mel_frames:
                    # F5's TextEmbedding otherwise truncates text to mel length silently.
                    stats["text_exceeds_mel_frames"] += 1
                    continue
            audio_path = resolve_path(row["audio_path"])
            if verify_paths and not audio_path.is_file():
                raise FileNotFoundError(f"{manifest_path}:{line_number}: {audio_path}")
            normalized = dict(row)
            normalized["audio_path"] = str(audio_path)
            normalized["duration"] = duration
            rows.append(normalized)
            stats["accepted_rows"] += 1
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No usable rows in {manifest_path}")
    return rows, dict(stats)


def vocabulary_audit(rows: list[dict[str, Any]], vocab: dict[str, int]) -> Counter[str]:
    unknown: Counter[str] = Counter()
    for row in rows:
        unknown.update(char for char in row["text"] if char not in vocab)
    return unknown


def validate_audio_headers(rows: list[dict[str, Any]], *, sample_rate: int, limit: int = 4) -> None:
    for row in rows[:limit]:
        info = sf.info(row["audio_path"])
        if info.samplerate != sample_rate or info.channels != 1 or info.frames <= 0:
            raise ValueError(
                f"Invalid processed WAV {row['audio_path']}: "
                f"sample_rate={info.samplerate}, channels={info.channels}, frames={info.frames}"
            )


def build_dataset(rows: list[dict[str, Any]], mel: dict[str, Any]) -> CustomDataset:
    return CustomDataset(rows, durations=[row["duration"] for row in rows], **mel)


def build_dataloader(
    dataset: CustomDataset,
    *,
    frames_per_batch: int,
    max_samples: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int = 2,
) -> DataLoader:
    sampler = DynamicBatchSampler(
        SequentialSampler(dataset),
        frames_threshold=frames_per_batch,
        max_samples=max_samples,
        random_seed=seed,
        drop_residual=False,
    )
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
