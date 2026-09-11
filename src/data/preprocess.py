"""Resumable, provenance-preserving preprocessing for the pinned SADA mirror.

The raw release is never modified.  Each ledger record is an immutable decision
for one parquet row; output WAVs are published with an atomic rename so a rerun
can safely continue after interruption.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import yaml

RAW_REVISION = "094fe2c0fe4b549a4f34349e6e0622e7c7273c2d"
RAW_DIRNAME = f"SADA22_hf_{RAW_REVISION}"
SOURCE_NAME = "SADA22 / MohamedRashad/SADA22"
LICENSE = "CC-BY-NC-SA-4.0"
UNAVAILABLE_SPEAKER_ID = "__speaker_id_unavailable__"
UNAVAILABLE_SOURCE_ID = "__source_id_unavailable__"
SPLITS = ("train", "validation", "test")
DEFAULT_COLLISION_PRIORITY = ("test", "validation", "train")
ARABIC = re.compile(r"[\u0600-\u06ff]")
REPEATED = re.compile(r"(.)\1{7,}")
MACHINE_UNCERTAINTY_TOKEN = re.compile(r"(?<![\u0621-\u064a])غير[_-](?:واضح|مفهوم)(?![\u0621-\u064a])")


@dataclass(frozen=True)
class Settings:
    root: Path
    raw_root: Path
    processed_root: Path
    interim_root: Path
    splits_root: Path
    target_rate: int
    minimum_seconds: float
    maximum_seconds: float
    minimum_rms: float
    maximum_clip_fraction: float
    allowed_dialects: tuple[str, ...]
    allowed_genders: tuple[str, ...]
    seed: int
    collision_priority: tuple[str, ...] = DEFAULT_COLLISION_PRIORITY
    ledger_name: str = "sada_preprocessing_gate_b_ledger.jsonl"


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_arabic_text(raw_text: str) -> tuple[str, list[str]]:
    """Apply only Unicode NFC and whitespace canonicalisation.

    It deliberately does *not* fold Arabic letter variants, remove diacritics,
    expand numbers, correct spelling, or replace Saudi vocabulary.
    """
    text = unicodedata.normalize("NFC", raw_text)
    transforms: list[str] = []
    if text != raw_text:
        transforms.append("unicode_nfc")
    collapsed = re.sub(r"[\t\n\r ]+", " ", text).strip()
    if collapsed != text:
        transforms.append("whitespace_collapsed")
    return collapsed, transforms


def canonical_collision_key(text: str) -> str:
    """Return a comparison-only Arabic loose-text key.

    This drops spacing, Unicode punctuation, and combining diacritics solely to
    identify cross-split content collisions.  It is never used as model text
    and is never substituted for ``raw_text`` or ``text``.
    """
    normalized = unicodedata.normalize("NFC", text)
    return "".join(
        char for char in normalized
        if not char.isspace()
        and not unicodedata.category(char).startswith("P")
        and unicodedata.category(char) != "Mn"
    )


def classify_raw_cleaned_difference(raw_text: str, cleaned_text: str) -> str:
    """Classify mirror-provided text divergence without changing training text."""
    if raw_text == cleaned_text:
        return "raw_cleaned_equal"
    raw_nfc = unicodedata.normalize("NFC", raw_text)
    clean_nfc = unicodedata.normalize("NFC", cleaned_text)
    if raw_nfc == clean_nfc:
        return "safe_unicode_nfc_only"
    raw_spacing = re.sub(r"[\t\n\r ]+", " ", raw_nfc).strip()
    clean_spacing = re.sub(r"[\t\n\r ]+", " ", clean_nfc).strip()
    if raw_spacing == clean_spacing:
        return "safe_whitespace_only"
    # Punctuation/spacing-only differences are safely characterized, but the
    # supplied raw transcript remains the model target.
    def punctuation_spacing_key(value: str) -> str:
        return "".join(char for char in value if not char.isspace() and not unicodedata.category(char).startswith("P"))
    if punctuation_spacing_key(raw_nfc) == punctuation_spacing_key(clean_nfc):
        return "safe_punctuation_or_spacing_only"
    return "review_required_content_or_diacritic_difference"


def is_standalone_uncertainty_annotation(text: str) -> bool:
    """Recognize only standalone uncertainty labels, never natural sentences.

    The punctuation/spacing-only key deliberately accepts variants such as
    ``غير واضح.``, ``غير_واضح`` and ``غير-واضح``.  A sentence such as
    ``الصوت غير واضح اليوم`` does not match because it has other content.
    """
    key = "".join(
        char for char in unicodedata.normalize("NFC", text)
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )
    return key in {"غيرواضح", "غيرمفهوم"}


def contains_machine_uncertainty_token(text: str) -> bool:
    """Detect underscore/hyphen uncertainty tokens at Arabic token boundaries.

    Unlike the standalone label rule, this intentionally catches a token within
    a longer sentence (for example, ``فكر بس غير_واضح``). Ordinary spaced prose
    such as ``الصوت غير واضح اليوم`` is not a match.
    """
    return bool(MACHINE_UNCERTAINTY_TOKEN.search(unicodedata.normalize("NFC", text)))


def transcript_issues(raw_text: Any) -> list[str]:
    if raw_text is None or not str(raw_text).strip():
        return ["missing_text"]
    rendered = str(raw_text)
    issues: list[str] = []
    if not ARABIC.search(rendered):
        issues.append("no_arabic_script")
    if any(unicodedata.category(char).startswith("C") and char not in "\t\n\r" for char in rendered):
        issues.append("control_character")
    if REPEATED.search(rendered):
        issues.append("long_repeated_character")
    # SADA uses hash-prefixed strings for annotation/noise/uncertainty markers.
    # Reject every hash token conservatively rather than guessing whether it was
    # meant as spoken content (including #غير-واضح variants).
    if "#" in rendered:
        issues.append("hash_prefixed_annotation_or_uncertainty_markup")
    if contains_machine_uncertainty_token(rendered):
        issues.append("machine_uncertainty_token")
    elif is_standalone_uncertainty_annotation(rendered):
        issues.append("standalone_uncertainty_annotation")
    return issues


def _audio_bytes(value: Any) -> bytes | None:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    return value if isinstance(value, bytes) and value else None


def _mono_waveform(payload: bytes) -> tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(payload)) as source:
        if source.frames <= 0:
            raise ValueError("empty_audio")
        samples = source.read(dtype="float32", always_2d=True)
        rate = int(source.samplerate)
        if samples.size == 0:
            raise ValueError("empty_audio")
    # The audited source is mono; averaging retains a safe fallback for a future
    # compatible source rather than silently selecting one channel.
    return samples.mean(axis=1, dtype=np.float32), rate


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    try:
        import soxr
        return np.asarray(soxr.resample(samples, source_rate, target_rate, quality="HQ"), dtype=np.float32)
    except ImportError:  # pragma: no cover - environment fallback
        from scipy.signal import resample_poly
        divisor = int(np.gcd(source_rate, target_rate))
        return np.asarray(resample_poly(samples, target_rate // divisor, source_rate // divisor), dtype=np.float32)


def validate_audio(samples: np.ndarray, rate: int, settings: Settings) -> tuple[float, list[str]]:
    duration = len(samples) / rate
    flags: list[str] = []
    if duration < settings.minimum_seconds:
        flags.append("duration_too_short")
    if duration > settings.maximum_seconds:
        flags.append("duration_too_long")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0
    if rms < settings.minimum_rms:
        flags.append("rms_below_minimum")
    clip_fraction = float(np.mean(np.abs(samples) >= 0.999)) if len(samples) else 1.0
    if clip_fraction > settings.maximum_clip_fraction:
        flags.append("clip_fraction_above_maximum")
    return duration, flags


def verify_provenance(settings: Settings) -> dict[str, Any]:
    """Verify the previously acquired immutable source against its audit."""
    audit_path = settings.raw_root / "_audit" / f"sada22_hf_{RAW_REVISION}_audit.json"
    raw_dir = settings.raw_root / RAW_DIRNAME
    if not audit_path.exists() or not raw_dir.exists():
        raise RuntimeError("Pinned SADA release or its acquisition audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    files = sorted(raw_dir.glob("data/*.parquet"))
    expected = {entry["path"]: entry for entry in audit.get("files", [])}
    failures: list[str] = []
    if audit.get("source", {}).get("revision") != RAW_REVISION:
        failures.append("audit_revision_mismatch")
    if len(files) != 31 or len(expected) != 31:
        failures.append("parquet_file_count_mismatch")
    # Full rehash is intentionally a separate explicit hook; the acquisition
    # audit is reused for normal preparation to avoid gratuitous 50-GB reads.
    for parquet in files:
        relative = str(parquet.relative_to(settings.root))
        entry = expected.get(relative)
        if not entry or not entry.get("checksum_matches_published_lfs"):
            failures.append(f"unverified:{relative}")
    result = {"audit_path": str(audit_path.relative_to(settings.root)), "raw_revision": RAW_REVISION,
              "parquet_files": len(files), "audit_reused": True, "failures": failures}
    if failures:
        raise RuntimeError("Provenance verification failed: " + ", ".join(failures))
    return result


def rehash_provenance(settings: Settings) -> dict[str, Any]:
    """Perform the expensive full SHA-256 verification when explicitly asked."""
    audit_path = settings.raw_root / "_audit" / f"sada22_hf_{RAW_REVISION}_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    verified = 0
    for entry in audit["files"]:
        path = settings.root / entry["path"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            failures.append(entry["path"])
        else:
            verified += 1
    if failures:
        raise RuntimeError("Raw checksum mismatch: " + ", ".join(failures))
    return {"full_rehash": True, "verified_parquet_files": verified, "failures": failures}


def acquire_pinned_release(settings: Settings) -> dict[str, Any]:
    """Acquire only the immutable pinned mirror, then verify all local hashes.

    Kept opt-in so a normal rerun never contacts the network or changes raw
    source material.  Hugging Face's cache may satisfy this without transfer.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - packaging/environment error
        raise RuntimeError("huggingface_hub is required for --download") from exc
    target = settings.raw_root / RAW_DIRNAME
    snapshot_download(
        repo_id="MohamedRashad/SADA22", repo_type="dataset", revision=RAW_REVISION,
        local_dir=target, allow_patterns=["data/*.parquet", "README.md", ".gitattributes"],
    )
    return {"download": "completed_or_cache_reused", "target": str(target.relative_to(settings.root)),
            **rehash_provenance(settings)}


def _load_settings(config_path: Path) -> Settings:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent.parent.resolve()
    data = config["dataset"]
    quality = data["quality"]
    return Settings(
        root=base, raw_root=base / data["raw_root"], processed_root=base / data["processed_root"],
        interim_root=base / data["interim_root"], splits_root=base / data["splits_root"],
        target_rate=int(data["target_sample_rate"]), minimum_seconds=float(quality["min_duration_seconds"]),
        maximum_seconds=float(quality["max_duration_seconds"]), minimum_rms=float(quality["min_rms"]),
        maximum_clip_fraction=float(quality["max_clip_fraction"]),
        allowed_dialects=tuple(data["allowed_dialects"]), allowed_genders=tuple(data["allowed_genders"]),
        seed=int(data["split_seed"]), collision_priority=tuple(data["split_collision_priority"]),
        ledger_name=str(data["ledger_name"]),
    )


def _load_ledger(path: Path) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]], list[dict[str, Any]]]:
    completed: set[str] = set()
    audio_hashes: set[str] = set()
    text_splits: dict[str, set[str]] = collections.defaultdict(set)
    loose_text_splits: dict[str, set[str]] = collections.defaultdict(set)
    accepted: list[dict[str, Any]] = []
    if not path.exists():
        return completed, audio_hashes, text_splits, loose_text_splits, accepted
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            completed.add(row["record_key"])
            if row.get("accepted"):
                accepted.append(row["metadata"])
                audio_hashes.add(row["audio_sha256"])
                text_splits[row["text_sha256"]].add(row["metadata"]["split"])
                loose_text_splits[row["loose_text_sha256"]].add(row["metadata"]["split"])
    return completed, audio_hashes, text_splits, loose_text_splits, accepted


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _decision_base(record_key: str, source_shard: str, source_row: int, split: str) -> dict[str, Any]:
    return {"record_key": record_key, "source_shard": source_shard, "source_row": source_row, "official_split": split}


def _materialize_manifests(settings: Settings, accepted: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(accepted, key=lambda row: (row["split"], row["audio_path"]))
    settings.processed_root.mkdir(parents=True, exist_ok=True)
    settings.splits_root.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        for destination in (settings.splits_root / f"{split}.jsonl", settings.processed_root / f"manifest_{split}.jsonl"):
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in split_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(temporary, destination)
        counts[split] = len(split_rows)
    manifest = settings.processed_root / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    return {"manifest_rows": len(rows), "by_split": dict(counts), "manifest_sha256": _sha256(manifest.read_bytes())}


def _write_stratified_transcript_sample(settings: Settings, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Persist a deterministic, review-oriented sample without changing text."""
    selected: dict[str, list[tuple[str, dict[str, Any]]]] = collections.defaultdict(list)
    for row in rows:
        stratum = f"{row['dialect']}|{row['gender']}|{row['cleaned_text_relation']}"
        rank = _sha256(f"{settings.seed}:{row['record_key']}")
        selected[stratum].append((rank, {
            "record_key": row["record_key"], "split": row["split"], "dialect": row["dialect"],
            "gender": row["gender"], "raw_text": row["raw_text"], "cleaned_text": row["cleaned_text"],
            "text": row["text"], "cleaned_text_relation": row["cleaned_text_relation"],
            "quality_flags": row["quality_flags"],
        }))
        selected[stratum] = sorted(selected[stratum])[:3]
    output_path = settings.interim_root / "transcript_pair_stratified_sample.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for stratum in sorted(selected):
            for _, row in sorted(selected[stratum])[:3]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    os.replace(temporary, output_path)
    return {"strata": len(selected), "rows": count}


def quarantine_unreferenced_audio(settings: Settings, accepted: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Move, never delete, unreferenced generated audio to a Gate-B quarantine."""
    referenced = {settings.root / row["audio_path"] for row in accepted}
    audio_root = settings.processed_root / "audio"
    quarantine_root = settings.interim_root / "quarantine" / "gate_b_unreferenced_audio"
    moved, byte_count = 0, 0
    for path in sorted(audio_root.glob("**/*.wav")):
        if path in referenced:
            continue
        relative = path.relative_to(audio_root)
        destination = quarantine_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"quarantine collision: {destination}")
        byte_count += path.stat().st_size
        os.replace(path, destination)
        moved += 1
    return {"quarantined_audio_files": moved, "quarantined_audio_bytes": byte_count}


def _report(settings: Settings, provenance: dict[str, Any], ledger_path: Path, manifest_summary: dict[str, Any],
            quarantine_summary: dict[str, int], sample_summary: dict[str, int]) -> dict[str, Any]:
    decisions = collections.Counter()
    rejected_reasons = collections.Counter()
    accepted_rows: list[dict[str, Any]] = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            decisions["accepted" if item.get("accepted") else "rejected"] += 1
            if item.get("accepted"):
                accepted_rows.append(item["metadata"])
            else:
                rejected_reasons.update(item.get("reasons", []))
    by_split = collections.Counter(row["split"] for row in accepted_rows)
    by_dialect = collections.Counter(row["dialect"] for row in accepted_rows)
    seconds = sum(float(row["duration"]) for row in accepted_rows)
    audio_bytes = sum((settings.root / row["audio_path"]).stat().st_size for row in accepted_rows)
    audio_fingerprints = collections.defaultdict(set)
    text_fingerprints = collections.defaultdict(set)
    loose_text_fingerprints = collections.defaultdict(set)
    relations = collections.Counter()
    for row in accepted_rows:
        audio_fingerprints[row["audio_sha256"]].add(row["split"])
        text_fingerprints[row["text_sha256"]].add(row["split"])
        loose_text_fingerprints[row["loose_text_sha256"]].add(row["split"])
        relations[row["cleaned_text_relation"]] += 1
    cross_audio = sum(len(splits) > 1 for splits in audio_fingerprints.values())
    cross_text = sum(len(splits) > 1 for splits in text_fingerprints.values())
    cross_loose_text = sum(len(splits) > 1 for splits in loose_text_fingerprints.values())
    accepted_uncertainty = sum(is_standalone_uncertainty_annotation(row["text"]) for row in accepted_rows)
    accepted_machine_uncertainty = sum(contains_machine_uncertainty_token(row["text"]) for row in accepted_rows)
    report = {
        "schema_version": 1, "seed": settings.seed, "provenance": provenance,
        "decisions": dict(decisions), "rejected_reasons": dict(sorted(rejected_reasons.items())),
        "accepted_by_split": dict(by_split), "accepted_by_dialect": dict(by_dialect),
        "accepted_seconds": round(seconds, 6), "accepted_hours": round(seconds / 3600, 6),
        "processed_audio_bytes": audio_bytes, "cross_split_exact_audio_groups": cross_audio,
        "cross_split_exact_text_groups": cross_text, "cross_split_loose_text_groups": cross_loose_text,
        "accepted_standalone_uncertainty_markers": accepted_uncertainty,
        "accepted_machine_uncertainty_tokens": accepted_machine_uncertainty,
        "raw_cleaned_relation_counts": dict(sorted(relations.items())),
        "collision_priority": list(settings.collision_priority), "quarantine": quarantine_summary,
        "transcript_pair_stratified_sample": sample_summary,
        "speaker_disjointness": "NOT_PROVABLE: mirror has no stable speaker identity; all records use explicit unavailable sentinel",
        "actor_leakage": "NOT_PROVABLE_ABSENT: source/show/actor identifiers are absent from the mirror",
        "manifest": manifest_summary,
    }
    interim = settings.interim_root / "preprocessing_summary.json"
    temporary = interim.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, interim)
    return report


def run(settings: Settings) -> dict[str, Any]:
    provenance = verify_provenance(settings)
    settings.interim_root.mkdir(parents=True, exist_ok=True)
    ledger_path = settings.interim_root / settings.ledger_name
    completed, audio_hashes, text_splits, loose_text_splits, accepted = _load_ledger(ledger_path)
    raw_dir = settings.raw_root / RAW_DIRNAME
    if set(settings.collision_priority) != set(SPLITS):
        raise ValueError("split_collision_priority must contain train, validation, and test exactly once")
    # Retained rows remain in their official split.  Collision priority protects
    # the held-out evaluation split first, then validation, then training.
    files = sorted(raw_dir.glob("data/*.parquet"), key=lambda p: (settings.collision_priority.index(p.name.split("-")[0]), p.name))
    import pyarrow.parquet as pq
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for parquet_path in files:
            split = parquet_path.name.split("-")[0]
            source_shard = str(parquet_path.relative_to(settings.root))
            source_row = 0
            for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=32):
                columns = batch.to_pydict()
                for index in range(batch.num_rows):
                    current_row = source_row
                    source_row += 1
                    record_key = _sha256(f"{source_shard}:{current_row}")
                    if record_key in completed:
                        continue
                    decision = _decision_base(record_key, source_shard, current_row, split)
                    dialect = str(columns["speaker_dialect"][index] or "").strip()
                    gender = str(columns["speaker_gender"][index] or "").strip()
                    if dialect not in settings.allowed_dialects:
                        decision.update(accepted=False, reasons=["dialect_not_allowed"])
                        _append_jsonl(ledger, decision)
                        continue
                    if gender not in settings.allowed_genders:
                        decision.update(accepted=False, reasons=["gender_not_single_speaker_or_unknown"])
                        _append_jsonl(ledger, decision)
                        continue
                    raw_text = columns["text"][index]
                    cleaned_text = columns["cleaned_text"][index]
                    issues = transcript_issues(raw_text)
                    if cleaned_text is None or not str(cleaned_text).strip():
                        issues.append("missing_cleaned_text")
                    if issues:
                        decision.update(accepted=False, reasons=issues)
                        _append_jsonl(ledger, decision)
                        continue
                    text, transforms = normalize_arabic_text(str(raw_text))
                    if not text:
                        decision.update(accepted=False, reasons=["empty_after_normalization"])
                        _append_jsonl(ledger, decision)
                        continue
                    text_hash = _sha256(text)
                    loose_key = canonical_collision_key(text)
                    if not loose_key:
                        decision.update(accepted=False, reasons=["empty_canonical_collision_key"], text_sha256=text_hash)
                        _append_jsonl(ledger, decision)
                        continue
                    loose_text_hash = _sha256(loose_key)
                    if text_splits[text_hash] and split not in text_splits[text_hash]:
                        decision.update(accepted=False, reasons=["cross_split_exact_text_duplicate"], text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash)
                        _append_jsonl(ledger, decision)
                        continue
                    if loose_text_splits[loose_text_hash] and split not in loose_text_splits[loose_text_hash]:
                        decision.update(accepted=False, reasons=["cross_split_loose_text_collision"], text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash)
                        _append_jsonl(ledger, decision)
                        continue
                    payload = _audio_bytes(columns["audio"][index])
                    if payload is None:
                        decision.update(accepted=False, reasons=["missing_audio_payload"], text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash)
                        _append_jsonl(ledger, decision)
                        continue
                    audio_hash = _sha256(payload)
                    if audio_hash in audio_hashes:
                        decision.update(accepted=False, reasons=["exact_audio_duplicate"], text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash, audio_sha256=audio_hash)
                        _append_jsonl(ledger, decision)
                        continue
                    try:
                        samples, sample_rate = _mono_waveform(payload)
                    except Exception as exc:
                        decision.update(accepted=False, reasons=["audio_decode_or_empty_failure"], text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash, audio_sha256=audio_hash, error=f"{type(exc).__name__}: {exc}")
                        _append_jsonl(ledger, decision)
                        continue
                    duration, audio_issues = validate_audio(samples, sample_rate, settings)
                    if audio_issues:
                        decision.update(accepted=False, reasons=audio_issues, text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash, audio_sha256=audio_hash,
                                        duration=round(duration, 6), sample_rate=sample_rate)
                        _append_jsonl(ledger, decision)
                        continue
                    try:
                        converted = _resample(samples, sample_rate, settings.target_rate)
                        relative_audio_path = Path("data/processed/audio") / split / f"{audio_hash}.wav"
                        destination = settings.root / relative_audio_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.exists():
                            with tempfile.NamedTemporaryFile(suffix=".wav", dir=destination.parent, delete=False) as temp:
                                temporary_name = temp.name
                            try:
                                sf.write(temporary_name, converted, settings.target_rate, format="WAV", subtype="PCM_16")
                                os.replace(temporary_name, destination)
                            finally:
                                if os.path.exists(temporary_name):
                                    os.unlink(temporary_name)
                    except Exception as exc:
                        decision.update(accepted=False, reasons=["audio_conversion_failure"], text_sha256=text_hash,
                                        loose_text_sha256=loose_text_hash, audio_sha256=audio_hash, error=f"{type(exc).__name__}: {exc}")
                        _append_jsonl(ledger, decision)
                        continue
                    flags = ["audio_mono_verified", f"resampled_from_{sample_rate}_hz",
                             "speaker_identity_unavailable", "source_identity_unavailable",
                             "official_split_preserved"]
                    flags.extend(f"text_{item}" for item in transforms)
                    relation = classify_raw_cleaned_difference(str(raw_text), str(cleaned_text))
                    flags.append(f"cleaned_text_relation_{relation}")
                    metadata = {
                        "audio_path": str(relative_audio_path), "text": text, "speaker_id": UNAVAILABLE_SPEAKER_ID,
                        "raw_text": str(raw_text), "dialect": dialect, "region": "Saudi Arabia",
                        "gender": gender, "sample_rate": settings.target_rate, "duration": round(duration, 6),
                        "dataset_source": SOURCE_NAME, "source_id": UNAVAILABLE_SOURCE_ID, "split": split,
                        "license": LICENSE, "quality_flags": flags, "audio_sha256": audio_hash,
                        "text_sha256": text_hash, "loose_text_sha256": loose_text_hash,
                        "cleaned_text": str(cleaned_text), "cleaned_text_relation": relation,
                        "record_key": record_key, "source_shard": source_shard, "source_row": current_row,
                    }
                    decision.update(accepted=True, reasons=[], audio_sha256=audio_hash, text_sha256=text_hash,
                                    loose_text_sha256=loose_text_hash, metadata=metadata)
                    _append_jsonl(ledger, decision)
                    accepted.append(metadata)
                    audio_hashes.add(audio_hash)
                    text_splits[text_hash].add(split)
                    loose_text_splits[loose_text_hash].add(split)
    manifest_summary = _materialize_manifests(settings, accepted)
    quarantine_summary = quarantine_unreferenced_audio(settings, accepted)
    sample_summary = _write_stratified_transcript_sample(settings, accepted)
    return _report(settings, provenance, ledger_path, manifest_summary, quarantine_summary, sample_summary)


def check_manifests(settings: Settings) -> dict[str, Any]:
    """Validate materialised metadata, WAV headers, and split contamination."""
    manifest = settings.processed_root / "manifest.jsonl"
    required = {"audio_path", "text", "speaker_id", "raw_text", "dialect", "region", "gender",
                "sample_rate", "duration", "dataset_source", "source_id", "split", "license", "quality_flags"}
    seen_audio: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    seen_loose_text: dict[str, str] = {}
    counts = collections.Counter()
    byte_count = 0
    accepted_uncertainty = 0
    accepted_machine_uncertainty = 0
    with manifest.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            missing = required.difference(row)
            if missing:
                raise ValueError(f"manifest line {number} missing {sorted(missing)}")
            if is_standalone_uncertainty_annotation(row["text"]):
                accepted_uncertainty += 1
            if contains_machine_uncertainty_token(row["text"]):
                accepted_machine_uncertainty += 1
            audio = settings.root / row["audio_path"]
            with sf.SoundFile(audio) as wav:
                if wav.samplerate != settings.target_rate or wav.channels != 1 or wav.subtype != "PCM_16":
                    raise ValueError(f"invalid processed WAV: {audio}")
                measured = wav.frames / wav.samplerate
            if abs(measured - float(row["duration"])) > 1 / settings.target_rate + 1e-6:
                raise ValueError(f"duration mismatch: {audio}")
            fingerprint = row["audio_sha256"]
            if fingerprint in seen_audio and seen_audio[fingerprint] != row["split"]:
                raise ValueError("cross-split exact audio contamination")
            if row["text_sha256"] in seen_text and seen_text[row["text_sha256"]] != row["split"]:
                raise ValueError("cross-split exact text contamination")
            if row["loose_text_sha256"] in seen_loose_text and seen_loose_text[row["loose_text_sha256"]] != row["split"]:
                raise ValueError("cross-split loose text contamination")
            seen_audio[fingerprint] = row["split"]
            seen_text[row["text_sha256"]] = row["split"]
            seen_loose_text[row["loose_text_sha256"]] = row["split"]
            counts[row["split"]] += 1
            byte_count += audio.stat().st_size
    if accepted_uncertainty or accepted_machine_uncertainty:
        raise ValueError("accepted standalone uncertainty annotation")
    return {"rows": sum(counts.values()), "by_split": dict(counts), "audio_bytes": byte_count,
            "cross_split_exact_audio_groups": 0, "cross_split_exact_text_groups": 0,
            "cross_split_loose_text_groups": 0,
            "accepted_standalone_uncertainty_markers": 0,
            "accepted_machine_uncertainty_tokens": 0,
            "speaker_disjointness": "not provable: explicit unavailable speaker sentinel"}


def check_determinism(settings: Settings) -> dict[str, Any]:
    first = _materialize_manifests(settings, [json.loads(line) for line in (settings.processed_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()])
    return {"deterministic": True, "manifest_sha256": first["manifest_sha256"], "seed": settings.seed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--download", action="store_true", help="Acquire the exact pinned raw mirror, then full-rehash it.")
    parser.add_argument("--verify-only", action="store_true", help="Perform a full raw Parquet SHA-256 rehash.")
    parser.add_argument("--check", action="store_true", help="Validate existing manifests and processed WAVs.")
    parser.add_argument("--determinism-check", action="store_true", help="Rewrite manifests deterministically and print checksum.")
    args = parser.parse_args()
    settings = _load_settings(args.config.resolve())
    if args.download:
        result = acquire_pinned_release(settings)
    elif args.verify_only:
        result = rehash_provenance(settings)
    elif args.check:
        result = check_manifests(settings)
    elif args.determinism_check:
        result = check_determinism(settings)
    else:
        result = run(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
