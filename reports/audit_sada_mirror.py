#!/usr/bin/env python3
"""Read-only forensic audit for the pinned public SADA Parquet mirror.

Reads only data/raw/SADA22_hf_* and writes JSON results under data/raw/_audit.
It never extracts, edits, normalizes, or rewrites source audio/transcripts.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:
    import pyarrow.parquet as pq
    import soundfile as sf
except ImportError as exc:
    raise SystemExit(f"Missing temporary audit dependency: {exc}")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/SADA22_hf_094fe2c0fe4b549a4f34349e6e0622e7c7273c2d"
OUT = ROOT / "data/raw/_audit/sada22_hf_094fe2c0fe4b549a4f34349e6e0622e7c7273c2d_audit.json"
TREE_MANIFEST = Path("/private/tmp/sada22_hf_tree.json")
EXPECTED_FILES = 31
EXPECTED_ROWS = {"train": 241834, "validation": 5139, "test": 6193}
CORE_DIALECTS = ("najd", "hijaz", "khalij", "khalee", "gulf")
ARABIC = re.compile(r"[\u0600-\u06ff]")
REPEATED = re.compile(r"(.)\1{7,}")


def clean_value(value):
    if value is None:
        return "<null>"
    value = str(value).strip()
    return value or "<empty>"


def audio_payload(value):
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    return value if isinstance(value, bytes) else None


def audio_header(payload):
    with sf.SoundFile(io.BytesIO(payload)) as f:
        frames = int(f.frames)
        rate = int(f.samplerate)
        channels = int(f.channels)
        return {"frames": frames, "sample_rate": rate, "channels": channels,
                "format": f.format, "subtype": f.subtype}


def content_metrics(payload):
    header = audio_header(payload)
    with sf.SoundFile(io.BytesIO(payload)) as f:
        frames = header["frames"]
        rate = header["sample_rate"]
        channels = header["channels"]
        # A deterministic, bounded inspection of actual sample values.
        samples = f.read(min(frames, max(rate * 10, 1)), dtype="float32", always_2d=True)
    if len(samples) == 0:
        return {"frames": frames, "sample_rate": rate, "channels": channels,
                "peak": 0.0, "rms": 0.0, "near_silent_fraction": 1.0}
    import math
    flat = samples.ravel()
    peak = max(abs(float(x)) for x in flat)
    mean_square = sum(float(x) * float(x) for x in flat) / len(flat)
    near_silent = sum(abs(float(x)) < 0.001 for x in flat) / len(flat)
    return {"frames": frames, "sample_rate": rate, "channels": channels,
            "peak": round(peak, 7), "rms": round(math.sqrt(mean_square), 7),
            "near_silent_fraction": round(near_silent, 7)}


def anomaly_codes(text, cleaned):
    codes = []
    if text is None or not str(text).strip():
        codes.append("missing_text")
    else:
        rendered = str(text)
        if not ARABIC.search(rendered):
            codes.append("no_arabic_script")
        if any(unicodedata.category(c).startswith("C") and c not in "\t\n\r" for c in rendered):
            codes.append("control_character")
        if REPEATED.search(rendered):
            codes.append("long_repeated_character")
    if cleaned is None or not str(cleaned).strip():
        codes.append("missing_cleaned_text")
    return codes


def duration_bucket(seconds):
    for bound, label in ((1, "<1s"), (3, "1-<3s"), (6, "3-<6s"),
                         (12, "6-<12s"), (30, "12-<30s")):
        if seconds < bound:
            return label
    return ">=30s"


def main():
    files = sorted(RAW.glob("data/*.parquet"))
    published_lfs = {}
    if TREE_MANIFEST.exists():
        for entry in json.loads(TREE_MANIFEST.read_text(encoding="utf-8")):
            lfs = entry.get("lfs") or {}
            if entry.get("path") and lfs.get("oid"):
                published_lfs[Path(entry["path"]).name] = lfs["oid"]
    output = {
        "audit_kind": "read_only_forensic_audit",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kind": "public_mirror",
            "repository": "https://huggingface.co/datasets/MohamedRashad/SADA22",
            "revision": "094fe2c0fe4b549a4f34349e6e0622e7c7273c2d",
            "upstream_official": "https://www.kaggle.com/datasets/sdaiancai/sada2022",
            "upstream_version": 23,
            "license_claim": "CC-BY-NC-SA-4.0",
        },
        "acquisition_state": "COMPLETE" if len(files) == EXPECTED_FILES else "PARTIAL",
        "expected_parquet_files": EXPECTED_FILES,
        "available_parquet_files": len(files),
        "expected_rows_from_pinned_card": EXPECTED_ROWS,
        "files": [],
        "rows": {"total": 0, "by_split": collections.Counter()},
        "metadata": {"dialect": collections.Counter(), "gender": collections.Counter(),
                     "age": collections.Counter(), "dialect_gender": collections.Counter()},
        "audio": {"missing_payload": 0, "header_decode_errors": 0,
                  "sample_rate": collections.Counter(), "channels": collections.Counter(),
                  "format": collections.Counter(), "subtype": collections.Counter(),
                  "duration_seconds": 0.0, "duration_bucket": collections.Counter(),
                  "duration_seconds_by_dialect": collections.Counter(),
                  "core_saudi_rows": 0, "core_saudi_duration_seconds": 0.0,
                  "duplicate_payloads": 0},
        "transcripts": {"anomaly_counts": collections.Counter(), "text_equals_cleaned": 0,
                        "text_differs_cleaned": 0},
        "representative_content_checks": [],
        "limitations": [
            "This mirror exposes six fields only; it does not retain show/source, environment, segment-time, local-speaker, or stable global-speaker identifiers.",
            "Speaker1..7 labels described for SADA must not be treated as global identities; this mirror cannot support source/show isolation or speaker-disjoint verification.",
            "Audio header validation is applied to every embedded payload. Actual waveform metrics are a deterministic bounded sample per observed dialect/gender stratum, not ASR transcript verification.",
            "No source files, embedded audio, or transcripts are modified by this audit."
        ],
    }
    seen_audio = set()
    inspected_strata = set()
    for parquet_path in files:
        split = parquet_path.name.split("-")[0]
        digest = hashlib.sha256()
        with parquet_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        pf = pq.ParquetFile(parquet_path)
        output["files"].append({
            "path": str(parquet_path.relative_to(ROOT)), "bytes": parquet_path.stat().st_size,
            "sha256": digest.hexdigest(), "row_groups": pf.num_row_groups,
            "rows": pf.metadata.num_rows, "schema": str(pf.schema_arrow),
            "published_lfs_sha256": published_lfs.get(parquet_path.name),
            "checksum_matches_published_lfs": digest.hexdigest() == published_lfs.get(parquet_path.name),
        })
        for batch in pf.iter_batches(batch_size=64):
            columns = batch.to_pydict()
            for index in range(batch.num_rows):
                output["rows"]["total"] += 1
                output["rows"]["by_split"][split] += 1
                dialect = clean_value(columns.get("speaker_dialect", [None])[index])
                gender = clean_value(columns.get("speaker_gender", [None])[index])
                age = clean_value(columns.get("speaker_age", [None])[index])
                output["metadata"]["dialect"][dialect] += 1
                output["metadata"]["gender"][gender] += 1
                output["metadata"]["age"][age] += 1
                output["metadata"]["dialect_gender"][f"{dialect} | {gender}"] += 1
                text = columns.get("text", [None])[index]
                cleaned = columns.get("cleaned_text", [None])[index]
                for code in anomaly_codes(text, cleaned):
                    output["transcripts"]["anomaly_counts"][code] += 1
                if text is not None and cleaned is not None:
                    if str(text) == str(cleaned):
                        output["transcripts"]["text_equals_cleaned"] += 1
                    else:
                        output["transcripts"]["text_differs_cleaned"] += 1
                payload = audio_payload(columns.get("audio", [None])[index])
                if not payload:
                    output["audio"]["missing_payload"] += 1
                    continue
                audio_hash = hashlib.sha256(payload).hexdigest()
                if audio_hash in seen_audio:
                    output["audio"]["duplicate_payloads"] += 1
                else:
                    seen_audio.add(audio_hash)
                try:
                    header = audio_header(payload)
                    output["audio"]["sample_rate"][str(header["sample_rate"])] += 1
                    output["audio"]["channels"][str(header["channels"])] += 1
                    seconds = header["frames"] / header["sample_rate"]
                    output["audio"]["duration_seconds"] += seconds
                    output["audio"]["duration_bucket"][duration_bucket(seconds)] += 1
                    output["audio"]["duration_seconds_by_dialect"][dialect] += seconds
                    if any(x in dialect.lower() for x in CORE_DIALECTS):
                        output["audio"]["core_saudi_rows"] += 1
                        output["audio"]["core_saudi_duration_seconds"] += seconds
                    output["audio"]["format"][header["format"]] += 1
                    output["audio"]["subtype"][header["subtype"]] += 1
                    stratum = f"{dialect} | {gender}"
                    if stratum not in inspected_strata and len(output["representative_content_checks"]) < 24:
                        inspected_strata.add(stratum)
                        metrics = content_metrics(payload)
                        output["representative_content_checks"].append({
                            "stratum": stratum, "file": parquet_path.name,
                            "row_in_batch": index, "text_characters": len(str(text or "")),
                            "contains_core_saudi_label": any(x in dialect.lower() for x in CORE_DIALECTS),
                            **metrics,
                        })
                except Exception as exc:  # recorded, never repaired
                    output["audio"]["header_decode_errors"] += 1
                    if len(output["audio"].setdefault("decode_error_examples", [])) < 20:
                        output["audio"]["decode_error_examples"].append({
                            "file": parquet_path.name, "row_in_batch": index,
                            "error": f"{type(exc).__name__}: {exc}"})
    output["rows"]["by_split"] = dict(sorted(output["rows"]["by_split"].items()))
    for category in ("metadata",):
        for name, values in output[category].items():
            output[category][name] = dict(sorted(values.items()))
    for name, values in list(output["audio"].items()):
        if isinstance(values, collections.Counter):
            output["audio"][name] = dict(sorted(values.items()))
    output["audio"]["duration_seconds"] = round(output["audio"]["duration_seconds"], 3)
    output["audio"]["core_saudi_duration_seconds"] = round(output["audio"]["core_saudi_duration_seconds"], 3)
    output["audio"]["duration_seconds_by_dialect"] = {
        key: round(value, 3) for key, value in output["audio"]["duration_seconds_by_dialect"].items()}
    for name, values in list(output["transcripts"].items()):
        if isinstance(values, collections.Counter):
            output["transcripts"][name] = dict(sorted(values.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
