"""Generic deterministic split helpers for datasets that expose real speakers."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable, Mapping


def speaker_disjoint_split(rows: Iterable[Mapping[str, str]], *, seed: int, proportions: Mapping[str, float]) -> list[dict[str, str]]:
    """Assign every real speaker to one split; reject unavailable identities."""
    names = tuple(proportions)
    if set(names) != {"train", "validation", "test"} or abs(sum(proportions.values()) - 1.0) > 1e-9:
        raise ValueError("proportions must contain train, validation, test and sum to one")
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        speaker = row.get("speaker_id")
        if not speaker or speaker.startswith("__speaker_id_unavailable__"):
            raise ValueError("speaker-disjoint splitting requires real stable speaker_id values")
        grouped[speaker].append(row)
    thresholds, running = [], 0.0
    for name in names:
        running += proportions[name]
        thresholds.append((name, running))
    output: list[dict[str, str]] = []
    for speaker in sorted(grouped):
        value = int(hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest()[:16], 16) / 2**64
        split = next(name for name, threshold in thresholds if value < threshold)
        for row in grouped[speaker]:
            item = dict(row)
            item["split"] = split
            output.append(item)
    return output


def assert_speaker_disjoint(rows: Iterable[Mapping[str, str]]) -> None:
    owners: dict[str, str] = {}
    for row in rows:
        speaker, split = row["speaker_id"], row["split"]
        if speaker in owners and owners[speaker] != split:
            raise ValueError(f"speaker {speaker!r} spans {owners[speaker]!r} and {split!r}")
        owners[speaker] = split
