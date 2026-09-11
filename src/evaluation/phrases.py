"""Validate the Saudi evaluation text suite and prove exact non-overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.training.config import resolve_path
from src.training.data import load_vocab

REQUIRED_CATEGORIES = {
    "conversational_saudi",
    "dialect_vocabulary",
    "msa",
    "numbers",
    "dates",
    "english_loanwords",
}


def load_phrases(path: str | Path) -> list[dict[str, str]]:
    phrases = []
    seen_ids: set[str] = set()
    with resolve_path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not all(isinstance(row.get(key), str) and row[key] for key in ("id", "category", "text")):
                raise ValueError(f"Invalid evaluation phrase at line {line_number}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate evaluation id {row['id']}")
            seen_ids.add(row["id"])
            phrases.append(row)
    categories = {row["category"] for row in phrases}
    missing = REQUIRED_CATEGORIES - categories
    if missing:
        raise ValueError(f"Evaluation suite is missing categories: {sorted(missing)}")
    return phrases


def audit_phrases(*, phrases_path: str | Path, manifest_paths: list[str | Path], vocab_path: str | Path) -> dict[str, Any]:
    phrases = load_phrases(phrases_path)
    training_texts: set[str] = set()
    for manifest_path in manifest_paths:
        with resolve_path(manifest_path).open(encoding="utf-8") as handle:
            training_texts.update(json.loads(line)["text"] for line in handle)
    overlap = [row["id"] for row in phrases if row["text"] in training_texts]
    if overlap:
        raise RuntimeError(f"Evaluation phrases exactly overlap dataset manifests: {overlap}")
    vocab, _ = load_vocab(vocab_path)
    unknown = sorted({char for row in phrases for char in row["text"] if char not in vocab})
    if unknown:
        raise RuntimeError(f"Evaluation phrases contain out-of-vocabulary characters: {unknown!r}")
    return {
        "phrase_count": len(phrases),
        "categories": sorted({row["category"] for row in phrases}),
        "manifest_text_count": len(training_texts),
        "exact_overlap_count": 0,
        "oov_characters": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    with resolve_path(args.config).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    result = audit_phrases(
        phrases_path=cfg["phrases"],
        manifest_paths=["data/splits/train.jsonl", "data/splits/validation.jsonl", "data/splits/test.jsonl"],
        vocab_path=cfg["vocab"],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
