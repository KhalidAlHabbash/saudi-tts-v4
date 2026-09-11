from __future__ import annotations

import numpy as np
import pytest

from data.preprocess import (Settings, canonical_collision_key, classify_raw_cleaned_difference,
                             contains_machine_uncertainty_token, is_standalone_uncertainty_annotation,
                             normalize_arabic_text, transcript_issues, validate_audio)
from data.splits import assert_speaker_disjoint, speaker_disjoint_split
from data.asr_qc import normalized_metric_text, score_pair


def test_normalization_is_conservative_and_preserves_saudi_words():
    value, transforms = normalize_arabic_text("  وش\tلونك يا خوي؟  ")
    assert value == "وش لونك يا خوي؟"
    assert transforms == ["whitespace_collapsed"]
    assert normalize_arabic_text("أهلاً بِك")[0] == "أهلاً بِك"


def test_transcript_validation_rejects_anomalies():
    assert transcript_issues(None) == ["missing_text"]
    assert "no_arabic_script" in transcript_issues("hello")
    assert "long_repeated_character" in transcript_issues("ياااااااااا")
    assert "hash_prefixed_annotation_or_uncertainty_markup" in transcript_issues("#غير-واضح")
    assert "hash_prefixed_annotation_or_uncertainty_markup" in transcript_issues("قال #هه")


def test_only_standalone_uncertainty_tokens_are_rejected():
    for value in ("غير واضح.", "  غير_واضح  ", "غير-واضح؟", "غير مفهوم،"):
        assert is_standalone_uncertainty_annotation(value)
        assert set(transcript_issues(value)).intersection({"standalone_uncertainty_annotation", "machine_uncertainty_token"})
    assert not is_standalone_uncertainty_annotation("الصوت غير واضح اليوم")
    assert not is_standalone_uncertainty_annotation("قال لي غير مفهوم من الكلام")


def test_embedded_machine_uncertainty_tokens_reject_but_spaced_prose_does_not():
    for value in ("فكر بس غير_واضح", "عندي غير-واضح ملفوف", "غير_مفهوم بسبب الضجيج"):
        assert contains_machine_uncertainty_token(value)
        assert "machine_uncertainty_token" in transcript_issues(value)
    assert not contains_machine_uncertainty_token("الصوت غير واضح اليوم")
    assert not contains_machine_uncertainty_token("الكلام غير مفهوم في هذا المكان")


def test_collision_key_is_loose_only_and_training_normalization_is_not():
    assert canonical_collision_key("عَلَيْكُم السلام،") == canonical_collision_key("عليكم   السلام.")
    assert normalize_arabic_text("عَلَيْكُم السلام،")[0] == "عَلَيْكُم السلام،"


def test_raw_cleaned_classification_separates_safe_and_review_required():
    assert classify_raw_cleaned_difference("أهلا\tبك", "أهلا بك") == "safe_whitespace_only"
    assert classify_raw_cleaned_difference("أهلا، بك", "أهلا بك") == "safe_punctuation_or_spacing_only"
    assert classify_raw_cleaned_difference("عَلَم", "عِلْم") == "review_required_content_or_diacritic_difference"


def test_audio_quality_boundaries(tmp_path):
    settings = Settings(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, 24000, 1.0, 20.0, .0005, .02,
                        ("Najdi",), ("Male",), 20260911)
    duration, flags = validate_audio(np.ones(8000, dtype=np.float32) * .1, 16000, settings)
    assert duration == .5 and "duration_too_short" in flags
    _, flags = validate_audio(np.ones(16000, dtype=np.float32), 16000, settings)
    assert "clip_fraction_above_maximum" in flags


def test_generic_split_is_deterministic_and_speaker_disjoint():
    rows = [{"speaker_id": "a", "text": "one"}, {"speaker_id": "a", "text": "two"},
            {"speaker_id": "b", "text": "three"}, {"speaker_id": "c", "text": "four"}]
    kwargs = {"seed": 20260911, "proportions": {"train": .95, "validation": .025, "test": .025}}
    first = speaker_disjoint_split(rows, **kwargs)
    assert first == speaker_disjoint_split(rows, **kwargs)
    assert_speaker_disjoint(first)
    with pytest.raises(ValueError, match="requires real"):
        speaker_disjoint_split([{"speaker_id": "__speaker_id_unavailable__"}], **kwargs)


def test_disjoint_assertion_catches_leakage():
    with pytest.raises(ValueError, match="spans"):
        assert_speaker_disjoint([{"speaker_id": "x", "split": "train"}, {"speaker_id": "x", "split": "test"}])


def test_asr_metric_normalization_is_comparison_only():
    assert normalized_metric_text("عَلَيْكُم،  السلام.") == "عليكم السلام"
    score = score_pair("عليكم السلام.", "عَلَيْكُم السلام")
    assert score["character_error_rate"] == 0
    assert score["word_error_rate"] == 0
