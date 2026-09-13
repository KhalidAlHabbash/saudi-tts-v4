# Diagnostic ASR quality-control report

Run date: 2026-09-11. This is a bounded diagnostic on the final accepted Gate B
corpus only. It did not change manifests, audio, split membership, `raw_text`,
or training `text`; it is not an automated rejection or transcript-correction
system.

## Model provenance

- Model: `openai/whisper-small`
- Immutable revision: `973afd24965f72e36ca33b3055d56a652f456b4d`
- Local cache snapshot:
  the local Hugging Face cache at revision `973afd24965f72e36ca33b3055d56a652f456b4d`
- `model.safetensors`: 966,995,080 bytes; SHA-256
  `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`
- Complete pinned file hashes, per-sample predictions, and summaries are in
  `reports/asr_quality_sample.json` (SHA-256
  `9e06f515c183c3af56b26f7707a7714e1590ebd9fe990982d22f2bd5aedac4de`).

The CPU run used Arabic transcription prompting on 24-kHz source WAVs resampled
in memory to 16 kHz for Whisper. No audio file was rewritten.

## Deterministic sample and metrics

The seed-20260911 sample selects the two lowest stable record-key ranks in each
of the 18 final `dialect × gender × split` strata, limited to practical 2–8 s
clips: **36 clips** total. It covers Najdi/Hijazi/Khaliji, Female/Male labels,
and train/validation/test.

For scoring only, both reference and ASR strings receive NFC, punctuation,
spacing, and combining-diacritic-insensitive comparison normalization. The
corpus transcript itself is never normalized this way. Character error rate
(CER) is Levenshtein character distance/reference characters; word error rate
(WER) is Levenshtein word distance/reference words. Repetitive ASR hallucinated
output can make CER/WER exceed 1.0. Character similarity is
`1 - distance / max(reference length, ASR length)`.

| Group | N | Mean CER | Mean WER | Mean character similarity |
|---|---:|---:|---:|---:|
| Overall | 36 | 1.118934 | 1.470403 | 0.685228 |
| Najdi | 12 | 2.837266 | 3.142884 | 0.564872 |
| Hijazi | 12 | 0.288684 | 0.657726 | 0.719743 |
| Khaliji | 12 | 0.230853 | 0.610600 | 0.771068 |
| Female | 18 | 1.215983 | 1.515528 | 0.733623 |
| Male | 18 | 1.021886 | 1.425279 | 0.636832 |
| Train | 12 | 0.272276 | 0.618072 | 0.730952 |
| Validation | 12 | 1.710792 | 2.079632 | 0.713909 |
| Test | 12 | 1.373735 | 1.713507 | 0.610823 |

The small diagnostic sample is not a corpus-wide error estimate and should not
be used to rank dialect quality. In particular, the high Najdi aggregate is
driven by a small number of severe Whisper repetitions.

## Representative manual inspection

Manual reading of representative outputs found both plausible dialect/spelling
variation and obvious ASR failure signals:

| Signal | Reference | Whisper output | Interpretation |
|---|---|---|---|
| High agreement (Khaliji/Female/test; similarity 0.955) | `حاضر يا عمتي حاضر ان شاء الله من عيوني بتأكد` | `حاضر يا امتي حاضر ان شاء الله من ايوني بتأكد` | Near match with expected dialectal/orthographic substitutions. |
| Good partial agreement (Khaliji/Male/validation; 0.884) | `وافق الرجال وقال أبي أجيب البنات مع أهاليهم.` | `ووافق الرجال وقال أبا جيب البنات مع أهديهم` | Mostly aligned wording, with substitutions. |
| Obvious failure (Najdi/Male/test; 0.032) | `خلال أقل من سنة إن شاء الله السداد جايك كل ال--.` | Repeated `لقى` hundreds of times | Whisper repetition/hallucination signal, not evidence that the supplied transcript is wrong. |
| Obvious failure (Najdi/Female/validation; 0.020) | `في ناس الله ساكنة كذا ني زي أكلنا مشوي مقلي.` | Repeated `أتمنى أن` hundreds of times | Whisper repetition/hallucination signal, not a transcript correction basis. |

This automated comparison establishes that the selected ASR model produces a
mixture of approximate matches and conspicuous failures on this broadcast,
dialectal material. It cannot establish that a disagreement is an annotation
error: Saudi dialect, names, code-switching, noise, overlapping speech, and the
model's own limitations all confound that inference. Native Saudi speaker
audio/transcript review remains the required external method for semantic
alignment and dialect correctness. No ASR output was used to overwrite or
auto-reject a human transcript.

## Validation

- `reports/asr_quality_sample.json` parsed successfully as JSON.
- Focused data tests: 8 passed.
- Gate B manifests were not changed by this QC run.

Reproduce after the pinned snapshot is cached:

```bash
PYTHONPATH=src .venv/bin/python -m data.asr_qc --config configs/data.yaml --per-stratum 2
```
