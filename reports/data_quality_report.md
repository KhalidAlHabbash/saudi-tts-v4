# SADA acquisition and data-quality report

Audit date: 2026-09-11.  This is a read-only audit: source Parquet files, embedded WAV payloads, and transcript strings were not edited, extracted, normalized, or replaced.

## Acquisition and provenance

- **Acquired release:** complete public Hugging Face mirror `MohamedRashad/SADA22`, immutable revision `094fe2c0fe4b549a4f34349e6e0622e7c7273c2d`, stored untouched at `data/raw/SADA22_hf_094fe2c0fe4b549a4f34349e6e0622e7c7273c2d/`.
- **Mirror relationship:** the mirror's included card identifies SADA as SDAIA + Saudi Broadcasting Authority material and cites the original Kaggle dataset `sdaiancai/sada2022`.  Kaggle's unauthenticated API confirms the official publisher, CC BY-NC-SA 4.0, version **23** (`2023-08-31`, “Data Update 2023-08-31”), and a 77,055,385,134-byte current release.  Its anonymous archive endpoint returned HTTP 404, so this acquisition is explicitly a pinned mirror, not an asserted byte-identical copy of Kaggle v23.
- **Completeness/integrity:** 31/31 Parquet shards, 50,401,167,803 bytes, 253,166 rows.  Each local Parquet SHA-256 matches the Hugging Face LFS object SHA-256 published for that exact commit (31/31).  The per-file paths, bytes, hashes, and verification flags are in the [machine audit JSON](../data/raw/_audit/sada22_hf_094fe2c0fe4b549a4f34349e6e0622e7c7273c2d_audit.json).  `README.md` SHA-256: `1b9d909aab47ad1a985e72407a279688190483ca067908bd1b1eab18a0616b21`; `.gitattributes` SHA-256: `e7a120ab07b1bc5b486be249e9fc6c83d59448d0093e1dfebe95d1566d9cafc0`.
- **Licence:** both the official Kaggle metadata and mirror card state CC BY-NC-SA 4.0.  This remains a non-commercial research pilot only; rights review is still required before sharing derived weights or raw audio.

## What is actually present

The mirror has exactly the six fields `audio`, `text`, `cleaned_text`, `speaker_age`, `speaker_gender`, and `speaker_dialect`; it does **not** expose source/show, scene/environment, segment time, local speaker tracking, or globally stable speaker identifiers.

| Split | Rows |
|---|---:|
| train | 241,834 |
| validation | 5,139 |
| test | 6,193 |
| total | 253,166 |

All 253,166 embedded payloads decode at the header level as mono, 16-kHz, 16-bit PCM WAV.  There are no missing payloads and no header/decode failures.  Total embedded duration is **1,575,163.157 s / 437.545 h**, indicating that this mirror contains the labelled material rather than the original release's reported roughly 667-hour labelled-plus-unlabelled total.  Three byte-identical embedded-audio payload duplicates were detected.

| Utterance duration | Rows |
|---|---:|
| <1 s | 34,840 |
| 1–<3 s | 85,645 |
| 3–<6 s | 54,370 |
| 6–<12 s | 39,861 |
| 12–<30 s | 32,494 |
| ≥30 s | 5,956 |

## Saudi-label suitability

The requested Najdi/Hijazi/Khaliji labels are present and form a substantial, auditable core: **161,101 rows / 195.402 h** (63.64% of rows; 44.66% of mirrored duration).

| Dialect label | Rows | Hours | Female / male / unknown |
|---|---:|---:|---:|
| Najdi | 94,611 | 122.369 | 31,278 / 63,231 / 102 |
| Hijazi | 36,170 | 41.993 | 7,020 / 29,137 / 13 |
| Khaliji | 30,320 | 31.040 | 7,489 / 22,816 / 15 |

This supports Saudi-dialect adaptation research, especially Najdi by scale.  It is not a ready single-voice corpus: known male labels substantially outnumber female labels, and the source is broadcast/drama rather than controlled voice recording.  A deterministic objective read of the first 10 seconds from 24 dialect/gender strata found normal nonzero PCM energy and variable silence/level characteristics; detailed metrics are in the JSON.  No ASR was run, and no claim of semantic transcript-to-audio alignment is made.

## Quality flags and limits

- `More than 1 speaker` accounts for 52,501 rows / 195.445 h; `Unknown` dialect accounts for 30,867 rows / 34.179 h.  Exclude multi-speaker and unknown-dialect rows from the first TTS candidate set.
- Transcript checks: 76 missing `text`, 0 missing `cleaned_text`, 34 nonempty `text` fields with no Arabic-script character, 3 control-character cases, and 63 long repeated-character cases.  `text` differs from `cleaned_text` in 212,657 rows; preserve both and defer a documented text-policy decision to preprocessing.
- The mirror strips source/show/environment/local-speaker fields.  It cannot prove speaker-disjointness, show isolation, recording condition, or the paper's local `Speaker1..7` grouping.  Never map these missing or local labels to global speaker identities.  This is the key limitation for the planned leakage-resistant split strategy.
- Rows labelled Egyptian (2,172), Levantine (966), MSA (4,302), and other/unknown/non-applicable labels are not evidence for the Saudi core and should not enter it by default.
- The audit establishes file/payload integrity, metadata availability, and objective signal properties.  Native Saudi review is still required for dialect accuracy and transcript/audio alignment before training selection.

## Recommendation

Accept this **complete, pinned mirror** as the research-pilot raw source.  Build the baseline candidate manifest only from Najdi, Hijazi, and Khaliji, beginning with single-speaker-labelled, non-anomalous transcript rows and conservative duration/quality filters.  Keep all source data immutable, retain the provenance fields above, and treat every downstream split as source/speaker-leakage-risked until access to authoritative show/local-speaker metadata is available.
