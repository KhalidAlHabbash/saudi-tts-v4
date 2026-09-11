# SADA Gate B preprocessing report

Run date: 2026-09-11. This is the remediated full preparation of the pinned
`MohamedRashad/SADA22` mirror revision
`094fe2c0fe4b549a4f34349e6e0622e7c7273c2d` (reported licence:
CC BY-NC-SA 4.0). Raw data remains immutable; this is a non-commercial
research-pilot artefact.

## Gate B policy

The baseline retains only single-speaker `Najdi`, `Hijazi`, and `Khaliji`
records with `Female`/`Male` metadata, valid Arabic transcripts, decodable
audio, 1.0–20.0 seconds, RMS >= 0.0005, and clipping <= 2%. It now rejects
any transcript containing `#`, including `#غير-واضح` spelling/spacing variants
and every other hash-prefixed annotation, noise, or uncertainty token. This is
a rejection policy: it never guesses or edits spoken content.

Gate B cleanup additionally rejects only a standalone/token-form uncertainty
label whose punctuation/spacing-only form is `غيرواضح` or `غيرمفهوم` (including
`غير واضح.`, `غير_واضح`, and `غير-واضح`). It does not reject natural sentences
which merely contain these words, such as `الصوت غير واضح اليوم`.

The final delta cleanup also rejects `غير_واضح`, `غير-واضح`, `غير_مفهوم`, and
`غير-مفهوم` wherever they occur as Arabic-token-boundary machine forms inside
a transcript. This catches an embedded token such as `فكر بس غير_واضح` while
still preserving ordinary spaced prose such as `الصوت غير واضح اليوم`.

`raw_text` remains unchanged. Training `text` still receives only Unicode NFC
and ASCII whitespace collapsing. It does not fold Arabic letters, remove
diacritics, alter Saudi vocabulary, correct spelling, use `cleaned_text`, or
use ASR.

Mirror `raw_text`/`cleaned_text` divergence is now classified, rather than
flagged wholesale:

| Classification | Accepted rows | Meaning |
|---|---:|---|
| Equal | 21,430 | Supplied strings match |
| Safe whitespace only | 3,558 | NFC/spacing presentation difference |
| Safe punctuation or spacing only | 24,769 | Presentation-only comparison difference |
| Review-required content or diacritic difference | 75,777 | May change lexical/vocalized content; no automatic replacement |

The safe labels only characterize the mirror fields; they do not change model
text. A deterministic 72-row, 24-stratum sample is in
`data/interim/transcript_pair_stratified_sample.jsonl`, selected by seed
20260911. It includes raw/cleaned/training text, relation, dialect, gender,
split, and flags for native review. Inspection of the reproducible sample
confirmed all four categories: equal pairs, punctuation-only pairs (for example
an Arabic question mark removed by the mirror), spacing-only pairs, and
review-required letter/diacritic/content changes. The sample is a metadata
inspection, not semantic audio/transcript validation.

## Split collision policy

Every retained row keeps its mirror-assigned official split. For cross-split
content collisions only, priority is **test → validation → train**: the first
encountered higher-priority official split is retained and colliding lower
priority rows are rejected. This protects held-out evaluation content without
reassigning a row to another split.

The comparison-only loose key uses NFC, removes spacing, Unicode punctuation,
and combining Arabic diacritics, then hashes the result. The raw key is never
written as training text or substituted for `raw_text`/`text`. Exact training
text hashes and source-payload audio hashes are also checked. Gate B rejected
1,804 rows in 125 exact-text collision groups and 1,236 rows in 124 additional
loose-text collision groups. The final full audit found zero cross-split exact
audio, exact-text, or loose-text groups.

## Results

| Outcome | Clips | Hours |
|---|---:|---:|
| Accepted | 125,534 | 148.722258 |
| Rejected | 127,632 | — |
| Train | 119,972 | included above |
| Validation | 2,717 | included above |
| Test | 2,845 | included above |

Accepted dialect counts: Najdi **73,387**, Hijazi **28,534**, Khaliji
**23,613**. The referenced 24-kHz mono PCM-16 WAVs occupy **25,704,751,276
bytes**. The canonical manifest SHA-256 is
`047163a1e010f9e87fed12eeec67d331870cf92d7ee4761ff046bb7cab3de194`.

| Rejection reason | Rows |
|---|---:|
| Dialect not in baseline | 92,065 |
| Under 1.0 s | 23,889 |
| Over 20.0 s | 4,384 |
| Hash annotation/noise/uncertainty markup | 4,104 |
| Standalone uncertainty annotation | 10 |
| Embedded underscore/hyphen machine uncertainty token | 1,953 |
| Cross-split exact text duplicate | 1,804 |
| Cross-split loose text collision | 1,236 |
| Gender unknown/multi-speaker | 130 |
| Long repeated character | 13 |
| Excessive clipping | 4 |
| Control character | 2 |
| Missing text | 1 |
| No Arabic script | 1 |

## Materialization, provenance, and validation

- `data/interim/sada_preprocessing_gate_b_token_cleanup_ledger.jsonl` is the final,
  append-only Gate B cleanup ledger for all **253,166** raw rows. Its atomic
  WAV writes and row-level records make reruns resumable.
- Normal preparation reused the existing acquisition inspection. An explicit
  final full SHA-256 rehash passed **31/31** pinned Parquet files, zero
  failures. `--download` remains an opt-in immutable-revision acquisition hook.
- All 125,534 manifest rows and WAV headers were opened: 24-kHz, mono PCM-16,
  positive duration within one output sample. Exact and loose cross-split
  audits returned zero contamination groups; the standalone/token-form
  standalone and embedded-machine uncertainty-marker scans returned zero
  accepted rows.
- Re-materializing sorted manifests was deterministic with the same hash above.
- **4,491** now-unreferenced generated WAVs (**1,061,203,542 bytes**) were
  moved, not deleted, to `data/interim/quarantine/gate_b_unreferenced_audio/`.

Every manifest keeps required canonical fields (`audio_path`, `text`,
`speaker_id`, `raw_text`, `dialect`, `region`, `gender`, `sample_rate`,
`duration`, `dataset_source`, `source_id`, `split`, `license`, `quality_flags`)
and adds auditable hashes/relations. Speaker and source fields are explicit
unavailable sentinels; this mirror has no stable speaker, actor, show, or source
identity. Therefore automated processing establishes formatting, stated-label,
audio, and collision checks only. It does **not** establish transcript/audio
semantic alignment, Saudi dialect correctness, actor isolation, voice consent,
or speaker-disjointness; native Saudi speaker review and rights review remain
external required gates. No ASR was used.

## Artefacts and commands

- Processed audio: `data/processed/audio/{train,validation,test}/`
- Manifests: `data/processed/manifest*.jsonl`, `data/splits/*.jsonl`
- Machine summary: `data/interim/preprocessing_summary.json`
- Sample: `data/interim/transcript_pair_stratified_sample.jsonl`

```bash
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml --check
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml --determinism-check
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml --verify-only
.venv/bin/python -m pytest -q
```
