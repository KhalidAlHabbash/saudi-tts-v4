# Independent Gate B validation report

Validation date: 2026-09-11. This is the final targeted revalidation of the
remediated Gate B artifacts after the two-row embedded machine-token cleanup.
The frozen scope is **private, local, non-commercial research preparation
only**. No real training launcher, model forward, synthesis, or additional
backward/optimizer update was run. The earlier single CPU update is retained
below as historical technical evidence.

## Gate B disposition

**PASS for the frozen research-preparation scope, with HIGH risks that must stay
visible.** The final manifest, split collision policy, uncertainty-marker
remediation, training dry-run, tokenizer, and tests are internally consistent
and independently rechecked. Prior independent evidence for the unchanged
dependency lock, model assets, bounded ASR diagnostic, and documented
host-visible MPS forward is retained below.

This is **not a go decision for full fine-tuning**. Native Saudi
audio/transcript/dialect review is still absent, stable actor/speaker/show IDs do
not exist in the mirror, and MPS backward/optimizer behavior is untested.

Expansion to distribution, commercial use, public voice release, or
identity-cloning is **BLOCKED and out of scope** pending rightsholder/counsel,
attribution, and voice-consent review. No result in this report weakens that
boundary.

## Classification summary

### BLOCKING outside the frozen scope

1. **Distribution, commercial use, and identity cloning remain blocked.** The
   mirror reports CC BY-NC-SA 4.0 and identifies SDAIA/Saudi Broadcasting
   Authority, but this validation does not resolve derived-weight treatment,
   commercial permissions, attribution/ShareAlike implementation, publicity or
   personality rights, or consent for identifiable broadcast voices. The scope
   decision in `docs/project_state.md` is therefore appropriate.

### HIGH

1. **Speaker/actor/show leakage is not measurable.** Every accepted row still
   uses `__speaker_id_unavailable__` and `__source_id_unavailable__`; there is no
   stable actor, show, scene, or source grouping key. Exact and loose text
   collision removal does not make the official splits speaker-disjoint. No
   evaluation or model-generalization claim may describe them as such.

2. **Native transcript and dialect validation is still missing.** Of 125,534
   accepted rows, 75,777 have a mirror `raw_text`/`cleaned_text` relation marked
   `review_required_content_or_diacritic_difference`. This is a review signal,
   not proof that either string is correct. Automated normalization, metadata
   inspection, and Whisper cannot establish semantic audio alignment or Saudi
   dialect correctness.

3. **MPS backward remains untested.** The host diagnostic demonstrates model
   construction, strict load, tensor placement, and a finite no-grad SILMA
   forward with unsupported-operation fallback disabled. It does not exercise
   activation-checkpoint recomputation, gradient storage, AdamW, EMA update,
   checkpoint writing from MPS state, or full-training memory behavior.

### MEDIUM

1. **The Whisper QC has poor, confounded metrics and is not ground truth.** Its
   deterministic 36-clip sample reports mean CER 1.118934 and mean WER 1.470403,
   with conspicuous repetition hallucinations. These values show that
   `whisper-small` is unreliable as an automatic judge on this small broadcast,
   dialectal sample. They must not be used to correct/reject transcripts or rank
   dialect quality. The report correctly preserves that boundary.

2. **Within-split repeated content remains.** The final corpus has 123,834
   unique exact texts and 122,864 unique configured loose keys across 125,534
   rows. Cross-split collisions are zero, but repeated phrases within one split
   remain intentionally allowed and can affect training weights.

3. **Processed-file integrity and perceptual duplication remain limited.** The
   manifest stores the embedded source-payload hash, not a hash of the produced
   24-kHz PCM bytes. Header/duration validation passes, but overlapping excerpts,
   re-encoded/perceptual duplicates, and later bit-level mutation are not
   excluded by the stored value.

4. **The F5 wheel is attested, but the Git mapping is not.** The local PyPI
   `f5_tts-1.1.7-py3-none-any.whl` independently hashes to
   `8deb87c8f3968f6f4783e8ec76b255c9d87e29ec6c2e9b5f37b01732c8c80a6a`,
   an allowed lock hash. Its metadata states F5-TTS 1.1.7 and MIT, but does not
   attest configured Git commit
   `c96c3aeed84f5e02aa54dc42c1193537ead39837`.

### LOW

1. TorchAudio 2.8 emits its announced 2.9 decoder-backend migration warning.
2. The host MPS measurements are recorded in a narrative report rather than a
   raw machine-readable capture. They are detailed and internally consistent,
   but this restricted reviewer process cannot independently rerun the
   host-visible command.

## Independent Gate B checks

| Check | Result | Evidence |
| --- | --- | --- |
| Canonical manifest | PASS | 125,534 rows; SHA-256 `047163a1e010f9e87fed12eeec67d331870cf92d7ee4761ff046bb7cab3de194`; split counts 119,972 train / 2,717 validation / 2,845 test. Independent raw-byte hash and row scan match the machine summary. |
| Split copies | PASS | `data/splits/*.jsonl` and `data/processed/manifest_*.jsonl` exactly equal the ordered canonical subsets. |
| Metadata/hashes | PASS | The final manifest is a strict subset of the previously hash-validated rows; all 125,534 source-payload audio hashes remain unique. |
| Cross-split exact audio | PASS | Zero groups. |
| Cross-split exact text | PASS | Zero groups. |
| Cross-split configured loose text | PASS | Zero groups after NFC plus spacing, Unicode-punctuation, and Arabic-combining-mark removal. |
| Annotation/uncertainty scan | PASS | Zero accepted texts contain `#`; zero match the standalone annotation rule; zero match underscore/hyphen machine forms at Arabic token boundaries. Two ordinary spaced-prose phrase matches remain intentionally and are covered by the native-review limitation rather than classified as machine tokens. |
| Final WAVs | PASS | The final project check opened all 125,534 referenced files: 25,704,751,276 bytes total, 24 kHz mono PCM-16, positive frames, duration within one sample. This narrow revalidation did not repeat the unchanged full audio scan. |
| Quarantine | PASS | Report/summary account for 4,491 unreferenced generated WAVs, 1,061,203,542 bytes, moved to Gate B quarantine rather than deleted. |
| Train dry-run | PASS | Final accepted rows under <=8 seconds: 104,176 train and 2,330 validation; batch `[1,100,201]`, length 201; zero OOV types. This path did not construct or execute the model. |
| Arabic vocabulary | PASS | Pinned 9,260-entry character vocabulary remains strict; final dry-run and eight evaluation phrases have no OOV characters. |
| Evaluation phrase overlap | PASS | Eight phrases have zero exact overlap, zero configured-loose overlap, and zero >=12-character configured-loose manifest-text substring matches. |
| Tests | PASS | 17 passed; one TorchAudio deprecation warning. |
| Installed environment | PASS | `pip check` reports no broken requirements. All 149 lock packages exactly match installed versions; only pip/build tooling is outside the runtime lock as documented. |
| Hash-lock consumption | PASS | Offline `pip install --dry-run --require-hashes -r requirements.lock.txt` accepted the populated environment; bootstrap uses the same file and `--require-hashes`. |
| Model assets | PASS | All six SILMA/Vocos files match `models/base/asset_manifest.json`. |
| ASR artifact/provenance | PASS as diagnostic only | JSON SHA-256 matches `9e06f515c183c3af56b26f7707a7714e1590ebd9fe990982d22f2bd5aedac4de`; sample selection exactly reproduces 36 rows in 18 strata; every score and summary recomputes; cached Whisper weight is 966,995,080 bytes with matching SHA-256 `1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b`. |
| MPS visibility in restricted process | UNAVAILABLE | Reproduced built `True`, available `False`, CPU selection, and failure to create an MPS tensor. This is not credited as an MPS pass or OS failure. |
| Host MPS tensor operation | PASS from documented host evidence | Same venv/interpreter reportedly returns built/available `True/True`; `(arange(8).square()+1).sum()` on `mps:0` returns 148.0. |
| Host SILMA MPS forward | PASS for forward feasibility only | `PYTORCH_ENABLE_MPS_FALLBACK=0`; 162,666,852 parameters; real batch `[1,100,201]`; finite loss 0.63564199; peak sampled current allocation 634.97 MiB, driver allocation 1.100 GiB, RSS 1.641 GiB. No CPU-parity claim. |
| MPS backward/optimizer | UNTESTED / HIGH | Explicitly not run. |
| Launcher | PASS static only | Shell syntax passes. `scripts/train.sh` was inspected and not executed. |

## Historical CPU training-path evidence

The earlier, single authorized development update predates Gate B
rematerialization but used a real sample retained in the final data. It remains
valid evidence for CPU integration only:

- finite CFM loss `0.568487286567688`;
- gradient norm `3.4156463146209717`;
- exactly one AdamW update, with 13,836 changed parameter elements across 157
  tensors;
- 2.4-GiB checkpoint containing model, optimizer, scheduler, EMA, update,
  epoch/batch position, and CPU/Python RNG state;
- strict full-model resume returned `(update=1, epoch=0, next_batch=1)` with all
  162,666,852 parameters finite;
- fixed base EMA plus local 13,531,650-parameter Vocos produced a valid
  3.829333-second, 24-kHz PCM-16 synthesis.

No second backward pass or optimizer update was run during Gate B revalidation.

## Validation-owned corrections

1. Updated final manifest counts/hash and dry-run counts in
   `docs/project_state.md` to the post-cleanup values.
2. Replaced the resolved uncertainty-marker queue with independently measured
   zero counts for standalone annotations and embedded underscore/hyphen machine
   forms; retained the broader native transcript-review limitation.
3. Corrected two stale raw/cleaned relation counts in
   `reports/preprocessing_report.md` to match the machine summary.
4. Retained the environment/MPS distinction: restricted-process MPS is
   unavailable; documented host no-grad forward is feasible; backward remains
   untested.

## Gate conditions after this review

- **Prepared private/local/non-commercial research artifacts:** PASS.
- **Full fine-tuning:** not authorized by this preparation gate; obtain native
  Saudi review and separately validate MPS backward first.
- **Speaker-disjoint or actor-disjoint evaluation claim:** prohibited by missing
  metadata.
- **Whisper-based transcript truth or automatic rejection claim:** prohibited.
- **Distribution, commercial use, public voice release, or identity cloning:**
  BLOCKED pending rights/consent/attribution review.
