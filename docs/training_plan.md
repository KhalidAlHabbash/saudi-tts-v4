# Training plan

This plan describes the next authorized experiment; it is not evidence that full fine-tuning has run. The preparation gate is PASS for private/local/non-commercial research, with native Saudi transcript review and MPS backward validation still outstanding.

## Frozen inputs

Use the processed manifests in `data/splits/` and the pinned SILMA/F5-TTS plus Vocos assets in `models/base/`. Do not add supplemental corpora without a new rights/provenance decision. Preserve raw text, provenance, and unavailable speaker/source sentinels. The final manifest has 125,534 rows, hash `047163a1e010f9e87fed12eeec67d331870cf92d7ee4761ff046bb7cab3de194`.

## Planned run

The only full-run command is:

```bash
./scripts/train.sh
```

`configs/train.yaml` specifies the local profile. The RunPod profile in `configs/train_runpod.yaml` continues the full update-100 state on one Community RTX 4090 using BF16, batch 2, accumulation 4, <=8-second clips, activation checkpointing, ordinary PyTorch attention/AdamW, four workers, pinned memory, and TF32. The scheduler ceiling remains 120,000 updates; stage stops are 13,122, 26,144, 52,188, 104,276, and 120,000. Validation runs every 250 updates, `model_last` around every 500, and immutable checkpoints every 5,000 plus stage boundaries.

Before a RunPod stage, follow the fail-closed order in `docs/runpod.md`: isolated batch1/acc8 then batch2/acc4 probes; deterministic production-order resume from update 100 to update 200 or 300 without changing the 120,000-update scheduler; strict durable upload of the resulting fingerprinted benchmark checkpoint; restore to a separate drill path; only then the formal 13,122 stage launch. The original fingerprint-less update-100 checkpoint is not eligible for durable `LATEST.json`. The update-300 benchmark includes validation/checkpoint overhead; compare workers 2/4/8 and prefetch 2/4 only in separately approved variants.

## Resume and evaluation

With `resume: auto`, rerunning the launcher restores only a full project checkpoint. New checkpoints include a sampler fingerprint, and matching RunPod restarts preserve exact epoch/batch position. The sole exception is the existing fingerprint-less update-100/epoch-0 Mac checkpoint, whose batch position is reset once for the changed CUDA sampler. Any later mismatch is a hard failure.

After an authorized run, synthesize held-out phrases with the actual CLI:

```bash
PYTHONPATH=src .venv/bin/python -m src.evaluation.synthesize \
  --checkpoint checkpoints/silma-saudi/model_last.pt \
  --phrase-id <id-from-configs/saudi_eval_phrases.jsonl> \
  --output outputs/evaluation/synthesis.wav
```

Report checkpoint hashes, device, configuration, manifest hash, phrase IDs, and human/native review. Do not present synthetic output as speaker identity reproduction or as evidence of speaker-disjoint generalization.

## Stop conditions

Stop on unsupported MPS operations, NaN/Inf loss or gradients, allocator pressure, checkpoint corruption, non-finite audio, manifest/hash drift, or any rights/consent uncertainty. Do not publish raw audio, weights, or a service; commercial use, distribution, and identity cloning require separate written rights and consent review.

The historical [`docs/training_guide.md`](training_guide.md) remains a supplementary command-oriented guide. This plan is the current scope and gate record; useful details there should be reconciled against the frozen configs and reports before any run.
