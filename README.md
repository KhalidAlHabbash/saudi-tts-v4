# Saudi Arabic TTS fine-tuning (private research pilot)

This repository is an auditable wrapper for adapting SILMA TTS v1 / F5-TTS to Saudi-labelled Arabic speech. The Mac run was stopped after its full update-100 checkpoint; the current phase is guarded migration and benchmarking on one RunPod Secure RTX PRO 4500 Blackwell with 32 GB VRAM.

## Scope and selected assets

The prepared corpus is the pinned `MohamedRashad/SADA22` Hugging Face mirror at revision `094fe2c0fe4b549a4f34349e6e0622e7c7273c2d`. Only rows labelled Najdi, Hijazi, or Khaliji and passing the Gate B audit are retained. SADA is broadcast speech, not a clean single-speaker corpus; the mirror does not contain stable speaker/show/actor IDs.

The selected model is SILMA TTS v1, the 162,666,852-parameter F5-TTS/DiT acoustic model. It is paired with fixed Vocos at 24 kHz. These exact revisions and file hashes are in [`models/base/asset_manifest.json`](models/base/asset_manifest.json). This choice is the pinned Saudi-research baseline; other datasets/models remain rights-gated candidates.

Use is private, local, and non-commercial research only under the reported SADA CC BY-NC-SA 4.0 terms. This project makes no speaker-disjoint claim. Do not distribute raw audio or weights, use the material commercially, or perform identity/voice cloning without separate rights, consent, and attribution review.

## Repository map

- `configs/`: data, training, evaluation, and held-out phrase contracts.
- `src/data/`: acquisition/preprocessing, split, and diagnostic ASR QC code.
- `src/training/`: asset loading, data/model integration, checkpoints, and trainer.
- `src/evaluation/`: phrase loading and synthesis.
- `scripts/`: environment bootstrap, asset download, and the sole full-run launcher.
- `data/`: immutable raw source plus processed audio/manifests and interim ledgers.
- `models/base/`: pinned SILMA and Vocos assets (materialized locally).
- `reports/`: acquisition, preprocessing, environment, MPS, validation, and ASR diagnostics.
- `docs/`: research decisions, governance, training plan, and the historical [`training_guide.md`](docs/training_guide.md).

## Bootstrap and data preparation

On the native arm64 CPython 3.11 Mac, bootstrap installs the hash-locked environment, verifies/downloads model assets, and runs tests. Project metadata also permits Python 3.12 for the official RunPod template:

```bash
./scripts/bootstrap.sh
```

To verify or acquire the pinned model assets independently:

```bash
./scripts/download_models.sh
```

The preprocessing CLI supports acquisition (`--download`), full raw rehash (`--verify-only`), existing-manifest/audio checks (`--check`), and deterministic manifest verification (`--determinism-check`). These are the actual options exposed by `src.data.preprocess`; acquisition and materialization can write data.

```bash
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml --check
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml --determinism-check
PYTHONPATH=src .venv/bin/python -m data.preprocess --config configs/data.yaml --verify-only
```

The completed Gate B history is recorded in [`reports/preprocessing_report.md`](reports/preprocessing_report.md); it is not necessary to rerun it to use the existing manifests.

## Audit, tests, and smoke history

The preparation validation recorded `pip check` success, lock/hash verification, exact split copies, zero cross-split exact-audio/exact-text/configured-loose-text groups, and zero accepted prohibited annotation tokens. The deterministic 36-clip Whisper comparison is diagnostic only; it did not modify manifests. Historical bounded compute is described in [`reports/validation_report.md`](reports/validation_report.md). No additional model compute is implied by this README.

Non-model checks:

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m src.training.train --help
PYTHONPATH=src .venv/bin/python -m src.evaluation.synthesize --help
```

`--smoke-one-step`, `--check-model`, and `--forward-only` are real options, but they respectively update smoke weights, construct/load the model, and execute a model forward; use only when that compute is explicitly authorized.

## Training and resume

The sole real training command is shown here and was not run during documentation validation:

```bash
./scripts/train.sh
```

It uses `configs/train.yaml`: FP32, batch size 1, gradient accumulation 8, zero workers, activation checkpointing, ordinary PyTorch attention, no frozen layers, and MPS-then-CPU selection. `runtime.allow_cpu_fallback` permits CPU when MPS is unavailable. This local profile does not select CUDA.

For RunPod, use the official Python 3.12/Torch 2.8 template and the separate
CUDA/BF16 profile in [docs/runpod.md](docs/runpod.md): batch 2, accumulation 4,
activation checkpointing, TF32, four workers, pinned memory, a 70 GB Pod working
volume, and a separate 100 GB STANDARD network volume accessed through RunPod
S3. The detached launcher is `./scripts/launch_runpod_training.sh`; ongoing
durability is RunPod-only, with no Mac backup workflow. The mandatory order is
resume probes, deterministic update-200/300 benchmark, strict durable upload,
separate restore drill, then a formal stage.

With `checkpointing.resume: auto`, an existing `checkpoints/silma-saudi/model_last.pt` is restored only when it contains the project model, optimizer, scheduler, EMA, progress, and RNG state. New checkpoints also store a sampler fingerprint and RunPod resumes reject mismatches before applying state. The only missing-fingerprint exception is the known update-100/epoch-0 Mac-to-CUDA migration, which resets its non-portable batch position once. Checkpoints are written under `checkpoints/silma-saudi/`; metrics under `runs/silma-saudi/metrics.jsonl`.

## Inference

Synthesis requires the exact reference WAV and transcript configured in `configs/evaluation.yaml`, and either `--text` or a held-out `--phrase-id`:

```bash
PYTHONPATH=src .venv/bin/python -m src.evaluation.synthesize \
  --checkpoint checkpoints/silma-saudi/model_last.pt \
  --text "هلا والله، وش أخبارك اليوم؟" \
  --output outputs/evaluation/synthesis.wav
```

Use `--cpu` to force CPU, `--nfe-steps` for a bounded synthesis setting, and `--phrase-id` with an ID from `configs/saudi_eval_phrases.jsonl`. The authoritative fine-tuned checkpoint is the stopped Mac update-100 artifact; RunPod probes write only to `outputs/runpod-resume-probe/`.

## Current measured state

- Raw mirror: 253,166 rows / 437.545 h; 31 Parquet shards; all acquired shard hashes match the pinned mirror revision.
- Final processed set: 125,534 accepted clips / 148.722258 h; 119,972 train, 2,717 validation, 2,845 test.
- Accepted dialect counts: 73,387 Najdi; 28,534 Hijazi; 23,613 Khaliji.
- Canonical manifest SHA-256: `047163a1e010f9e87fed12eeec67d331870cf92d7ee4761ff046bb7cab3de194`.
- Processed audio: 24-kHz mono PCM-16; 25,704,751,276 bytes.
- Base assets: SILMA revision `ac81834c5ce305504fe4c7602187042fdd6913db`; F5-TTS source revision `c96c3aeed84f5e02aa54dc42c1193537ead39837`; Vocos revision `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`.

## MPS, CPU, limitations, and troubleshooting

MPS is selected only when available in the host process. Restricted execution reported MPS built but unavailable and therefore selected CPU; documented host evidence showed a finite, no-grad SILMA forward with MPS fallback disabled. MPS backward/optimizer behavior and end-to-end training memory are untested. Start with FP32 and `num_workers: 0`; if MPS is unavailable or unstable, use CPU and preserve the report distinction. Do not enable fallback and infer parity, and do not treat Whisper disagreement as transcript truth.

For failures, first verify `.venv/bin/python --version`, run `.venv/bin/python -m pip check`, inspect `reports/environment_report.md` and `reports/mps_diagnostic.md`, and confirm all asset hashes against `models/base/asset_manifest.json`. Keep raw data immutable and do not delete quarantine material while investigating.

For research decisions and known risks, see [`docs/project_state.md`](docs/project_state.md), [`reports/validation_report.md`](reports/validation_report.md), and [`docs/training_plan.md`](docs/training_plan.md). For provenance and rights boundaries, see [`docs/data_governance.md`](docs/data_governance.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
