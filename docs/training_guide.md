# SILMA/F5-TTS training and evaluation

The project wraps the pinned F5-TTS 1.1.7 model APIs; it does not vendor or
patch upstream model source. `configs/train.yaml` reproduces SILMA v1 Small's
18-layer DiT, character vocabulary, 100-bin/24-kHz mel frontend, ordinary
PyTorch attention, CFM loss, and fixed Vocos decoder. The immutable model and
vocoder files and their SHA-256 values are recorded in
`models/base/asset_manifest.json`.

Bootstrap or re-verify assets:

```bash
./scripts/download_models.sh
PYTHONPATH=. .venv/bin/python -m src.training.assets
```

Safe validation commands that do not update weights:

```bash
PYTHONPATH=. .venv/bin/python -m src.training.train
PYTHONPATH=. .venv/bin/python -m src.training.train --check-model
PYTHONPATH=. .venv/bin/python -m src.training.train --forward-only
PYTHONPATH=. .venv/bin/python -m src.evaluation.phrases
```

An independent reviewer can exercise one optimizer update, isolated under
`checkpoints/smoke/`, with:

```bash
PYTHONPATH=. .venv/bin/python -m src.training.train --smoke-one-step
```

The only launcher for a meaningful fine-tuning run is:

```bash
./scripts/train.sh
```

Fresh adaptation initializes the online model from the base checkpoint's EMA
weights, then starts a new AdamW optimizer, linear warmup/decay schedule, EMA,
and update counter. This is intentionally distinct from resume. Project
checkpoints use F5's `model_state_dict`, `optimizer_state_dict`,
`ema_model_state_dict`, `scheduler_state_dict`, and `update` keys plus exact
epoch/batch/RNG state. `resume: auto` restores all of those only from
`model_last.pt`; a vendor checkpoint is never mistaken for a local resume.

Synthesis needs reference audio and its exact transcript. Defaults are a held-
out validation clip; callers can override both:

```bash
PYTHONPATH=. .venv/bin/python -m src.evaluation.synthesize \
  --checkpoint checkpoints/silma-saudi/model_last.pt \
  --text "هلا والله، وش أخبارك اليوم؟"
```

Apple Silicon starts in FP32 with batch one, accumulation eight, zero workers,
activation checkpointing, and no frozen layers. MPS is selected only when it is
available; otherwise CPU is explicit. CUDA is never selected. For compatibility
discovery only, a reviewer may set `PYTORCH_ENABLE_MPS_FALLBACK=1` before a smoke
command; the real launcher does not silently enable it.

The packaging metadata now directly pins `ema-pytorch`, `huggingface-hub`, and
`vocos` because project-owned modules import them. They were already transitive
members of the clean F5-TTS environment; no installed version changed.
