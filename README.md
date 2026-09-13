# Saudi TTS V4

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Saudi%20TTS%20V4-yellow)](https://huggingface.co/khalidhabbash/Saudi-tts-v4)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11–3.12-blue)](pyproject.toml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Weights: CC BY--NC--SA 4.0](https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-lightgrey)](MODEL_LICENSE.md)

Saudi TTS V4 is a reference-conditioned text-to-speech model adapted
from [SILMA TTS v1](https://huggingface.co/silma-ai/silma-tts). It targets
Najdi-, Hijazi-, and Khaliji-oriented Arabic, produces 24-kHz speech, and lets
the caller provide a short reference recording at inference time.

The default inference artifact is the **EMA weight set from update 104,276**:
`models/stage4/model.safetensors`. It contains acoustic-model weights only—no
optimizer, scheduler, random-number state, dataset audio, or private reference
recordings.

- Size: 650,704,952 bytes
- SHA-256: `0f89df26517617ae935f8267452f2f196467cb39b80df0af93d13be5ead3d431`

## Architecture

| Component | Design |
| --- | --- |
| Acoustic model | F5-TTS conditional flow matching |
| Backbone | 18-layer DiT, hidden size 768, 12 attention heads |
| Parameters | 162,666,852 acoustic-model parameters |
| Conditioning | User-provided reference audio plus its exact transcript |
| Acoustic representation | 100-channel mel spectrogram |
| Vocoder | Fixed `charactr/vocos-mel-24khz` |
| Output | Mono, 24 kHz |

The model is multi-speaker and reference-conditioned; it does not expose a
fixed list of built-in voices. A clean reference clip guides voice identity and
prosody for each request.

## Training

The data pipeline selected Saudi-labelled Najdi, Hijazi, and Khaliji clips from
the pinned `MohamedRashad/SADA22` mirror. It preserved dialect text, normalized
Unicode and whitespace conservatively, resampled accepted audio to 24-kHz mono,
rejected invalid or annotated samples, and created deterministic held-out
validation and test manifests.

Stage 4 completed at update 104,276:

- 104,176 eligible training clips per complete pass, each at most eight seconds
- eight completed dataset passes
- effective batch size eight (batch two, gradient accumulation four)
- AdamW with gradient clipping, activation checkpointing, and EMA tracking
- CUDA BF16 training on one NVIDIA GPU
- validation every 250 optimizer updates
- fixed 120,000-update scheduler horizon; training intentionally stopped at the
  Stage 4 boundary

The full data audit and training record are in `reports/` and `docs/`. Dataset
audio and full resumable checkpoints are intentionally excluded from Git.

## Quick start from GitHub

The Stage 4 weight is tracked with Git LFS. Install Git LFS before cloning or
run `git lfs pull` in an existing clone.

```bash
git lfs install
git clone https://github.com/KhalidAlHabbash/saudi-tts-finetune.git
cd saudi-tts-finetune
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m src.training.assets --download
```

The asset command downloads and hash-verifies the pinned SILMA configuration,
vocabulary, base checkpoint, and Vocos files. The Stage 4 EMA weight remains the
default inference checkpoint.

## Synthesize speech

Provide all three text/audio inputs explicitly:

1. a reference audio file you have permission to use;
2. its exact transcript;
3. the Arabic text to synthesize.

```bash
PYTHONPATH=. .venv/bin/python -m src.evaluation.synthesize \
  --reference-audio ./my_reference.wav \
  --reference-text "هلا والله، هذا تسجيل مرجعي واضح للصوت." \
  --text "أهلاً وسهلاً، هذه تجربة للنموذج السعودي." \
  --output ./outputs/my_sample.wav
```

The default device order on macOS is MPS then CPU. On a CUDA host, add
`--device-preference cuda_then_mps_then_cpu`. Use `--cpu` to force CPU and
`--nfe-steps` to trade generation speed for quality.

Reference-audio recommendations:

- clean speech with minimal music, reverb, or background noise;
- approximately 5–12 seconds;
- one speaker;
- a verbatim transcript matching the spoken recording;
- WAV is preferred, although the upstream preprocessing utility can handle
  other common formats when FFmpeg is available.

No personal reference audio is included in this repository or the Hugging Face
release.

## Hugging Face inference package

The [Hugging Face model repository](https://huggingface.co/khalidhabbash/Saudi-tts-v4)
contains the same SafeTensors artifact, a standalone `inference.py`, the custom
vocabulary, pinned inference configuration, requirements, and model card.

```bash
hf download khalidhabbash/Saudi-tts-v4 --local-dir ./Saudi-tts-v4
cd ./Saudi-tts-v4
pip install -r requirements.txt
python inference.py \
  --reference-audio ../my_reference.wav \
  --reference-text "هلا والله، هذا تسجيل مرجعي واضح للصوت." \
  --text "أهلاً وسهلاً، هذه تجربة للنموذج السعودي." \
  --output ../output.wav
```

## Reproduce preprocessing or training

The repository retains the audited preprocessing and training implementation:

- `src/data/`: validation, normalization, manifest creation, and splits
- `src/training/`: SILMA/F5 model integration, batching, training, EMA, and
  strict resumable checkpoints
- `src/evaluation/`: single-prompt and fixed-suite synthesis
- `configs/`: local MPS and RunPod CUDA profiles
- `scripts/`: bootstrap, asset verification, guarded launch, and durable storage

The local training entry point is:

```bash
./scripts/train.sh
```

This command starts training and is not needed for inference. Consult
`docs/training_plan.md` and `docs/runpod.md` before attempting a new run.

## Tests

```bash
.venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python -m src.evaluation.synthesize --help
PYTHONPATH=. .venv/bin/python -m src.training.export_ema --help
```

The release exporter verifies the immutable full checkpoint SHA-256, training
update, EMA tensor names, shapes, dtypes, and every tensor value before writing
the SafeTensors artifact and its metadata.

## Limitations and responsible use

- Output quality depends strongly on reference-audio cleanliness and transcript
  accuracy.
- Pronunciation can fail on uncommon names, dense diacritics, long numbers,
  English code switching, and text outside the training domain.
- SADA broadcast metadata lacks stable actor, program, and speaker identifiers;
  strict speaker-disjoint evaluation therefore cannot be claimed.
- Use only reference voices you own or are authorized to synthesize. Do not use
  this project for impersonation, fraud, deceptive media, or misrepresentation.
- Clearly disclose generated audio as synthetic where appropriate.

## Licenses and attribution

- Project code: [MIT](LICENSE)
- Stage 4 EMA weights: [CC BY-NC-SA 4.0](MODEL_LICENSE.md)
- Base model: SILMA TTS v1, Apache-2.0
- F5-TTS runtime: MIT
- Training data: SADA/SADA22, attributed to SDAIA and the Saudi Broadcasting
  Authority under the recorded CC BY-NC-SA 4.0 terms

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact upstream
revisions, hashes, and provenance links.
