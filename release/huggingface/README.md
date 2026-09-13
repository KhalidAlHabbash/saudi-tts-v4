---
language:
- ar
license: cc-by-nc-sa-4.0
library_name: f5-tts
pipeline_tag: text-to-speech
base_model: silma-ai/silma-tts
tags:
- text-to-speech
- arabic
- saudi-arabic
- f5-tts
- voice-cloning
---

# Saudi TTS V4

Saudi TTS V4 is a 162.7M-parameter, reference-conditioned F5-TTS/DiT
acoustic model adapted from [SILMA TTS v1](https://huggingface.co/silma-ai/silma-tts).
It produces 24-kHz Arabic speech and supports Najdi-, Hijazi-, and
Khaliji-oriented text. A fixed Vocos decoder converts the predicted mel
spectrogram to waveform audio.

Source code, preprocessing, training configuration, and validation reports are
available at [KhalidAlHabbash/saudi-tts-finetune](https://github.com/KhalidAlHabbash/saudi-tts-finetune).

This repository contains the inference-only exponential-moving-average (EMA)
weights from training update **104,276**. Optimizer, scheduler, random-number,
and resumable training state are intentionally excluded.

- File: `model.safetensors`
- Size: 650,704,952 bytes
- SHA-256: `0f89df26517617ae935f8267452f2f196467cb39b80df0af93d13be5ead3d431`

## Architecture

- Backbone: F5-TTS conditional flow matching with an 18-layer DiT
- Acoustic parameters: 162,666,852
- Hidden size: 768; attention heads: 12
- Mel channels: 100
- Output sample rate: 24 kHz
- Vocoder: `charactr/vocos-mel-24khz`, pinned to revision
  `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`
- Base checkpoint: `silma-ai/silma-tts`, pinned to revision
  `ac81834c5ce305504fe4c7602187042fdd6913db`

## Training summary

The model was trained on 104,176 eligible Saudi-labelled clips per complete
dataset pass, selected from SADA22 and limited to audio of at most eight
seconds. Training used eight completed passes, an effective batch size of eight,
AdamW, gradient clipping, activation checkpointing, and EMA tracking. The
training profile was CUDA BF16 on one NVIDIA GPU. Validation and test manifests
were held out from optimization.

The source is multi-speaker broadcast audio. Consequently, this is a
reference-conditioned model rather than a single fixed voice. Output identity,
prosody, and quality depend heavily on the reference clip and the accuracy of
its transcript.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

Download this model repository, then provide your own reference recording and
its exact transcript:

```bash
python inference.py \
  --reference-audio ./my_reference.wav \
  --reference-text "هلا والله، هذا تسجيل مرجعي واضح للصوت." \
  --text "أهلاً وسهلاً، هذه تجربة للنموذج السعودي." \
  --output ./output.wav
```

The reference audio is a runtime input and is not bundled with this release.
For best results, use a clean mono recording of roughly 5–12 seconds with
minimal background noise and provide a verbatim Arabic transcript. Use only a
voice you own or have permission to synthesize.

Device selection is automatic in the order CUDA → Apple MPS → CPU. Override it
with `--device cuda`, `--device mps`, or `--device cpu`.

## Limitations

- Pronunciation may be unreliable for diacritics, uncommon names, code
  switching, long numbers, and text outside the Saudi/Arabic training domain.
- The training mirror did not provide stable speaker, actor, or program IDs, so
  strict speaker-disjoint evaluation could not be established.
- Reference conditioning can reproduce identity cues. Obtain consent and do not
  use the model for impersonation, fraud, or deceptive media.
- Generated audio should be disclosed as synthetic when appropriate.
- This release has not been evaluated for safety-critical use.

## Licenses and attribution

The Stage 4 weights are released under CC BY-NC-SA 4.0. SILMA TTS v1 is an
Apache-2.0 upstream model; F5-TTS is MIT-licensed; SADA22 is attributed to SDAIA
and the Saudi Broadcasting Authority. See `THIRD_PARTY_NOTICES.md` for pinned
revisions and complete provenance pointers.
