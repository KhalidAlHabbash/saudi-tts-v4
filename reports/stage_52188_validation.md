# Stage 3 validation — update 52,188

Status: **PASS**
Next state: **Stage 4 active with a hard stop at update 104,276**

## Completion and metrics

- Stage 3 stopped exactly at update 52,188 with `reason=stage` and the unchanged 120,000-update scheduler horizon.
- Updates 26,145 through 52,188 contain 26,044 unique training records with no gaps. Every recorded loss and gradient norm is finite.
- Mean training loss was `0.679446` over the first 500 Stage 3 updates and `0.672088` over the last 500.
- All 104 Stage 3 validation results are finite. Best validation loss was `0.719321` at update 48,250; the last pre-boundary validation was `0.728843` at update 52,000.
- No traceback, OOM, CUDA, NCCL, segmentation-fault, killed-process, runtime-error, or durable-upload-failure signature was found.

## Checkpoint and durability

- Local checkpoint: `/workspace/saudi-tts-finetune/checkpoints/silma-saudi/stage_52188.pt`.
- Size: `2,603,223,647` bytes.
- SHA-256: `a3f0b212aec4f239aff3201a14fe76d075addc6797f3b06842445167dc0963af`.
- `stage_52188.pt` and `model_last.pt` are the same two-link inode, so no duplicate local 2.5-GB payload is consumed.
- Strict metadata inspection reports update 52,188, epoch 4, next batch 0, 104,176 eligible rows, seed `20260911`, and sampler fingerprint `66c4cfe2bdff2d93327d1b409d45aa8fb15e9e8e56dab2524f1f1e5ae04aa886`.
- Durable RunPod object: `checkpoints/objects/update-000052188/sha256-a3f0b212aec4f239aff3201a14fe76d075addc6797f3b06842445167dc0963af.pt` on network volume `saudi-tts-durable` (`9rl2cz32m7`).
- The synchronous fail-closed upload completed full streaming SHA-256 readback, published the checksum sidecar, and assigned both `resumable` and `stage` roles before the trainer exited.

## Reusable Stage 3 evaluation

- Fixed prompt suite: `configs/checkpoint_eval_prompts.jsonl` (10 prompts: 3 easy, 4 medium, 3 hard).
- The suite has zero exact overlap with train/validation/test and no characters outside the pinned SILMA vocabulary.
- Reference: private evaluator-provided audio; intentionally excluded from the public release.
- Exact Stage 3 EMA weights were loaded from `stage_52188.pt`; generation used 32 NFE steps and deterministic per-prompt seeds 20260911 through 20260920.
- Output: a private ignored directory under `outputs/evaluation/`; intentionally excluded from the public release.
- All 10 WAV files passed manifest-hash, nonempty-audio, mono-channel, and 24-kHz checks.
- The 10 clips, generation manifest, shared prompts, evaluator config, and evaluator code were uploaded as 14 objects under `evaluation/` on network volume `9rl2cz32m7`.

## Stage 4 startup

- Detached PID: `58340`.
- Resume line: `update=52188 epoch=4 next_batch=0`.
- Hard stop: update 104,276; scheduler horizon remains 120,000.
- The first Stage 4 validation passed at update 52,250 with finite loss `0.723697`.
- Fresh CUDA BF16 training updates are finite, with no startup failure or CUDA fallback.
- Stage 4 passed update 52,500 and atomically wrote a fresh `model_last.pt` recovery checkpoint; the immutable `stage_52188.pt` remains a separate unchanged inode.
