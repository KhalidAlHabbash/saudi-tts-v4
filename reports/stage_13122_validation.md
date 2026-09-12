# First safety stage validation — update 13,122

Status: **PASS**  
Training state: **stopped at the configured stage boundary; stage 2 not launched**

## Completion and process state

- Final trainer marker: `complete update=13122 checkpoint=/workspace/saudi-tts-finetune/checkpoints/silma-saudi/model_last.pt reason=stage scheduler_horizon=120000`.
- The trainer PID exited after that marker. The GPU returned to idle immediately after completion.
- No traceback, OOM, CUDA, NCCL, exception, segmentation fault, or killed-process signature appears in the stage log. TorchAudio deprecation warnings are non-blocking.

## Metrics

- 13,022 unique CUDA training records cover every update from 101 through 13,122 without a gap.
- Every recorded training loss and gradient norm is finite.
- Mean loss: first 500 stage updates `0.726081`; last 500 stage updates `0.683357`.
- All 52 validation records are finite. Validation loss was `0.775904` at update 250, best `0.725440` at update 10,000, and `0.730000` at update 13,000. The recent values show mild noise/plateauing, not divergence.

## Checkpoint durability

- Local checkpoint: `/workspace/saudi-tts-finetune/checkpoints/silma-saudi/stage_13122.pt`.
- Size: `2,603,223,583` bytes.
- SHA-256: `41fb7da89c6a56e4b626c4240dd3deef82f86f8ffadeafdc7e75ce0a784b7fb7`.
- Durable object: `checkpoints/objects/update-000013122/sha256-41fb7da89c6a56e4b626c4240dd3deef82f86f8ffadeafdc7e75ce0a784b7fb7.pt` on RunPod network volume `saudi-tts-durable` (`9rl2cz32m7`).
- Durable manifest roles: `resumable`, `stage`; matching checksum sidecar present.
- A fresh restore to a separate RunPod path completed successfully; restored size and SHA-256 exactly match the source. No partial/temp upload residue was found.

## Functional validation

- Strict metadata load reports update 13,122, seed `20260911`, 104,176 eligible rows, batch size 2, accumulation 4, and the expected manifest/sampler fingerprint.
- EMA checkpoint synthesis completed and produced a finite 3.829-second, 24-kHz mono PCM-16 WAV at `outputs/evaluation/stage_13122_smoke.wav`.
- This confirms checkpoint loading and synthesis mechanically. Human/native-Saudi listening review remains separate from this automated gate.

## Resume-boundary remediation

The checkpoint was written at the exact end of epoch 0 as `next_batch=52088`. The earlier loop would have decoded and discarded those 52,088 completed batches before entering epoch 1. The checkpoint itself is valid and was not rewritten. The training code now normalizes an exact boundary to epoch 1 / batch 0 before iteration and writes future boundary checkpoints canonically.

- Local complete suite: **69 passed**.
- Live RunPod complete suite: **69 passed**.
- Actual checkpoint read-only check: update 13,122, saved `(epoch 0, next batch 52088)`, normalized `(epoch 1, next batch 0)` — **PASS**.

## Independent review

- Stage metrics/process audit (GPT-5.6 Sol, high): **PASS**.
- Checkpoint/durability audit (GPT-5.6 Terra, high): **PASS**.

No blocking issue remains for the first safety stage. Continuing to stage 2 is a separate launch decision.
