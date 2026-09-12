# Stage 2 startup validation — target update 26,144

Status: **PASS — detached training remains active**

- Source checkpoint: verified permanent stage checkpoint at update 13,122.
- Trainer: PID 50,589 on RunPod `saudi-tts-rtxpro4500` (`dgutezp5s8rnov`).
- Runtime: CUDA BF16 on one NVIDIA RTX PRO 4500 Blackwell.
- Hard stop: update 26,144; scheduler horizon remains update 120,000.
- Sampler fingerprint matched: `66c4cfe2bdff2d93327d1b409d45aa8fb15e9e8e56dab2524f1f1e5ae04aa886`.
- Resume boundary normalized from saved epoch 0 / batch 52,088 to epoch 1 / batch 0 before the loader was iterated.
- Audit window: updates 13,123 through 13,601, 479 consecutive records with no gap and all recorded losses/gradient norms finite.
- Validation: update 13,250 loss `0.726376`; update 13,500 loss `0.730929`. The small change is within the previously observed validation noise and does not indicate divergence.
- Rolling recovery checkpoint: update 13,500, epoch 1, next batch 1,512, exact sampler fingerprint, atomic publication, no temporary residue — **PASS**.
- Process/log health: trainer running; no traceback, OOM, CUDA, NCCL, segmentation-fault, killed-process, or exception signature.
- GPU snapshot: approximately 4.47 GiB allocated from 32 GiB; normal utilization, temperature, and power.
- Next durability event: routine immutable network-volume upload at update 15,000. Stage 1 remains permanently stored on the network volume while stage 2 runs.

This report records startup acceptance only. Stage 2 is not complete until the trainer stops at update 26,144 and the permanent stage checkpoint passes upload/readback validation.
