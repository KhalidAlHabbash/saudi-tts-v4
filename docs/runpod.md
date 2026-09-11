# RunPod Community RTX 4090 runbook

This runbook continues the full FP32 Mac update-100 state on one Community RTX
4090 24 GB. It does not authorize provisioning by an agent, uploading without
the user, or starting training. The target is the official `runpod-torch-v280`
template (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`): Python 3.12,
Torch 2.8, and CUDA 12.8.

## Frozen resource and storage design

- One Community RTX 4090 24 GB; no distributed training.
- 75 GB persistent Pod working volume mounted at `/workspace`.
- A separate 100 GB STANDARD RunPod network volume in the same datacenter,
  accessed from the Pod through RunPod's S3-compatible API. Do not attach it at
  `/workspace`, because that would replace the Pod working-volume mount.
- The repository, processed dataset, base model, and checkpoints stay within
  RunPod after the initial upload. There is deliberately no checkpoint-back-to-
  Mac procedure.
- Expose TCP 22 and install the prepared public SSH key. Do not put API or S3
  credentials in the repository.

The working payload is approximately 30.1 GiB. `model_last`, three numbered
checkpoints, and one temporary save need about another 12.1 GiB. The trainer
requires at least 20 GiB free before starting and before every checkpoint save,
so a 75 GB working volume has a usable but monitored margin.

## 1. Verify and transfer the immutable source state

The authoritative source is
`checkpoints/silma-saudi/model_last.pt`: update 100, epoch 0, next batch 800,
about 2.6 GB, with full model/EMA/AdamW/scheduler/RNG state. Confirm there is no
temporary save before the initial transfer:

```bash
ls -lh checkpoints/silma-saudi/model_last.pt
test ! -e checkpoints/silma-saudi/model_last.pt.tmp
shasum -a 256 checkpoints/silma-saudi/model_last.pt
```

After the Pod is created, replace `PORT`, `IP`, and `KEY`. The explicit trailing
destinations are important: they put the base assets and checkpoint exactly
where config and auto-resume expect them.

```bash
ssh -p PORT -i KEY root@IP \
  'mkdir -p /workspace/saudi-tts-finetune/data/processed \
    /workspace/saudi-tts-finetune/data/splits \
    /workspace/saudi-tts-finetune/models/base \
    /workspace/saudi-tts-finetune/checkpoints/silma-saudi'

rsync -az --progress -e 'ssh -p PORT -i KEY' \
  README.md THIRD_PARTY_NOTICES.md pyproject.toml requirements-runpod.txt requirements.txt \
  configs docs reports scripts src tests \
  root@IP:/workspace/saudi-tts-finetune/

rsync -az --progress -e 'ssh -p PORT -i KEY' \
  data/processed/ root@IP:/workspace/saudi-tts-finetune/data/processed/
rsync -az --progress -e 'ssh -p PORT -i KEY' \
  data/splits/ root@IP:/workspace/saudi-tts-finetune/data/splits/
rsync -az --progress -e 'ssh -p PORT -i KEY' \
  models/base/ root@IP:/workspace/saudi-tts-finetune/models/base/
rsync -az --progress -e 'ssh -p PORT -i KEY' \
  checkpoints/silma-saudi/model_last.pt \
  root@IP:/workspace/saudi-tts-finetune/checkpoints/silma-saudi/model_last.pt
```

Verify the checkpoint hash on the Pod against the Mac value. This is an initial
upload integrity check, not an ongoing Mac backup process.

```bash
sha256sum /workspace/saudi-tts-finetune/checkpoints/silma-saudi/model_last.pt
```

## 2. Reuse the template Torch and verify the environment

```bash
cd /workspace/saudi-tts-finetune
./scripts/setup_runpod.sh
.venv-runpod/bin/python -m pytest -q
bash -n scripts/*.sh
```

Setup creates `.venv-runpod` with `--system-site-packages`; it must reuse the
template's CUDA Torch rather than installing a second Torch wheel. It rejects
the wrong Python/Torch/CUDA versions, unavailable CUDA, or missing BF16 support,
runs `pip check`, and writes:

- `reports/runpod_environment_diagnostic.json`
- `reports/runpod_environment_freeze.txt`

Live Linux validation of those artifacts remains pending until the Pod exists.

## 3. Run the isolated production-shaped resume probes

Do not use `--smoke-one-step`: the migration gate requires the real copied
update-100 model, EMA, optimizer, scheduler, and RNG state. This probe selects
worst-case eligible clips at least 7.5 seconds long, checks finite states/losses,
performs two optimizer and EMA updates per profile, saves and strictly reloads a
probe checkpoint, and reports CUDA peak reserved memory. Probe outputs are under
`outputs/runpod-resume-probe/` and cannot be placed inside the authoritative
checkpoint directory.

```bash
cd /workspace/saudi-tts-finetune
./scripts/runpod_resume_probe.sh
```

The script deliberately runs in this order:

1. batch 1, frames threshold 750, accumulation 8;
2. batch 2, frames threshold 1500, accumulation 4.

Both JSON reports must say `GO`, complete at least two optimizer updates, reload
strictly, and report `cuda_peak_reserved_gib <= 21.5`. Any non-finite value,
reload failure, or higher reserved-memory peak is a NO-GO. The probe does not
overwrite `checkpoints/silma-saudi/model_last.pt`.

## 4. Safe ordered gate

Do these steps in order. A later step is blocked until the preceding result is
verified:

1. Complete both resume probes above.
2. Run one deterministic production-order benchmark from update 100 to update
   300. This is exactly 200 real optimizer updates and includes validation at
   update 250 plus final checkpoint overhead. It is a diagnostic stop, not a
   formal stage, and the scheduler horizon remains 120,000:

   ```bash
   ./scripts/launch_runpod_training.sh --benchmark-stop-update 300
   tail -F /workspace/saudi-tts-training.log
   ```

   For the shorter 100-update measurement, use
   `--benchmark-stop-update 200`. The outputs are respectively
   `benchmark_300.pt` or `benchmark_200.pt` plus `model_last.pt`.
3. After exporting the S3 environment described below, publish the benchmark
   checkpoint as the first strict resumable object:

   ```bash
   ./scripts/sync_runpod_checkpoint.sh \
     checkpoints/silma-saudi/benchmark_300.pt --role resumable
   ```

4. Restore that object to a separate drill path and validate the reported
   update/hash before touching the production path:

   ```bash
   mkdir -p outputs/restore-drill
   ./scripts/restore_runpod_checkpoint.sh outputs/restore-drill/model_last.pt
   ```

5. Only after the restore drill passes, launch the first formal stage:

   ```bash
   ./scripts/launch_runpod_training.sh --stage-stop-update 13122
   ```

The original update-100 checkpoint intentionally cannot enter strict durable
`LATEST.json`: it predates sampler fingerprints. The probes leave it untouched;
the deterministic benchmark creates the first production-order CUDA checkpoint
with a RunPod sampler fingerprint, making that artifact eligible for strict
durable upload and restore.

## 5. Establish RunPod-only durable storage

Generate the S3 API key in the RunPod Console. Credentials exist only in the Pod
shell environment; `AWS_ACCESS_KEY_ID` is the RunPod `user_...` id and
`AWS_SECRET_ACCESS_KEY` is the `rps_...` S3 key. The datacenter value is passed
as-is to `--region`; the endpoint hostname is lowercased by the scripts.

```bash
export RUNPOD_NETWORK_VOLUME_ID='<100GB-network-volume-id>'
export RUNPOD_S3_DATACENTER='<DATACENTER-ID>'
export AWS_ACCESS_KEY_ID='user_...'
export AWS_SECRET_ACCESS_KEY='rps_...'

./scripts/sync_runpod_payload.sh --dry-run
./scripts/sync_runpod_payload.sh
```

The payload sync keeps the repository, processed data, split manifests, and
base assets on the 100 GB RunPod network volume. It shares a non-blocking lock
with checkpoint sync so only one durable operation runs at once.

For checkpoints, the high-level AWS CLI `s3 cp` path is multipart-capable.
Objects use immutable update/SHA names, carry SHA metadata, and have SHA-256
sidecars. The script verifies object sizes with HeadObject and checks the remote
sidecar before publishing `checkpoints/LATEST.json`.

In the CUDA production config, durable upload is automatic and synchronous for
every immutable 5,000-update checkpoint and every formal stage checkpoint.
Routine artifacts receive the `resumable` role; stage artifacts receive both
`resumable` and `stage`. Upload happens after the local immutable checkpoint and
`model_last` are durably published. Any upload or verification failure preserves
those local files, terminates the trainer nonzero, and prevents another optimizer
update. Automatic upload is disabled for Mac, smoke, probe, and benchmark paths.

```bash
# Routine resumable checkpoint, every 5,000 updates (2,500 is also acceptable
# if measured upload/checkpoint overhead is small):
./scripts/sync_runpod_checkpoint.sh \
  checkpoints/silma-saudi/model_last.pt --role resumable

# A stage boundary is one object with two retained roles, not two 2.6 GB uploads:
./scripts/sync_runpod_checkpoint.sh \
  checkpoints/silma-saudi/stage_13122.pt --role resumable --role stage

# Optional manually approved labels:
./scripts/sync_runpod_checkpoint.sh CHECKPOINT --role milestone
./scripts/sync_runpod_checkpoint.sh CHECKPOINT --role best
```

`LATEST.json` retains the latest three resumable full checkpoints. Stage,
milestone, and best records are permanent and are never pruned by resumable
retention. An object shared by multiple roles is stored once. A failed upload
cannot advance `LATEST.json`.

Restore drills should target a separate path:

```bash
mkdir -p outputs/restore-drill
./scripts/restore_runpod_checkpoint.sh outputs/restore-drill/model_last.pt
```

Restore downloads to `.part`, checks remote and local size, verifies the SHA-256
sidecar and digest, requires a sampler fingerprint, strictly inspects all resume
keys/update metadata, and only then atomically publishes the destination. For a
real recovery, restore directly to
`checkpoints/silma-saudi/model_last.pt` only when it is absent; overwriting an
existing file requires explicit `--replace`.

If an interrupted AWS CLI transfer leaves an orphan multipart upload, list it
and abort only the confirmed key/upload id:

```bash
aws s3api list-multipart-uploads \
  --bucket "$RUNPOD_NETWORK_VOLUME_ID" \
  --region "$RUNPOD_S3_DATACENTER" \
  --endpoint-url "https://s3api-${RUNPOD_S3_DATACENTER,,}.runpod.io/"

aws s3api abort-multipart-upload \
  --bucket "$RUNPOD_NETWORK_VOLUME_ID" \
  --key '<confirmed-object-key>' --upload-id '<confirmed-upload-id>' \
  --region "$RUNPOD_S3_DATACENTER" \
  --endpoint-url "https://s3api-${RUNPOD_S3_DATACENTER,,}.runpod.io/"
```

A real S3 roundtrip is intentionally pending; local validation covers manifest,
command, locking, retention, fail-closed behavior, and dry-run logic only.

## 6. Launch, monitor, and stop safely

The production profile is BF16 autocast, batch 2, 1,500 frames, accumulation 4,
effective batch 8, activation checkpointing, TF32, four workers, pinned memory,
ordinary AdamW, and fixed FP32 model/EMA/gradient/optimizer state. It does not
enable fused/capturable AdamW or `torch.compile` and uses no GradScaler.

```bash
cd /workspace/saudi-tts-finetune
./scripts/launch_runpod_training.sh
```

The launcher uses `setsid ... </dev/null &`; PID and log files are
`/workspace/saudi-tts-training.pid` and
`/workspace/saudi-tts-training.log`. Do not pipe the launcher through `tee`.

```bash
tail -F /workspace/saudi-tts-training.log
ps -p "$(cat /workspace/saudi-tts-training.pid)" -o pid,etime,stat,cmd
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

The first migration may reset `next_batch` only when the checkpoint lacks a
sampler fingerprint and is exactly update 100 / epoch 0. Every new checkpoint
stores manifest hash/eligible row count, seed, F5 version, frames threshold,
maximum samples, and accumulation. Matching restarts preserve `next_batch`;
any mismatch fails before state is applied.

```bash
./scripts/stop_runpod_training.sh
tail -n 50 /workspace/saudi-tts-training.log
test ! -e checkpoints/silma-saudi/model_last.pt.tmp
```

SIGTERM/SIGINT requests a stop after the next complete optimizer update and an
atomic checkpoint. Do not send SIGKILL while a `.tmp` save exists.

## Formal stages and throughput interpretation

The scheduler horizon remains 120,000 updates for benchmarks and every stage.
The configured first formal stop is 13,122; later approved resumes use
`--stage-stop-update` with one of 26,144, 52,188, 104,276, or 120,000. Stage
boundaries save a durable local `stage_<update>.pt` plus `model_last` without
serializing identical state twice, then synchronously publish the same object
with resumable and stage roles.

Use the deterministic update-300 benchmark for ETA because it includes
validation and checkpoint costs. Durable upload timing is measured separately
immediately afterward. Compare workers 2/4/8 and prefetch 2/4 only in separately
approved benchmark variants, retaining the smallest setting that saturates the
GPU. A probe GO is only a memory/resume gate; it is not throughput or quality
evidence.
