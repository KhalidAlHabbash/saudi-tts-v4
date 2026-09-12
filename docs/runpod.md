# RunPod Secure RTX PRO 4500 runbook

This runbook continues the full FP32 Mac update-100 state on the provisioned
Secure RTX PRO 4500 Blackwell with 32 GB VRAM. Provisioning and the initial
integrity-checked upload are complete; this document does not authorize starting
production training. The target is the official `runpod-torch-v280`
template (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`): Python 3.12,
Torch 2.8, and CUDA 12.8.

## Frozen resource and storage design

- Pod `saudi-tts-rtxpro4500` (`dgutezp5s8rnov`) in `EU-RO-1`: one Secure RTX PRO 4500 Blackwell with 32 GB VRAM at $0.72/hour; no distributed training.
- 70 GB persistent Pod working volume mounted at `/workspace`.
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
so the 70 GB working volume has a usable but monitored margin.

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
template's CUDA Torch rather than installing a second Torch wheel. Each run
clears only that project venv so an interrupted install cannot leave a partial
environment. F5-TTS is installed from its pinned wheel with dependency
resolution disabled; a version-guarded patch makes its unused upstream
Trainer/Hugging Face dataset/ASR/plotting imports lazy. The pinned runtime
therefore excludes bitsandbytes, flash-attn, Transformers, Datasets, Gradio,
W&B, Accelerate, and other UI/training stacks not used by this project. Setup
rejects the wrong Python/Torch/CUDA versions, unavailable CUDA, missing BF16
support, forbidden packages, unexpected `pip check` failures, or failed
production imports. The same version guard trims only those upstream metadata
requirements made unavailable by this runtime patch, so ordinary `pip check`
must exit cleanly. It writes:

- `reports/runpod_dependency_diagnostic.json`
- `reports/runpod_environment_diagnostic.json`
- `reports/runpod_environment_freeze.txt`

Live Linux validation of those artifacts remains pending until setup completes.

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

Install the official AWS CLI v2 under the persistent working volume. The fixed
binary location prevents a Pod restart from removing it and is checked before
any live launch, upload, sync, or restore:

```bash
installer_dir="$(mktemp -d)"
cd "$installer_dir"
curl -fsSLo awscliv2.zip https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
unzip -q awscliv2.zip
./aws/install --install-dir /workspace/.local/aws-cli --bin-dir /workspace/.local/bin
/workspace/.local/bin/aws --version
cd /workspace/saudi-tts-finetune
```

Generate the S3 API key in the RunPod Console and store it as the encrypted
managed secret `saudi-tts-durable`. Configure the Pod environment in Console,
then restart the Pod so RunPod injects all four values into the process
environment:

| Pod environment variable | Console value |
| --- | --- |
| `RUNPOD_NETWORK_VOLUME_ID` | The 100 GB network-volume ID |
| `RUNPOD_S3_DATACENTER` | The volume datacenter ID, such as `EU-RO-1` |
| `AWS_ACCESS_KEY_ID` | The RunPod user ID associated with the S3 key |
| `AWS_SECRET_ACCESS_KEY` | `{{ RUNPOD_SECRET_saudi-tts-durable }}` |

Never create `/workspace/.saudi-tts/runtime.env` or any other persistent
plaintext credential file. `/workspace` is MooseFS on this Pod and did not
honor restrictive file modes: a requested `0600` was observed as `0666`. The
scripts reject the legacy path if it exists and never source credential files.
They validate the four injected values in memory, reject an unresolved managed
secret placeholder without printing it, prepend `/workspace/.local/bin` to
`PATH`, and require `/workspace/.local/bin/aws` for every live launch, upload,
sync, or restore. The detached training process inherits the validated values,
allowing automatic fail-closed durable checkpoints to invoke AWS later.

Then validate the archive plan before the first live transfer:

```bash

./scripts/sync_runpod_payload.sh --dry-run
./scripts/sync_runpod_payload.sh
```

Explicit `--dry-run` checkpoint/restore calls preserve their previous ability
to run with test-provided or absent credentials where no S3 request occurs. The
forbidden plaintext path is rejected even during dry-run. Payload `--dry-run`
only validates and prints the two archive components; it does not hash the 30
GiB payload or contact S3.

The live payload command creates two deterministic, uncompressed archives: one
for `data/processed`, `data/splits`, and `models/base`, and one for the much
smaller project files. Splitting them avoids re-uploading 30 GiB when only code
changes. Uncompressed tar is intentional because the audio/model payload is
already compressed and CPU compression would extend paid Pod time.

Each archive is read once to establish its SHA-256 and exact size, then streamed
directly into multipart-capable `aws s3 cp -`; it is never staged as a 30 GiB
file on the 70 GB `/workspace`. The upload stream is hashed and must match the
first pass. The live uploader uses 128 MB parts, two concurrent requests, and
15 standard retry attempts. This stays below RunPod's 500 MB maximum part size
while avoiding the thousands of default 8 MB parts that can fail during
multipart finalization. RunPod S3 strips user-defined SHA metadata, so successful HeadObject
size validation is followed by a full remote stream-download whose byte count
and SHA-256 must both match. No downloaded archive is buffered or written to
disk. Objects are immutable SHA-addressed names with SHA sidecars, and
`payload/LATEST.json` is published only after both components verify. This
reduces 125,538 small-file S3 operations to two archive uploads and shares the
checkpoint sync lock.

Preserve the original fingerprintless migration seed in its deliberately
separate integrity-only namespace:

```bash
./scripts/sync_runpod_legacy_seed.sh \
  checkpoints/silma-saudi/model_last.pt
```

That command accepts only update 100, epoch 0, next batch 800, no sampler
fingerprint, and SHA-256
`7b54ac7c31462d7c24b76e16ed679a95754cd00f7bdc5ab10d86961fe418e82e`.
It uploads one multipart checkpoint object, stream-downloads and hashes all
remote bytes in constant memory, then publishes a SHA sidecar and immutable JSON
integrity record under `legacy-seeds/update-000000100/`. The record explicitly
sets `strict_latest_eligible: false`; this path never reads or publishes
`checkpoints/LATEST.json` and does not relax production resume validation. It
is a disaster-recovery copy of the one-time migration input, not a strict
resumable RunPod checkpoint. Only the first production-order CUDA checkpoint
created by the benchmark may enter strict durable `LATEST.json`.

For checkpoints, the high-level AWS CLI `s3 cp` path is multipart-capable.
Objects use immutable update/SHA names and have SHA-256 sidecars. The script
checks HeadObject size and rejects mismatched SHA metadata when the backend
returns it. Because RunPod strips that metadata, it always stream-downloads the
remote checkpoint through AWS CLI and verifies its byte count and SHA-256 in
constant memory before publishing `checkpoints/LATEST.json`.

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

A live S3 roundtrip passed on 2026-09-12, followed by full streaming readback
verification of the 2,603,204,372-byte legacy update-100 checkpoint and both
payload archives. The published assets archive is 29,015,511,040 bytes with
SHA-256 `729d216f23275b8727584611065b75a62a3116425df67c919eddcccdc746c533`;
the project archive is 1,310,720 bytes with SHA-256
`9abd161704d5c766dac99435491ab4fe206eecea25e3b1f7e7940d06d63ab2b2`.
`payload/LATEST.json` was independently downloaded after its publish-last step.
See `reports/runpod_storage_validation.md` for the concise evidence record.

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
