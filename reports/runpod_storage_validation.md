# RunPod durable-storage validation

Validated 2026-09-12 against network volume `saudi-tts-durable`
(`9rl2cz32m7`, `EU-RO-1`, 100 GB STANDARD).

## PASS

- A real 28-byte upload/download/compare roundtrip passed.
- The fingerprint-less Mac update-100 migration seed was stored outside strict
  production `checkpoints/LATEST.json`, downloaded as a stream, and verified at
  2,603,204,372 bytes with SHA-256
  `7b54ac7c31462d7c24b76e16ed679a95754cd00f7bdc5ab10d86961fe418e82e`.
- The complete training payload was archived without an archive-sized local
  temporary file. The assets archive contains `data/processed`, `data/splits`,
  and `models/base`; it was read back completely from S3 and verified at
  29,015,511,040 bytes with SHA-256
  `729d216f23275b8727584611065b75a62a3116425df67c919eddcccdc746c533`.
- The project archive contains the repository code, configurations, tests,
  reports, and documentation. It was read back completely and verified at
  1,310,720 bytes with SHA-256
  `9abd161704d5c766dac99435491ab4fe206eecea25e3b1f7e7940d06d63ab2b2`.
- Both SHA sidecars passed content checks. `payload/LATEST.json` was published
  only after both archives verified, then independently downloaded and matched
  the recorded keys, sizes, and hashes.
- The failed first assets attempt left no listed orphan multipart upload.
- Credentials remain injected through the RunPod managed secret; no plaintext
  credential file exists in the repository or persistent workspace.
- The final Pod-side non-training suite passed: 63 tests plus shell syntax
  validation. The only output was known third-party Jieba/TorchAudio
  deprecation warnings.

## Remediation applied

The first 29 GB upload used AWS CLI defaults (8 MB parts and ten concurrent
requests) and failed safely at `CompleteMultipartUpload` with missing parts. No
payload manifest was published. The uploader was changed to 128 MB parts, two
concurrent requests, a four-item queue, 15 standard retry attempts, and extended
CLI timeouts. The retry regenerated the same deterministic source hash and
completed successfully. The 128 MB part size is below RunPod's documented 500
MB maximum.

## Still gated

The legacy update-100 seed intentionally lacks the new sampler fingerprint, so
it cannot become a strict production durable checkpoint. After the authorized
CUDA migration probe/benchmark creates a fingerprinted checkpoint, upload it to
strict `checkpoints/LATEST.json` and restore it into a separate drill path before
formal training. Production training has not been started.
