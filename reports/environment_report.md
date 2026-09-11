# Local environment report

Captured 2026-09-11 for the project’s Apple-Silicon smoke-validation baseline.

| Item | Observed value |
| --- | --- |
| Host | MacBook Pro (Mac14,6) |
| Chip | Apple M2 Max; 12 cores (8 performance, 4 efficiency) |
| Unified memory | 32 GB |
| OS | macOS 26.5 (build 25F71) |
| Process architecture | arm64 |
| Python | CPython 3.11.7, universal2 executable running arm64 |
| Python package manager | pip 24.3.1 in `.venv` |
| PyTorch | 2.8.0 |
| TorchAudio | 2.8.0 |
| F5-TTS | 1.1.7; package import succeeds |
| PyYAML | 6.0.2; YAML import succeeds |
| `torch.backends.mps.is_built()` | `True` |
| `torch.backends.mps.is_available()` | `False` in this process |
| Selected device | CPU |

## Interpretation

The installed PyTorch distribution includes the MPS backend, but the backend was
not available to this restricted process. Project utilities therefore select
CPU here. A separate normal host-visible check with the same interpreter reports
MPS built/available `True/True` and completes a finite SILMA no-grad forward with
fallback disabled; see `reports/mps_diagnostic.md`. That proves forward
feasibility only, not MPS backward or optimizer support.

The dependency manifest pins `f5-tts==1.1.7`, PyTorch 2.8.0, and the small
project-owned stack. It deliberately excludes FlashAttention, bitsandbytes,
Triton, TensorRT, DeepSpeed, NCCL, and CUDA-specific wheels. The pinned arm64
dependencies resolved and installed successfully in `.venv`; `pip check` found
no broken requirements. PyPI access required an approved networked execution.

`f5-tts==1.1.7` resolved with its upstream dependency declarations, including
`gradio==5.35.0`, `vocos==0.1.0`, and host-built pure-Python/source wheels for
`jieba`, `encodec`, and `transformers_stream_generator`. No compiled CUDA
package was introduced. The bootstrap imports project utilities through its
`src/` path, avoiding an unnecessary editable-install build step.

## Reproducibility lock and F5-TTS artifact attestation

`requirements.txt` remains the readable list of direct project dependencies.
`requirements.lock.txt` is the generated, transitive, hash-locked installation
artifact for CPython 3.11; it records every resolved runtime dependency and is
the default bootstrap input via `pip install --require-hashes`. It was generated
with pip-tools 7.5.1 and pip 24.3.1 after resolving the native arm64 environment.
The lock’s exact installed runtime versions are its package pins; bootstrap tools
(`pip`, `setuptools`, and `wheel`) are deliberately outside the runtime lock.

F5-TTS attestation: the downloaded PyPI universal wheel
`f5_tts-1.1.7-py3-none-any.whl` has SHA-256
`8deb87c8f3968f6f4783e8ec76b255c9d87e29ec6c2e9b5f37b01732c8c80a6a`.
This is one of the lock’s allowed F5-TTS hashes. Its wheel metadata declares
version 1.1.7, MIT license, and homepage `https://github.com/SWivid/F5-TTS`;
it does not embed a Git commit or `direct_url.json`. This attests the PyPI
distribution, not a wheel-to-`c96c3aeed84f5e02aa54dc42c1193537ead39837`
source-commit mapping.

The generated lock includes hashes for all published distributions of each
resolved version, as pip-tools normally does. It therefore enforces version and
artifact integrity but is not a single-wheel-only lock across macOS/Linux or
CPU architectures. The attested F5 wheel hash above is the native environment’s
actual downloaded artifact; regenerate the lock/attestation for a different
Python minor version, platform, or index policy.

## Validation completed

`scripts/bootstrap.sh` completed against the populated `.venv` (including with
`PIP_NO_INDEX=1`), device-selection tests passed (3/3), and `pip check` reports
no broken requirements. No arm64 incompatibility surfaced at package import
time. This report's environment check is dependency/import validation; the
separate host diagnostic adds an MPS no-grad forward, while MPS backward remains
untested.
