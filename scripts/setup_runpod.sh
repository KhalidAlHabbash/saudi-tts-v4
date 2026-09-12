#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python3 - <<'PY'
import sys
import torch
import torchaudio

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"runpod-torch-v280 requires Python 3.12; found {sys.version.split()[0]}")
if torch.__version__.split("+")[0] != "2.8.0":
    raise SystemExit(f"Expected template Torch 2.8.0; found {torch.__version__}")
if torchaudio.__version__.split("+")[0] != "2.8.0":
    raise SystemExit(f"Expected template TorchAudio 2.8.0; found {torchaudio.__version__}")
if not torch.version.cuda or not torch.version.cuda.startswith("12.8"):
    raise SystemExit(f"Expected template CUDA 12.8; found {torch.version.cuda}")
PY

# A clean rebuild makes an interrupted resolver attempt harmless. --clear only
# resets this project venv; the template's system Python and Torch stay intact.
python3 -m venv --clear --system-site-packages .venv-runpod
.venv-runpod/bin/python -m pip install --no-deps --only-binary=:all: setuptools==80.9.0 wheel==0.45.1
.venv-runpod/bin/python -m pip install --no-deps --no-build-isolation -r requirements-runpod.txt
.venv-runpod/bin/python scripts/prepare_f5_runtime.py
PYTHONPATH="$project_root" .venv-runpod/bin/python scripts/verify_runpod_dependencies.py \
  --report reports/runpod_dependency_diagnostic.json

PYTHONPATH="$project_root" .venv-runpod/bin/python - <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio
from src.training.config import load_config
from src.utils.device import select_device

cfg = load_config("configs/train_runpod.yaml")
choice = select_device(preference=cfg["runtime"]["device_preference"])
if choice.device.type != "cuda":
    raise SystemExit("CUDA is unavailable; do not start paid training on this Pod")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This GPU does not support the required BF16 profile")
if torch.__version__.split("+")[0] != "2.8.0" or not torch.version.cuda.startswith("12.8"):
    raise SystemExit("The venv is not reusing the official template Torch 2.8/CUDA 12.8 build")

nvidia_smi = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ],
    check=True,
    text=True,
    capture_output=True,
).stdout.strip()
report = {
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_file": torch.__file__,
    "torchaudio": torchaudio.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "device": torch.cuda.get_device_name(0),
    "nvidia_smi": nvidia_smi,
    "config": "configs/train_runpod.yaml",
    "template_id": "runpod-torch-v280",
    "template_image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "dependency_profile": "minimal-f5-runtime-v1",
}
Path("reports").mkdir(exist_ok=True)
Path("reports/runpod_environment_diagnostic.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

.venv-runpod/bin/python -m pip freeze --all > reports/runpod_environment_freeze.txt
echo "RunPod environment verified; diagnostics written under reports/."
