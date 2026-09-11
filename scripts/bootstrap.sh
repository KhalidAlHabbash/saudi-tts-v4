#!/usr/bin/env bash
# Create a native arm64 CPython 3.11 environment for the pinned local stack.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-$project_root/.venv}"
requirements_file="${REQUIREMENTS_FILE:-$project_root/requirements.lock.txt}"

"$python_bin" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"CPython 3.11 is required; found {sys.version.split()[0]}")
if platform.machine() != "arm64":
    raise SystemExit(f"Native arm64 Python is required; found {platform.machine()}")
PY

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install --require-hashes -r "$requirements_file"
PYTHONPATH="$project_root" "$venv_dir/bin/python" -m src.training.assets --download
"$venv_dir/bin/python" -m pytest "$project_root/tests"
PYTHONPATH="$project_root/src" "$venv_dir/bin/python" - <<'PY'
from importlib.metadata import version

import f5_tts
import torch
import torchaudio
import yaml
from utils.device import select_device

print(f"f5_tts={version('f5-tts')}")
print(f"torch={torch.__version__}")
print(f"torchaudio={torchaudio.__version__}")
print(f"pyyaml={yaml.__version__}")
print(f"mps_built={torch.backends.mps.is_built()}")
print(f"mps_available={torch.backends.mps.is_available()}")
print(f"selected_device={select_device().device.type}")
PY
