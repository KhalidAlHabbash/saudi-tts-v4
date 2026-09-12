#!/usr/bin/env python3
"""Fail closed unless the RunPod venv is the pinned, minimal F5 runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQUIRED_VERSIONS = {
    "ema-pytorch": "0.8.3",
    "f5-tts": "1.1.7",
    "huggingface-hub": "1.31.0",
    "numpy": "1.26.4",
    "pytest": "8.4.2",
    "soundfile": "0.13.1",
    "torchdiffeq": "0.2.5",
    "vocos": "0.1.0",
    "x-transformers": "2.28.4",
}
FORBIDDEN_DISTRIBUTIONS = {
    "accelerate",
    "bitsandbytes",
    "datasets",
    "encodec",
    "flash-attn",
    "gradio",
    "librosa",
    "matplotlib",
    "transformers",
    "wandb",
}
def inspect_pip_check(output: str) -> list[str]:
    """Require a clean pip check after the version-guarded metadata patch."""
    unexpected: list[str] = []
    for line in (candidate.strip() for candidate in output.splitlines()):
        if not line or line == "No broken requirements found.":
            continue
        unexpected.append(line)
    if unexpected:
        raise RuntimeError("Unexpected pip check failures:\n" + "\n".join(unexpected))
    return []


def verify_installed_set() -> None:
    for package, expected in REQUIRED_VERSIONS.items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f"Expected {package} {expected}; found {actual}")
    present = sorted(
        name for name in FORBIDDEN_DISTRIBUTIONS if importlib.util.find_spec(name.replace("-", "_"))
    )
    if present:
        raise RuntimeError(f"Forbidden packages are importable in the RunPod venv: {present}")


def verify_runtime_imports() -> None:
    from ema_pytorch import EMA  # noqa: F401
    from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text  # noqa: F401
    from f5_tts.model import CFM, DiT  # noqa: F401
    from f5_tts.model.dataset import CustomDataset, DynamicBatchSampler, collate_fn  # noqa: F401
    from vocos import Vocos  # noqa: F401

    import src.evaluation.synthesize  # noqa: F401
    import src.training.train  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    verify_installed_set()
    verify_runtime_imports()
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    inspect_pip_check(check.stdout)
    if check.returncode != 0:
        raise RuntimeError(f"pip check exited {check.returncode} without a recognized diagnostic")
    report = {
        "forbidden_distributions": sorted(FORBIDDEN_DISTRIBUTIONS),
        "pip_check_exit_code": check.returncode,
        "pip_check_output": check.stdout.strip(),
        "required_versions": REQUIRED_VERSIONS,
        "status": "ok",
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except PackageNotFoundError as exc:
        raise SystemExit(f"Required package is missing: {exc}") from exc
