#!/usr/bin/env python3
"""Trim F5-TTS 1.1.7 eager imports to the runtime paths this project uses."""

from __future__ import annotations

import argparse
import re
from importlib.metadata import Distribution, distribution, version
from pathlib import Path


EXPECTED_VERSION = "1.1.7"
F5_METADATA_OMISSIONS = {
    "accelerate",
    "bitsandbytes",
    "cached-path",
    "datasets",
    "gradio",
    "hydra-core",
    "librosa",
    "matplotlib",
    "pydantic",
    "safetensors",
    "tomli",
    "transformers",
    "transformers-stream-generator",
    "unidecode",
    "wandb",
}
VOCOS_METADATA_OMISSIONS = {"encodec"}
REQUIRES_DIST_RE = re.compile(r"^Requires-Dist: ([A-Za-z0-9_.-]+)")


PATCHES: dict[str, tuple[tuple[str, str], ...]] = {
    "model/__init__.py": (
        (
            "from f5_tts.model.trainer import Trainer\n",
            "# Project runtime does not import the upstream Accelerate/W&B Trainer.\n",
        ),
        (
            '__all__ = ["CFM", "UNetT", "DiT", "MMDiT", "Trainer"]',
            '__all__ = ["CFM", "UNetT", "DiT", "MMDiT"]',
        ),
    ),
    "model/dataset.py": (
        (
            "from datasets import Dataset as Dataset_\nfrom datasets import load_from_disk\n",
            "try:\n"
            "    from datasets import Dataset as Dataset_\n"
            "    from datasets import load_from_disk\n"
            "except ModuleNotFoundError:\n"
            "    Dataset_ = None\n"
            "    load_from_disk = None\n",
        ),
        (
            '    print("Loading dataset ...")',
            "    if Dataset_ is None or load_from_disk is None:\n"
            '        raise RuntimeError("Hugging Face dataset loading is not installed in the minimal RunPod runtime")\n\n'
            '    print("Loading dataset ...")',
        ),
    ),
    "model/modules.py": (
        (
            "from librosa.filters import mel as librosa_mel_fn\n",
            "# BigVGAN-only librosa import is lazy in the project runtime.\n",
        ),
        (
            "    device = waveform.device\n    key = ",
            "    from librosa.filters import mel as librosa_mel_fn\n\n"
            "    device = waveform.device\n    key = ",
        ),
    ),
    "infer/utils_infer.py": (
        (
            'import matplotlib\n\n\nmatplotlib.use("Agg")\n\nimport matplotlib.pylab as plt\n',
            "# Plotting imports are lazy in the project runtime.\n",
        ),
        (
            "from transformers import pipeline\n",
            "# ASR-only Transformers import is lazy in the project runtime.\n",
        ),
        (
            "def initialize_asr_pipeline(device: str = device, dtype=None):\n",
            "def initialize_asr_pipeline(device: str = device, dtype=None):\n"
            "    from transformers import pipeline\n\n",
        ),
        (
            "def save_spectrogram(spectrogram, path):\n",
            "def save_spectrogram(spectrogram, path):\n"
            "    import matplotlib\n\n"
            '    matplotlib.use("Agg")\n'
            "    import matplotlib.pylab as plt\n\n",
        ),
    ),
}

VOCOS_PATCHES: dict[str, tuple[tuple[str, str], ...]] = {
    "feature_extractors.py": (
        (
            "from encodec import EncodecModel\n",
            "# Encodec-only import is lazy in the fixed MelSpectrogram Vocos runtime.\n",
        ),
        (
            "        super().__init__()\n        if encodec_model == ",
            "        super().__init__()\n        from encodec import EncodecModel\n\n        if encodec_model == ",
        ),
    ),
}


def f5_package_root() -> Path:
    actual = version("f5-tts")
    if actual != EXPECTED_VERSION:
        raise RuntimeError(f"Expected f5-tts {EXPECTED_VERSION}; found {actual}")
    return Path(distribution("f5-tts").locate_file("f5_tts"))


def patch_runtime(
    root: Path, patches: dict[str, tuple[tuple[str, str], ...]] = PATCHES
) -> list[Path]:
    """Apply exact, idempotent patches and reject an unexpected upstream tree."""
    changed: list[Path] = []
    for relative, replacements in patches.items():
        path = root / relative
        original = path.read_text(encoding="utf-8")
        updated = original
        for before, after in replacements:
            if after in updated:
                continue
            if before not in updated:
                raise RuntimeError(f"Refusing to patch unexpected F5-TTS source: {path} lacks {before!r}")
            updated = updated.replace(before, after, 1)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)
    return changed


def metadata_path(dist: Distribution) -> Path:
    matches = [dist.locate_file(item) for item in dist.files or () if str(item).endswith(".dist-info/METADATA")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one METADATA file for {dist.metadata['Name']}; found {matches}")
    return Path(matches[0])


def prune_metadata(path: Path, omissions: set[str], marker: str) -> bool:
    """Remove intentionally unsupported requirements so plain pip check stays strict."""
    marker_line = f"X-Saudi-TTS-Runtime-Patch: {marker}"
    lines = path.read_text(encoding="utf-8").splitlines()
    if marker_line in lines:
        return False
    found: set[str] = set()
    kept: list[str] = []
    for line in lines:
        match = REQUIRES_DIST_RE.match(line)
        name = match.group(1).lower().replace("_", "-") if match else None
        if name in omissions and "extra ==" not in line:
            found.add(name)
            continue
        kept.append(line)
    missing = omissions - found
    if missing:
        raise RuntimeError(f"Refusing to patch unexpected package metadata {path}; missing {sorted(missing)}")
    header_end = kept.index("")
    kept.insert(header_end, marker_line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="F5 package root; defaults to the installed 1.1.7 package")
    args = parser.parse_args()
    root = args.root if args.root is not None else f5_package_root()
    changed = patch_runtime(root)
    f5_dist = distribution("f5-tts")
    vocos_dist = distribution("vocos")
    vocos_root = Path(vocos_dist.locate_file("vocos"))
    vocos_changed = patch_runtime(vocos_root, VOCOS_PATCHES)
    metadata_changed = prune_metadata(
        metadata_path(f5_dist), F5_METADATA_OMISSIONS, "minimal-f5-runtime-v1"
    )
    vocos_metadata_changed = prune_metadata(
        metadata_path(vocos_dist), VOCOS_METADATA_OMISSIONS, "fixed-melspec-runtime-v1"
    )
    print(f"F5 runtime import patch verified ({len(changed)} files changed): {root}")
    print(f"Vocos runtime import patch verified ({len(vocos_changed)} files changed): {vocos_root}")
    print(f"Runtime metadata verified (changed={metadata_changed or vocos_metadata_changed})")


if __name__ == "__main__":
    main()
