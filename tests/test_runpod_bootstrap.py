from __future__ import annotations

import shutil
from importlib.metadata import distribution
from pathlib import Path

import pytest

from scripts.prepare_f5_runtime import PATCHES, VOCOS_PATCHES, patch_runtime, prune_metadata
from scripts.verify_runpod_dependencies import inspect_pip_check


def copy_unpatched_fixture(
    source: Path,
    target: Path,
    patches: dict[str, tuple[tuple[str, str], ...]],
) -> None:
    """Copy upstream files and reverse our patch when the live tree is patched."""
    target.mkdir()
    for relative, replacements in patches.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        text = destination.read_text(encoding="utf-8")
        for before, after in reversed(replacements):
            if after in text:
                text = text.replace(after, before, 1)
            elif before not in text:
                raise AssertionError(f"Fixture source has neither patch form for {relative}: {before!r}")
        destination.write_text(text, encoding="utf-8")


def test_minimal_requirements_exclude_heavy_upstream_stacks() -> None:
    lines = Path("requirements-runpod.txt").read_text(encoding="utf-8").lower().splitlines()
    for package in (
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
    ):
        assert not any(line.startswith(f"{package}==") for line in lines)
    assert not any(line.startswith("torch==") for line in lines)
    assert not any(line.startswith("torchaudio==") for line in lines)
    assert "f5-tts==1.1.7" in lines
    assert "pytest==8.4.2" in lines
    assert "iniconfig==2.3.0" in lines
    assert "pluggy==1.6.0" in lines
    assert "pygments==2.21.0" in lines
    assert "setuptools==80.9.0" in lines
    assert "wheel==0.45.1" in lines


@pytest.mark.parametrize("source_is_patched", [False, True])
def test_f5_runtime_patch_is_exact_and_idempotent(tmp_path: Path, source_is_patched: bool) -> None:
    installed_f5 = Path(distribution("f5-tts").locate_file("f5_tts"))
    source = tmp_path / "source-f5"
    copy_unpatched_fixture(installed_f5, source, PATCHES)
    if source_is_patched:
        assert patch_runtime(source)
    target = tmp_path / "f5_tts"
    copy_unpatched_fixture(source, target, PATCHES)

    assert patch_runtime(target)
    assert patch_runtime(target) == []
    assert "from f5_tts.model.trainer import Trainer" not in (target / "model/__init__.py").read_text()
    first_lines = (target / "infer/utils_infer.py").read_text().splitlines()[:40]
    assert "from transformers import pipeline" not in first_lines

    installed_vocos = Path(distribution("vocos").locate_file("vocos"))
    vocos_source = tmp_path / "source-vocos"
    copy_unpatched_fixture(installed_vocos, vocos_source, VOCOS_PATCHES)
    if source_is_patched:
        assert patch_runtime(vocos_source, VOCOS_PATCHES)
    vocos_target = tmp_path / "vocos"
    copy_unpatched_fixture(vocos_source, vocos_target, VOCOS_PATCHES)
    assert patch_runtime(vocos_target, VOCOS_PATCHES)
    assert patch_runtime(vocos_target, VOCOS_PATCHES) == []
    assert "from encodec import EncodecModel" not in (vocos_target / "feature_extractors.py").read_text().splitlines()[:10]


def test_metadata_pruning_is_exact_and_pip_check_stays_strict(tmp_path: Path) -> None:
    metadata = tmp_path / "METADATA"
    metadata.write_text(
        "Metadata-Version: 2.4\n"
        "Name: example\n"
        "Requires-Dist: torch>=2\n"
        "Requires-Dist: wandb\n"
        "\n"
        "description\n",
        encoding="utf-8",
    )
    assert prune_metadata(metadata, {"wandb"}, "test-v1")
    assert not prune_metadata(metadata, {"wandb"}, "test-v1")
    assert "Requires-Dist: torch>=2" in metadata.read_text()
    assert "Requires-Dist: wandb" not in metadata.read_text()
    assert inspect_pip_check("No broken requirements found.\n") == []
    with pytest.raises(RuntimeError, match="Unexpected pip check failures"):
        inspect_pip_check("f5-tts 1.1.7 requires wandb, which is not installed.")
