"""Export an immutable full training checkpoint as inference-only EMA SafeTensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.training.checkpoints import durable_replace, inspect_checkpoint, sha256_file
from src.training.model import _online_state

STAGE4_UPDATE = 104_276
STAGE4_CHECKPOINT_SHA256 = "8acd277aaf007f272c888477d8be2909a4211ef8cd34a7bef8fff7cd3e07d89e"


def export_ema_checkpoint(
    source: Path,
    destination: Path,
    *,
    expected_source_sha256: str,
    expected_update: int,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Validate ``source`` and atomically publish normalized EMA-only weights."""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = sha256_file(source)
    if source_sha256 != expected_source_sha256:
        raise RuntimeError(
            f"Source checkpoint SHA-256 mismatch: expected {expected_source_sha256}, "
            f"found {source_sha256}"
        )
    checkpoint_metadata = inspect_checkpoint(source)
    if checkpoint_metadata["update"] != expected_update:
        raise RuntimeError(
            f"Checkpoint update mismatch: expected {expected_update}, "
            f"found {checkpoint_metadata['update']}"
        )

    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in sorted(_online_state(checkpoint, use_ema=True).items())
    }
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("EMA export produced an empty or non-tensor state dictionary")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        save_file(
            state,
            temporary,
            metadata={
                "format": "pt",
                "weights": "ema",
                "training_update": str(expected_update),
                "source_checkpoint_sha256": source_sha256,
                "base_model": "silma-ai/silma-tts",
            },
        )
        durable_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    with safe_open(destination, framework="pt", device="cpu") as handle:
        exported_keys = list(handle.keys())
        if exported_keys != sorted(state):
            raise RuntimeError("Exported SafeTensors key set does not match the EMA source")
        for key in exported_keys:
            exported = handle.get_tensor(key)
            source_tensor = state[key]
            if exported.shape != source_tensor.shape or exported.dtype != source_tensor.dtype:
                raise RuntimeError(f"Exported tensor metadata mismatch for {key}")
            if not torch.equal(exported, source_tensor):
                raise RuntimeError(f"Exported tensor values differ for {key}")

    result: dict[str, Any] = {
        "format_version": 1,
        "artifact": destination.name,
        "weights": "ema",
        "training_update": expected_update,
        "source_checkpoint_sha256": source_sha256,
        "artifact_sha256": sha256_file(destination),
        "artifact_size_bytes": destination.stat().st_size,
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "base_model": "silma-ai/silma-tts",
        "base_model_revision": "ac81834c5ce305504fe4c7602187042fdd6913db",
        "f5_tts_version": "1.1.7",
        "sample_rate": 24_000,
    }
    metadata_path = (metadata_path or destination.with_name("metadata.json")).resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{result['artifact_sha256']}  {destination.name}\n",
        encoding="ascii",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/stage4/model.safetensors"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--expected-update", type=int, default=STAGE4_UPDATE)
    parser.add_argument("--expected-source-sha256", default=STAGE4_CHECKPOINT_SHA256)
    args = parser.parse_args()
    result = export_ema_checkpoint(
        args.source,
        args.output,
        expected_source_sha256=args.expected_source_sha256,
        expected_update=args.expected_update,
        metadata_path=args.metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
