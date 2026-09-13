"""Build the upload directory for the Hugging Face Stage 4 model repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.training.checkpoints import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dist/huggingface-stage4")
    args = parser.parse_args()
    model_dir = PROJECT_ROOT / "models/stage4"
    model = model_dir / "model.safetensors"
    metadata_path = model_dir / "metadata.json"
    if not model.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Export models/stage4/model.safetensors and metadata.json first")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["artifact_sha256"] != sha256_file(model):
        raise RuntimeError("Stage 4 SafeTensors hash does not match metadata.json")

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(PROJECT_ROOT / "release/huggingface", output)
    shutil.copy2(model, output / "model.safetensors")
    shutil.copy2(metadata_path, output / "metadata.json")
    shutil.copy2(model_dir / "model.safetensors.sha256", output / "model.safetensors.sha256")
    shutil.copy2(
        PROJECT_ROOT / "models/base/silma-tts-v1-ac81834c/vocab.txt",
        output / "vocab.txt",
    )
    shutil.copy2(PROJECT_ROOT / "MODEL_LICENSE.md", output / "MODEL_LICENSE.md")
    shutil.copy2(PROJECT_ROOT / "THIRD_PARTY_NOTICES.md", output / "THIRD_PARTY_NOTICES.md")
    print(output)


if __name__ == "__main__":
    main()
