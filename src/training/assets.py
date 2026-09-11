"""Download and verify immutable acoustic-model and vocoder assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "models/base/asset_manifest.json"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_asset_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != 1 or not manifest.get("assets"):
        raise ValueError(f"Unsupported or empty asset manifest: {path}")
    return manifest


def materialize_assets(*, download: bool, manifest_path: Path = DEFAULT_MANIFEST) -> list[Path]:
    manifest = load_asset_manifest(manifest_path)
    verified: list[Path] = []
    for asset in manifest["assets"]:
        local_dir = PROJECT_ROOT / asset["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        for filename, expected in asset["files"].items():
            target = local_dir / filename
            if download and not target.exists():
                hf_hub_download(
                    repo_id=asset["repo_id"],
                    revision=asset["revision"],
                    filename=filename,
                    local_dir=local_dir,
                )
            if not target.is_file():
                raise FileNotFoundError(
                    f"Missing {target}. Run `python -m src.training.assets --download`."
                )
            actual = sha256(target)
            if actual != expected:
                raise RuntimeError(f"SHA-256 mismatch for {target}: expected {expected}, found {actual}")
            verified.append(target)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing files at immutable revisions")
    args = parser.parse_args()
    verified = materialize_assets(download=args.download)
    for path in verified:
        print(f"verified {path.relative_to(PROJECT_ROOT)} {sha256(path)}")


if __name__ == "__main__":
    main()
