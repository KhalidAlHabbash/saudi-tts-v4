from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


VALIDATOR = Path("scripts/validate_runpod_runtime_env.sh").resolve()
ENTRYPOINTS = (
    Path("scripts/launch_runpod_training.sh"),
    Path("scripts/sync_runpod_checkpoint.sh"),
    Path("scripts/sync_runpod_payload.sh"),
    Path("scripts/sync_runpod_legacy_seed.sh"),
    Path("scripts/restore_runpod_checkpoint.sh"),
)
INJECTED_ENV = {
    "RUNPOD_NETWORK_VOLUME_ID": "volume_test",
    "RUNPOD_S3_DATACENTER": "EU-RO-1",
    "AWS_ACCESS_KEY_ID": "user_test",
    "AWS_SECRET_ACCESS_KEY": "rps_test",
}


def run_validator(
    aws_binary: Path,
    forbidden_file: Path,
    *,
    mode: str = "live",
    injected: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script = """
source "$1"
_validate_runpod_runtime_env "$2" "$3" "$4" || exit $?
printf 'validated:%s:%s:%s\\n' "${RUNPOD_NETWORK_VOLUME_ID:-}" "${RUNPOD_S3_DATACENTER:-}" "$PATH"
"""
    environment = os.environ.copy()
    for name in INJECTED_ENV:
        environment.pop(name, None)
    environment["PATH"] = "/usr/bin:/bin"
    environment.update(injected or {})
    return subprocess.run(
        ["bash", "-c", script, "validator-test", str(VALIDATOR), mode, str(aws_binary), str(forbidden_file)],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )


@pytest.fixture
def fake_aws(tmp_path: Path) -> Path:
    executable = tmp_path / "aws"
    executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    return executable


def test_injected_environment_is_validated_exported_and_path_is_prepended(
    tmp_path: Path, fake_aws: Path
) -> None:
    result = run_validator(fake_aws, tmp_path / "forbidden.env", injected=INJECTED_ENV)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "validated:volume_test:EU-RO-1:/workspace/.local/bin:/usr/bin:/bin\n"
    assert "rps_test" not in result.stdout


@pytest.mark.parametrize("missing", sorted(INJECTED_ENV))
def test_live_validation_rejects_each_missing_injected_value(
    tmp_path: Path, fake_aws: Path, missing: str
) -> None:
    injected = dict(INJECTED_ENV)
    injected.pop(missing)
    result = run_validator(fake_aws, tmp_path / "forbidden.env", injected=injected)
    assert result.returncode != 0
    assert missing in result.stderr
    assert "rps_test" not in result.stderr


@pytest.mark.parametrize(
    "placeholder",
    ["{{ RUNPOD_SECRET_saudi-tts-durable }}", "RUNPOD_SECRET_saudi-tts-durable"],
)
def test_unresolved_managed_secret_placeholder_fails_closed(
    tmp_path: Path, fake_aws: Path, placeholder: str
) -> None:
    injected = dict(INJECTED_ENV, AWS_SECRET_ACCESS_KEY=placeholder)
    result = run_validator(fake_aws, tmp_path / "forbidden.env", injected=injected)
    assert result.returncode != 0
    assert "was not resolved" in result.stderr
    assert placeholder not in result.stderr


def test_persistent_plaintext_credential_file_is_rejected(tmp_path: Path, fake_aws: Path) -> None:
    forbidden = tmp_path / "runtime.env"
    forbidden.write_text("AWS_SECRET_ACCESS_KEY=must-not-be-read\n", encoding="utf-8")
    result = run_validator(fake_aws, forbidden, injected=INJECTED_ENV)
    assert result.returncode != 0
    assert "Refusing persistent plaintext credential file" in result.stderr
    assert "must-not-be-read" not in result.stderr


def test_live_validation_rejects_missing_fixed_aws_cli(tmp_path: Path) -> None:
    result = run_validator(tmp_path / "missing-aws", tmp_path / "forbidden.env", injected=INJECTED_ENV)
    assert result.returncode != 0
    assert "AWS CLI is missing" in result.stderr


def test_explicit_dry_run_preserves_missing_environment_and_aws_behavior(tmp_path: Path) -> None:
    result = run_validator(tmp_path / "missing-aws", tmp_path / "forbidden.env", mode="dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("validated:::/workspace/.local/bin:")


def test_all_entrypoints_validate_injected_environment_without_sourcing_credentials() -> None:
    for entrypoint in ENTRYPOINTS:
        text = entrypoint.read_text(encoding="utf-8")
        assert 'source "$project_root/scripts/validate_runpod_runtime_env.sh"' in text
        assert "validate_runpod_runtime_env" in text
        assert "runtime.env" not in text
    validator = VALIDATOR.read_text(encoding="utf-8")
    assert '"/workspace/.local/bin/aws"' in validator
    assert '"/workspace/.saudi-tts/runtime.env"' in validator
    assert "source \"$forbidden_plaintext_file\"" not in validator
