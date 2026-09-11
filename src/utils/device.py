"""Portable CUDA, Apple Silicon, and CPU device selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSelection:
    """Selected Torch device and the MPS capability facts used to select it."""

    device: torch.device
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    reason: str


def select_device(
    *,
    prefer_mps: bool = True,
    preference: str | None = None,
) -> DeviceSelection:
    """Select a device according to an explicit, validated preference chain.

    ``prefer_mps`` remains for compatibility with the local inference CLI. The
    default path does not query CUDA, preserving the original Mac contract.
    """
    preference = preference or ("mps_then_cpu" if prefer_mps else "cpu")
    supported = {"cuda_then_mps_then_cpu", "mps_then_cpu", "cpu"}
    if preference not in supported:
        raise ValueError(f"Unsupported device preference: {preference!r}")

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available()) if preference.startswith("cuda") else False

    if preference.startswith("cuda") and cuda_available:
        return DeviceSelection(
            device=torch.device("cuda"),
            mps_built=mps_built,
            mps_available=mps_available,
            cuda_available=True,
            reason=f"CUDA is available ({torch.cuda.get_device_name(0)}).",
        )

    if preference != "cpu" and mps_built and mps_available:
        return DeviceSelection(
            device=torch.device("mps"),
            mps_built=mps_built,
            mps_available=mps_available,
            cuda_available=cuda_available,
            reason="MPS backend is built and available.",
        )

    if preference == "cpu":
        reason = "Accelerator preference disabled; using CPU."
    elif preference.startswith("cuda") and not cuda_available and not mps_available:
        reason = "CUDA and MPS are unavailable; using CPU."
    elif not mps_built:
        reason = "PyTorch was not built with MPS; using CPU."
    else:
        reason = "MPS is built but unavailable to this process; using CPU."
    return DeviceSelection(torch.device("cpu"), mps_built, mps_available, cuda_available, reason)


def selected_device() -> torch.device:
    """Return only the selected device for simple call sites."""
    return select_device().device
