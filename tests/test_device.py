import torch

from utils.device import select_device, selected_device


def test_selects_mps_when_backend_is_built_and_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    choice = select_device()

    assert choice.device.type == "mps"
    assert choice.mps_built is True
    assert choice.mps_available is True


def test_falls_back_to_cpu_when_mps_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    choice = select_device()

    assert choice.device.type == "cpu"
    assert "unavailable" in choice.reason
    assert selected_device().type == "cpu"


def test_cpu_preference_does_not_query_cuda(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    choice = select_device(prefer_mps=False)

    assert choice.device.type == "cpu"
    assert "preference disabled" in choice.reason


def test_cuda_profile_selects_cuda_first(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test GPU")
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    choice = select_device(preference="cuda_then_mps_then_cpu")

    assert choice.device.type == "cuda"
    assert choice.cuda_available is True
    assert "Test GPU" in choice.reason
