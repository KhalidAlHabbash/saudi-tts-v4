from __future__ import annotations

import json

import torch
from ema_pytorch import EMA
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.training.checkpoints import save_checkpoint, sha256_file
from src.training.export_ema import export_ema_checkpoint


def test_export_ema_checkpoint_is_inference_only_and_exact(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    ema = EMA(model, include_online_model=False)
    source = tmp_path / "stage.pt"
    save_checkpoint(
        source,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        update=104_276,
        epoch=8,
        next_batch=0,
        device=torch.device("cpu"),
        sampler_fingerprint={"format_version": 1, "digest": "test"},
    )
    destination = tmp_path / "model.safetensors"
    result = export_ema_checkpoint(
        source,
        destination,
        expected_source_sha256=sha256_file(source),
        expected_update=104_276,
    )

    exported = load_file(destination)
    expected = {
        key.removeprefix("ema_model."): value
        for key, value in ema.state_dict().items()
        if key.startswith("ema_model.")
    }
    assert set(exported) == set(expected)
    assert all(torch.equal(exported[key], expected[key]) for key in exported)
    assert result["weights"] == "ema"
    assert result["training_update"] == 104_276
    assert result["artifact_sha256"] == sha256_file(destination)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["artifact_sha256"] == result["artifact_sha256"]
    assert metadata["parameter_count"] == sum(value.numel() for value in exported.values())
    assert "optimizer" not in " ".join(exported).lower()
