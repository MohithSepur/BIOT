"""Checkpoint-aligned BIOT classifier for three-class SEED emotion decoding."""

from __future__ import annotations

from pathlib import Path

import torch

from model.biot import BIOTClassifier
from seed_data import BIOT_PREST16_CHANNELS


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added
        return torch.load(path, map_location="cpu")


def build_seed_biot(
    checkpoint: Path | None,
    freeze_encoder: bool = False,
) -> tuple[BIOTClassifier, dict[str, object]]:
    model = BIOTClassifier(
        emb_size=256,
        heads=8,
        depth=4,
        n_classes=3,
        n_channels=len(BIOT_PREST16_CHANNELS),
        n_fft=200,
        hop_length=100,
    )
    loaded = False
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"BIOT checkpoint not found: {checkpoint}")
        state = _load_checkpoint(checkpoint)
        if not isinstance(state, dict):
            raise ValueError("BIOT checkpoint must contain a state dictionary")
        index = state.get("index")
        tokens = state.get("channel_tokens.weight")
        if index is None or tokens is None:
            raise ValueError("Checkpoint lacks BIOT channel-token tensors")
        if int(index.numel()) != 16 or tuple(tokens.shape) != (16, 256):
            raise ValueError(
                "SEED bipolar adapter requires the 16-channel PREST checkpoint; "
                f"found index={tuple(index.shape)}, tokens={tuple(tokens.shape)}"
            )
        model.biot.load_state_dict(state, strict=True)
        loaded = True

    if freeze_encoder:
        for parameter in model.biot.parameters():
            parameter.requires_grad = False

    report = {
        "checkpoint_loaded": loaded,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "freeze_encoder": freeze_encoder,
        "input_channels": list(BIOT_PREST16_CHANNELS),
        "num_classes": 3,
    }
    return model, report
