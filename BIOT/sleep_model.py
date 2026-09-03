"""Sleep-EDF classification boundary around the unchanged BIOT backbone."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from model.biot import BIOTClassifier
from sleep_data import EPOCH_SAMPLES, NUM_CLASSES, SLEEP_CHANNELS


SUPPORTED_PRETRAINED_CHANNELS = (16, 18)
EMBEDDING_SIZE = 256


class Conv1dWithConstraint(nn.Conv1d):
    """Constrained projection used by the repository's BIOT Sleep-EDF script."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(inputs)


class BIOTSleepModel(nn.Module):
    def __init__(self, backbone: BIOTClassifier, projected_channels: int):
        super().__init__()
        self.channel_projection = Conv1dWithConstraint(
            len(SLEEP_CHANNELS), projected_channels, kernel_size=1, max_norm=1.0
        )
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = (len(SLEEP_CHANNELS), EPOCH_SAMPLES)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(
                f"BIOT Sleep-EDF expects [B,{expected[0]},{expected[1]}], "
                f"got {tuple(inputs.shape)}"
            )
        if not inputs.is_floating_point():
            raise TypeError(f"Sleep-EDF input must be floating point, got {inputs.dtype}")
        if not bool(torch.isfinite(inputs).all()):
            raise ValueError("Sleep-EDF input contains NaN or infinity")
        return self.backbone(self.channel_projection(inputs))


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def build_sleep_biot(
    checkpoint: Path | None,
    freeze_encoder: bool = False,
) -> tuple[BIOTSleepModel, dict[str, object]]:
    loaded = False
    resolved_checkpoint: Path | None = None
    state = None
    projected_channels = SUPPORTED_PRETRAINED_CHANNELS[0]
    if checkpoint is not None:
        resolved_checkpoint = Path(checkpoint).expanduser().resolve()
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"BIOT checkpoint not found: {resolved_checkpoint}")
        state = _load_checkpoint(resolved_checkpoint)
        if not isinstance(state, dict):
            raise ValueError("BIOT checkpoint must contain a state dictionary")
        index = state.get("index")
        tokens = state.get("channel_tokens.weight")
        if index is None or tokens is None:
            raise ValueError("Checkpoint lacks BIOT channel-token tensors")
        projected_channels = int(index.numel())
        if projected_channels not in SUPPORTED_PRETRAINED_CHANNELS or tuple(tokens.shape) != (
            projected_channels,
            EMBEDDING_SIZE,
        ):
            raise ValueError(
                "Sleep-EDF adapter requires an official 16- or 18-channel checkpoint; "
                f"found index={tuple(index.shape)}, tokens={tuple(tokens.shape)}"
            )

    backbone = BIOTClassifier(
        emb_size=EMBEDDING_SIZE,
        heads=8,
        depth=4,
        n_classes=NUM_CLASSES,
        n_channels=projected_channels,
        n_fft=200,
        hop_length=100,
    )
    if state is not None:
        backbone.biot.load_state_dict(state, strict=True)
        loaded = True

    model = BIOTSleepModel(backbone, projected_channels)
    if freeze_encoder:
        for parameter in model.backbone.biot.parameters():
            parameter.requires_grad = False

    report = {
        "checkpoint_loaded": loaded,
        "checkpoint": str(resolved_checkpoint) if resolved_checkpoint else None,
        "freeze_encoder": freeze_encoder,
        "raw_channels": list(SLEEP_CHANNELS),
        "projected_channels": projected_channels,
        "num_classes": NUM_CLASSES,
    }
    return model, report
