"""Raw EvoBrain-contract adapter at BIOT's model boundary."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from seizure_data import BIOT_CHB16_CHANNELS


class AdaptedBIOTBatch(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    seq_len: torch.Tensor
    supports: torch.Tensor
    adj_mat: torch.Tensor
    writeout_fn: tuple[str, ...]


class BIOTSeizureOutput(NamedTuple):
    logits: torch.Tensor
    y: torch.Tensor
    seq_len: torch.Tensor
    supports: torch.Tensor
    adj_mat: torch.Tensor
    writeout_fn: tuple[str, ...]


class BIOTRawAdapter(nn.Module):
    """Convert `[B,10,C,200]` raw clips to BIOT's `[B,C,2000]` input."""

    def forward(self, batch) -> AdaptedBIOTBatch:
        if not isinstance(batch, (tuple, list)) or len(batch) != 6:
            raise ValueError("Expected the six-field EvoBrain seizure batch")
        x, y, seq_len, supports, adj_mat, writeout_fn = batch
        if x.dtype != torch.float32 or x.ndim != 4:
            raise ValueError("Expected float32 x with shape [B,10,C,200]")
        if x.shape[1] != 10 or x.shape[-1] != 200:
            raise ValueError(
                "BIOT is raw-only: expected ten one-second steps of 200 samples"
            )
        native = x.permute(0, 2, 1, 3).contiguous()
        native = native.reshape(native.shape[0], native.shape[1], -1)
        return AdaptedBIOTBatch(
            native,
            y,
            seq_len,
            supports,
            adj_mat,
            tuple(str(name) for name in writeout_fn),
        )


class BIOTSeizureModel(nn.Module):
    """Keep the complete contract around an unmodified BIOT classifier."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.adapter = BIOTRawAdapter()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.reshape(logits.shape[0], -1)
        if logits.shape[1] != 1:
            raise ValueError(f"BIOT must emit one seizure logit, got {tuple(logits.shape)}")
        return logits[:, 0]

    def forward_contract(self, batch) -> BIOTSeizureOutput:
        adapted = self.adapter(batch)
        return BIOTSeizureOutput(
            self(adapted.x),
            adapted.y,
            adapted.seq_len,
            adapted.supports,
            adapted.adj_mat,
            adapted.writeout_fn,
        )


def _checkpoint_state(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before `weights_only` was introduced.
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("BIOT checkpoint must contain an encoder state dictionary")
    prefixes = ("model.biot.", "biot.")
    state = {}
    for key, value in checkpoint.items():
        normalized = key
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        state[normalized] = value
    return state


def checkpoint_channel_count(path) -> int:
    state = _checkpoint_state(path)
    weight = state.get("channel_tokens.weight")
    index = state.get("index")
    if weight is None or weight.ndim != 2:
        raise ValueError("Checkpoint has no BIOT channel_tokens.weight")
    if index is not None and index.numel() != weight.shape[0]:
        raise ValueError("Checkpoint index and channel-token row counts disagree")
    return int(weight.shape[0])


def _may_reuse_checkpoint_channels(dataset, checkpoint_rows: int) -> bool:
    return bool(
        checkpoint_rows in (16, 18)  # BIOT/README.md:52-54 checkpoint variants.
        and getattr(dataset, "num_nodes", None) == len(BIOT_CHB16_CHANNELS)
        and getattr(dataset, "channel_order_verified", False)
        and tuple(getattr(dataset, "channel_names", ())) == BIOT_CHB16_CHANNELS
    )


def build_biot_seizure_model(args, train_dataset):
    """Build BIOT and apply the explicit pretrained channel-alignment policy."""
    from model.biot import BIOTClassifier

    detected_channels = int(train_dataset.num_nodes)
    checkpoint_path = getattr(args, "pretrain_model_path", "")
    state = _checkpoint_state(checkpoint_path) if checkpoint_path else None
    if state is not None and "channel_tokens.weight" not in state:
        raise ValueError("Checkpoint has no BIOT channel_tokens.weight")
    checkpoint_rows = int(state["channel_tokens.weight"].shape[0]) if state else None
    reuse_channels = bool(
        state is not None
        and _may_reuse_checkpoint_channels(train_dataset, checkpoint_rows)
    )
    model_channels = checkpoint_rows if reuse_channels else detected_channels
    backbone = BIOTClassifier(
        n_classes=1,
        n_channels=model_channels,
        n_fft=args.token_size,
        hop_length=args.hop_length,
    )

    policy = "random_initialization"
    if state is not None:
        if reuse_channels:
            backbone.biot.load_state_dict(state, strict=True)
            policy = f"reused_verified_first_{detected_channels}_channel_rows"
        else:
            non_channel_state = {
                key: value
                for key, value in state.items()
                if key not in {"channel_tokens.weight", "index"}
            }
            incompatible = backbone.biot.load_state_dict(non_channel_state, strict=False)
            expected_missing = {"channel_tokens.weight", "index"}
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise ValueError(
                    "Checkpoint is incompatible beyond BIOT's channel embeddings: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            policy = f"reinitialized_{detected_channels}_channel_rows"
    return BIOTSeizureModel(backbone), policy
