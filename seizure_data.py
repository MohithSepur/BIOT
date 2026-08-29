"""EvoBrain-compatible raw TUSZ and presegmented CHB-MIT-PKL loaders."""

from __future__ import annotations

from pathlib import Path
import h5py
import pickle
import random

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset


def pin_memory_enabled():
    """Pin batches only for the automatically selected CUDA execution path."""
    return torch.cuda.is_available()


# Mirrored from EvoBrain/constants.py:2-32 and args.py:67-79.
FREQUENCY = 200
WINDOW_SECONDS = 10
TIME_STEP_SECONDS = 1
TUSZ_CHANNELS = (
    "EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4",
    "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7", "EEG F8",
    "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ", "EEG PZ",
)
BIOT_CHB16_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
)
def _require_contract(max_seq_len, time_step_size, use_fft):
    if use_fft:
        raise ValueError("BIOT performs STFT internally and requires use_fft=False")
    if max_seq_len != WINDOW_SECONDS or time_step_size != TIME_STEP_SECONDS:
        raise ValueError("The mirrored contract requires 10-second clips and one-second steps")


def _load_scalar(path, name):
    if path is None:
        raise ValueError(f"{name} path is required when TUSZ standardization is enabled")
    with Path(path).open("rb") as handle:
        value = np.asarray(pickle.load(handle))
    if value.size != 1:
        raise ValueError(f"{name} must contain one scalar, got {value.shape}")
    return float(value.reshape(-1)[0])


def _dynamic_adjacency(eeg_clip, top_k=3):
    norms = np.linalg.norm(eeg_clip, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-8
    normalized = eeg_clip / norms
    adjacency = np.abs(normalized @ normalized.swapaxes(-1, -2)).astype(np.float32)
    for step in range(adjacency.shape[0]):
        np.fill_diagonal(adjacency[step], 1.0)
        if top_k is not None and top_k < adjacency.shape[1]:
            keep = np.argpartition(adjacency[step], -top_k, axis=1)[:, -top_k:]
            mask = np.zeros_like(adjacency[step], dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            adjacency[step] = np.where(mask, adjacency[step], 0.0)
            np.fill_diagonal(adjacency[step], 1.0)
    return torch.from_numpy(adjacency)


def _marker_entries(marker_dir, split, clip_len, seed):
    root = Path(marker_dir)
    seizure_file = root / f"{split}Set_seq2seq_{clip_len}s_sz.txt"
    nonseizure_file = root / f"{split}Set_seq2seq_{clip_len}s_nosz.txt"
    if not seizure_file.exists() or not nonseizure_file.exists():
        raise FileNotFoundError(f"Missing TUSZ marker files: {seizure_file}, {nonseizure_file}")
    seizure = [line.strip() for line in seizure_file.read_text().splitlines() if line.strip()]
    nonseizure = [line.strip() for line in nonseizure_file.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    if split == "train":
        rng.shuffle(seizure)
        rng.shuffle(nonseizure)
        nonseizure = nonseizure[:len(seizure)]
    combined = seizure + nonseizure
    rng.shuffle(combined)
    return [tuple(item.rsplit(",", 1)) for item in combined]


def _resampled_path(input_dir, split, marker_name):
    base_name = marker_name.split(".edf")[0] + ".h5"
    aliases = ("test", "eval") if split == "test" else (split,)
    candidates = [Path(input_dir) / alias / base_name for alias in aliases]
    candidates.append(Path(input_dir) / base_name)
    match = next((path for path in candidates if path.exists()), None)
    if match is None:
        raise FileNotFoundError(f"No resampled HDF5 for {marker_name}; checked {candidates}")
    return match


class TUSZDataset(Dataset):
    def __init__(
        self,
        input_dir,
        marker_dir,
        split,
        max_seq_len=WINDOW_SECONDS,
        time_step_size=TIME_STEP_SECONDS,
        standardize=True,
        mean_path=None,
        std_path=None,
        data_augment=False,
        top_k=3,
        seed=123,
        use_fft=False,
    ):
        _require_contract(max_seq_len, time_step_size, use_fft)
        if input_dir is None or marker_dir is None:
            raise ValueError("input_dir and marker_dir are required for TUSZ")
        self.input_dir = Path(input_dir)
        self.marker_dir = Path(marker_dir)
        self.max_seq_len = int(max_seq_len)
        self.time_step_size = int(time_step_size)
        self.standardize = standardize
        self.mean = _load_scalar(mean_path, "mean") if standardize else None
        self.std = _load_scalar(std_path, "std") if standardize else None
        if self.std == 0:
            raise ValueError("std must be non-zero")
        self.data_augment = data_augment
        self.top_k = top_k
        self.entries = []
        for marker_name, label_text in _marker_entries(self.marker_dir, split, self.max_seq_len, seed):
            clip_index = int(marker_name.rsplit("_", 1)[1].split(".h5")[0])
            self.entries.append(
                (_resampled_path(self.input_dir, split, marker_name), clip_index,
                 int(label_text), marker_name.split(".h5")[0])
            )
        self.num_nodes = len(TUSZ_CHANNELS)
        self.channel_names = TUSZ_CHANNELS
        self.channel_order_verified = True
        sample = self.entries[:min(3000, len(self.entries))]
        positives = sum(label == 1 for _, _, label, _ in sample)
        negatives = len(sample) - positives
        self.pos_weight = float(negatives) / positives if positives else 260.0

    def __len__(self):
        return len(self.entries)

    def _read_window(self, h5_path, clip_index):
        length = self.max_seq_len * FREQUENCY
        start = clip_index * length
        with h5py.File(h5_path, "r") as handle:
            if int(handle["resample_freq"][()]) != FREQUENCY:
                raise ValueError(f"Unexpected resample frequency in {h5_path}")
            window = handle["resampled_signal"][:, start:start + length]
        if window.shape[0] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} TUSZ channels, got {window.shape[0]}")
        if window.shape[-1] < length:
            if window.shape[-1] == 0:
                raise ValueError(f"Empty TUSZ window {h5_path.name}:{clip_index}")
            window = np.pad(window, ((0, 0), (0, length - window.shape[-1])), mode="edge")
        step = self.time_step_size * FREQUENCY
        return np.stack([window[:, offset:offset + step] for offset in range(0, length, step)])

    def __getitem__(self, index):
        h5_path, clip_index, label, writeout_fn = self.entries[index]
        eeg_clip = self._read_window(h5_path, clip_index)
        feature = eeg_clip.copy()
        if self.data_augment:
            pairs = ((0, 1), (2, 3), (10, 11), (4, 5), (12, 13), (14, 15), (8, 9))
            if np.random.choice((True, False)):
                for left, right in pairs:
                    feature[:, [left, right], :] = feature[:, [right, left], :]
            feature *= np.random.uniform(0.8, 1.2)
        if self.standardize:
            feature = (feature - self.mean) / self.std
        adjacency = _dynamic_adjacency(eeg_clip, self.top_k)
        supports = torch.zeros(self.max_seq_len, 2, self.num_nodes, self.num_nodes)
        return (
            torch.as_tensor(feature, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
            torch.tensor([self.max_seq_len], dtype=torch.int64),
            supports,
            adjacency,
            writeout_fn,
        )


class CHBMITPklDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split,
        max_seq_len=WINDOW_SECONDS,
        time_step_size=TIME_STEP_SECONDS,
        use_fft=False,
        declared_channel_order="unknown",
    ):
        _require_contract(max_seq_len, time_step_size, use_fft)
        self.data_dir = Path(data_dir)
        self.max_seq_len = int(max_seq_len)
        self.time_step_size = int(time_step_size)
        aliases = (split, "val") if split == "dev" else (split,)
        split_dir = next((self.data_dir / name for name in aliases if (self.data_dir / name).is_dir()), None)
        self.files = sorted(
            path for path in (split_dir.rglob("*.pkl") if split_dir else self.data_dir.glob("*.pkl"))
            if not path.name.startswith(".")
        )
        self.num_nodes, metadata_names = self._inspect_first()
        self.channel_names = metadata_names
        self.channel_order_verified = metadata_names == BIOT_CHB16_CHANNELS
        if metadata_names is None and declared_channel_order == "biot_process2" and self.num_nodes == 16:
            self.channel_names = BIOT_CHB16_CHANNELS
            self.channel_order_verified = True
        self.pos_weight = self._estimate_pos_weight() if split == "train" and self.files else None

    @staticmethod
    def _decode(data):
        channel_names = None
        if isinstance(data, dict):
            raw = data.get("X", data.get("data"))
            label = int(data.get("y", data.get("label", 0)))
            names = data.get("channel_names", data.get("channels"))
            channel_names = tuple(names) if names is not None else None
        elif isinstance(data, (tuple, list)):
            raw, label = data[0], int(data[1])
        else:
            raw, label = data, 0
        if raw is None:
            raise ValueError("PKL sample does not contain X or data")
        return np.asarray(raw, dtype=np.float32), label, channel_names

    def _read(self, path):
        with path.open("rb") as handle:
            return self._decode(pickle.load(handle))

    def _inspect_first(self):
        if not self.files:
            return 16, None  # EvoBrain/data/dataloader_chb.py:498 fallback.
        try:
            raw, _, names = self._read(self.files[0])
            if raw.ndim == 2:
                channels = raw.shape[0]
            elif raw.ndim == 3 and raw.shape[0] == self.max_seq_len:
                channels = raw.shape[1]
            elif raw.ndim == 3 and raw.shape[1] == self.max_seq_len:
                channels = raw.shape[0]
            else:
                channels = 16
            return channels, names
        except Exception:
            return 16, None

    def _estimate_pos_weight(self):
        sample_size = min(3000, len(self.files))  # EvoBrain/data/dataloader_chb.py:544.
        indices = np.random.RandomState(123).choice(len(self.files), sample_size, replace=False)
        positives = 0
        try:
            for index in indices:
                try:
                    _, label, _ = self._read(self.files[int(index)])
                    positives += int(label == 1)
                except Exception:
                    pass
            negatives = sample_size - positives
            return float(negatives) / positives if positives else 260.0
        except Exception:
            return 260.0

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        raw, label, _ = self._read(path)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if raw.ndim == 2:
            channels, samples = raw.shape
            self.num_nodes = channels
            target = self.max_seq_len * FREQUENCY
            if samples != target:
                raw = resample_poly(raw, target, samples, axis=-1)
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            step = self.time_step_size * FREQUENCY
            eeg_clip = np.stack([raw[:, offset:offset + step] for offset in range(0, target, step)])
        elif raw.ndim == 3:
            eeg_clip = raw.transpose(1, 0, 2) if raw.shape[1] == self.max_seq_len else raw
        else:
            raise ValueError(f"Expected 2-D or 3-D CHB input, got {raw.shape}")
        expected_shape = (self.max_seq_len, self.num_nodes, FREQUENCY)
        if eeg_clip.shape != expected_shape:
            raise ValueError(
                f"Expected presegmented CHB clip {expected_shape}, got {eeg_clip.shape}"
            )
        feature = np.nan_to_num(eeg_clip.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        mean, std = np.mean(feature), np.std(feature)
        if std > 1e-5:
            feature = (feature - mean) / std
        feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.maximum(np.linalg.norm(feature, axis=-1, keepdims=True), 1e-5)
        adjacency = np.abs((feature / norms) @ (feature / norms).swapaxes(-1, -2)).astype(np.float32)
        adjacency = np.clip(np.nan_to_num(adjacency, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        for step in range(adjacency.shape[0]):
            np.fill_diagonal(adjacency[step], 1.0)
        return (
            torch.as_tensor(feature, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
            torch.tensor([self.max_seq_len], dtype=torch.int64),
            torch.empty(0, dtype=torch.float32),
            torch.from_numpy(adjacency),
            path.stem,
        )


def build_seizure_dataloaders(args):
    loaders = {}
    for split in ("train", "dev", "test"):
        if args.dataset == "TUSZ":
            dataset = TUSZDataset(
                args.input_dir,
                args.marker_dir,
                split,
                max_seq_len=args.sample_length,
                standardize=args.standardize,
                mean_path=args.scaler_mean_path,
                std_path=args.scaler_std_path,
                data_augment=args.data_augment if split == "train" else False,
                top_k=args.top_k,
                seed=args.seed,
                use_fft=args.use_fft,
            )
        elif args.dataset == "CHB-MIT":
            dataset = CHBMITPklDataset(
                args.input_dir,
                split,
                max_seq_len=args.sample_length,
                use_fft=args.use_fft,
                declared_channel_order=args.chb_channel_order,
            )
        else:
            raise ValueError(f"Unsupported seizure dataset {args.dataset}")
        key = "val" if split == "dev" else split
        loaders[key] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=pin_memory_enabled(),
        )
    return loaders
