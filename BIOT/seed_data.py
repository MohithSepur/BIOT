"""SEED three-class emotion data loading for BIOT.

This module reads the standard SEED ``Preprocessed_EEG`` MAT layout.  It keeps
subject, session, and trial identity in every window and converts SEED's
monopolar electrodes to the exact 16 bipolar montage used by BIOT's
``EEG-PREST-16-channels.ckpt``.  No model-backbone code is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import gcd
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset


SEED_CHANNELS = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4",
    "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6",
    "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1",
    "O1", "OZ", "O2", "CB2",
)

BIOT_PREST16_PAIRS = (
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
)
BIOT_PREST16_CHANNELS = tuple(f"{left}-{right}" for left, right in BIOT_PREST16_PAIRS)

_CHANNEL_INDEX = {name: index for index, name in enumerate(SEED_CHANNELS)}
_TRIAL_PATTERN = re.compile(r"(?:^|_)eeg(?P<trial>\d+)$", re.IGNORECASE)
_RECORDING_PATTERN = re.compile(r"^(?P<subject>\d+)_(?P<session>.+)$")


@dataclass(frozen=True)
class SeedWindow:
    mat_path: Path
    variable: str
    subject: int
    session: str
    trial: int
    start_sample: int
    stop_sample: int
    label: int

    @property
    def writeout_fn(self) -> str:
        return (
            f"sub{self.subject:02d}_{self.session}_trial{self.trial:02d}_"
            f"{self.start_sample:08d}-{self.stop_sample:08d}"
        )


def _label_vector(label_file: Path) -> np.ndarray:
    if not label_file.is_file():
        raise FileNotFoundError(
            f"SEED label MAT not found: {label_file}. Pass --label-file if it is elsewhere."
        )
    content = loadmat(label_file)
    if "label" not in content:
        raise ValueError(f"{label_file} has no 'label' variable")
    labels = np.asarray(content["label"]).reshape(-1)
    if labels.size == 0 or not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("SEED labels must contain only -1, 0, and 1")
    return labels.astype(np.int64) + 1


def discover_seed_windows(
    data_dir: Path,
    label_file: Path | None = None,
    source_sampling_rate: int = 200,
    window_seconds: float = 10.0,
    stride_seconds: float = 10.0,
) -> list[SeedWindow]:
    """Index complete fixed-length windows without loading full recordings."""
    data_dir = Path(data_dir).expanduser().resolve()
    label_file = (
        Path(label_file).expanduser().resolve()
        if label_file is not None
        else data_dir / "label.mat"
    )
    if source_sampling_rate <= 0:
        raise ValueError("source_sampling_rate must be positive")
    window_samples = int(round(window_seconds * source_sampling_rate))
    stride_samples = int(round(stride_seconds * source_sampling_rate))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride must produce at least one sample")

    labels = _label_vector(label_file)
    windows: list[SeedWindow] = []
    recordings = sorted(path for path in data_dir.glob("*.mat") if path.resolve() != label_file)
    if not recordings:
        raise FileNotFoundError(f"No SEED recording MAT files found in {data_dir}")

    for mat_path in recordings:
        match = _RECORDING_PATTERN.match(mat_path.stem)
        if match is None:
            continue
        subject = int(match.group("subject"))
        session = match.group("session")
        trial_variables: list[tuple[int, str, tuple[int, ...]]] = []
        for variable, shape, _dtype in whosmat(mat_path):
            trial_match = _TRIAL_PATTERN.search(variable)
            if trial_match is not None:
                trial_variables.append((int(trial_match.group("trial")), variable, shape))

        for trial, variable, shape in sorted(trial_variables):
            if trial < 1 or trial > len(labels):
                raise ValueError(
                    f"{mat_path}:{variable} has trial {trial}, but label.mat has {len(labels)} labels"
                )
            if len(shape) != 2 or shape[0] != len(SEED_CHANNELS):
                raise ValueError(
                    f"{mat_path}:{variable} has shape {shape}; expected [62, samples]"
                )
            for start in range(0, shape[1] - window_samples + 1, stride_samples):
                windows.append(
                    SeedWindow(
                        mat_path=mat_path,
                        variable=variable,
                        subject=subject,
                        session=session,
                        trial=trial,
                        start_sample=start,
                        stop_sample=start + window_samples,
                        label=int(labels[trial - 1]),
                    )
                )
    if not windows:
        raise ValueError("SEED MAT files were found, but no complete windows could be indexed")
    return windows


def split_seed_windows(
    windows: Iterable[SeedWindow], test_subject: int, dev_subject: int
) -> dict[str, list[SeedWindow]]:
    if test_subject == dev_subject:
        raise ValueError("test_subject and dev_subject must differ")
    split = {"train": [], "dev": [], "test": []}
    available: set[int] = set()
    for window in windows:
        available.add(window.subject)
        if window.subject == test_subject:
            split["test"].append(window)
        elif window.subject == dev_subject:
            split["dev"].append(window)
        else:
            split["train"].append(window)
    missing = {test_subject, dev_subject} - available
    if missing:
        raise ValueError(f"Requested split subjects not found: {sorted(missing)}")
    if any(not values for values in split.values()):
        raise ValueError("Every split must contain at least one window")
    return split


def to_biot_prest16(monopolar: np.ndarray) -> np.ndarray:
    if monopolar.ndim != 2 or monopolar.shape[0] != len(SEED_CHANNELS):
        raise ValueError(f"Expected SEED [62, samples], got {monopolar.shape}")
    return np.stack(
        [monopolar[_CHANNEL_INDEX[left]] - monopolar[_CHANNEL_INDEX[right]]
         for left, right in BIOT_PREST16_PAIRS],
        axis=0,
    )


@lru_cache(maxsize=4)
def _load_trial(path: str, variable: str) -> np.ndarray:
    content = loadmat(path, variable_names=[variable])
    if variable not in content:
        raise KeyError(f"{path} has no variable {variable!r}")
    return np.asarray(content[variable])


class SeedBIOTDataset(Dataset):
    def __init__(
        self,
        windows: Iterable[SeedWindow],
        source_sampling_rate: int = 200,
        target_sampling_rate: int = 200,
        normalize: bool = True,
    ):
        self.windows = list(windows)
        self.source_sampling_rate = int(source_sampling_rate)
        self.target_sampling_rate = int(target_sampling_rate)
        self.normalize = normalize
        if self.source_sampling_rate <= 0 or self.target_sampling_rate <= 0:
            raise ValueError("sampling rates must be positive")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        reference = self.windows[index]
        trial = _load_trial(str(reference.mat_path), reference.variable)
        clip = np.asarray(
            trial[:, reference.start_sample:reference.stop_sample], dtype=np.float32
        )
        if not np.isfinite(clip).all():
            raise ValueError(f"Non-finite EEG in {reference.writeout_fn}")
        clip = to_biot_prest16(clip)

        if self.source_sampling_rate != self.target_sampling_rate:
            common = gcd(self.source_sampling_rate, self.target_sampling_rate)
            clip = resample_poly(
                clip,
                up=self.target_sampling_rate // common,
                down=self.source_sampling_rate // common,
                axis=-1,
            ).astype(np.float32, copy=False)

        if self.normalize:
            scale = np.quantile(np.abs(clip), 0.95, axis=-1, keepdims=True)
            clip = clip / np.maximum(scale, 1e-8)
        if not np.isfinite(clip).all():
            raise ValueError(f"Non-finite transformed EEG in {reference.writeout_fn}")
        return (
            torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)),
            torch.tensor(reference.label, dtype=torch.long),
            reference.writeout_fn,
        )


def class_counts(windows: Iterable[SeedWindow]) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for window in windows:
        counts[window.label] += 1
    return counts
