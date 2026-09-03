"""Fast, subject-disjoint Sleep-EDF loading for BIOT.

The companion ``prepare_sleep_edf.py`` writes one NumPy array per recording
and a compact manifest.  Workers open those arrays lazily as memory maps, so a
night is not repeatedly deserialized for every 30-second epoch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SLEEP_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
SLEEP_STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
NUM_CLASSES = len(SLEEP_STAGE_NAMES)
SAMPLING_RATE = 100
EPOCH_SECONDS = 30
EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_SECONDS


@dataclass(frozen=True)
class SleepRecording:
    recording: str
    subject: str
    split: str
    samples_path: Path
    labels_path: Path
    onsets_path: Path
    epochs: int


@dataclass(frozen=True)
class SleepEpoch:
    recording: SleepRecording
    index: int
    label: int
    onset_seconds: float


def _resolve_split_name(name: str) -> str:
    return "dev" if name in {"dev", "val", "valid", "validation"} else name


def load_sleep_manifest(data_dir: Path) -> dict[str, list[SleepEpoch]]:
    """Validate and expand a processed Sleep-EDF manifest into epoch refs."""
    data_dir = Path(data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    split_path = data_dir / "subject_split.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; run prepare_sleep_edf.py on the raw EDF files"
        )
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing subject split declaration: {split_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with split_path.open("r", encoding="utf-8") as handle:
        declared_splits = json.load(handle)

    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported Sleep-EDF manifest version: {manifest.get('format_version')}")
    if manifest.get("channels") != list(SLEEP_CHANNELS):
        raise ValueError(f"Expected channels {SLEEP_CHANNELS}, got {manifest.get('channels')}")
    if manifest.get("sampling_rate") != SAMPLING_RATE:
        raise ValueError(f"Expected {SAMPLING_RATE} Hz Sleep-EDF, got {manifest.get('sampling_rate')}")
    if manifest.get("epoch_samples") != EPOCH_SAMPLES:
        raise ValueError(f"Expected {EPOCH_SAMPLES} samples per epoch")

    declared: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for raw_name, subjects in declared_splits.items():
        name = _resolve_split_name(raw_name)
        if name not in declared:
            raise ValueError(f"Unexpected split {raw_name!r} in subject_split.json")
        declared[name].update(str(subject) for subject in subjects)
    if any(not values for values in declared.values()):
        raise ValueError("train/dev/test must each contain at least one subject")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = declared[left] & declared[right]
        if overlap:
            raise ValueError(f"Subject leakage between {left} and {right}: {sorted(overlap)}")

    output: dict[str, list[SleepEpoch]] = {"train": [], "dev": [], "test": []}
    seen_recordings: set[str] = set()
    observed_subjects: dict[str, set[str]] = {name: set() for name in output}
    for item in manifest.get("recordings", []):
        split = _resolve_split_name(str(item["split"]))
        if split not in output:
            raise ValueError(f"Unexpected recording split {split!r}")
        subject = str(item["subject"])
        recording_name = str(item["recording"])
        if recording_name in seen_recordings:
            raise ValueError(f"Duplicate recording in manifest: {recording_name}")
        seen_recordings.add(recording_name)
        if subject not in declared[split]:
            raise ValueError(f"Recording {recording_name} contradicts subject_split.json")

        samples_path = data_dir / item["samples"]
        labels_path = data_dir / item["labels"]
        onsets_path = data_dir / item["onsets"]
        for path in (samples_path, labels_path, onsets_path):
            if not path.is_file():
                raise FileNotFoundError(f"Manifest references missing file: {path}")
        samples = np.load(samples_path, mmap_mode="r")
        labels = np.load(labels_path, mmap_mode="r")
        onsets = np.load(onsets_path, mmap_mode="r")
        epochs = int(item["epochs"])
        if samples.shape != (epochs, len(SLEEP_CHANNELS), EPOCH_SAMPLES):
            raise ValueError(f"Bad sample shape for {recording_name}: {samples.shape}")
        if labels.shape != (epochs,) or onsets.shape != (epochs,):
            raise ValueError(f"Bad label/onset shape for {recording_name}")
        label_values = np.asarray(labels)
        if not np.isin(label_values, np.arange(NUM_CLASSES)).all():
            raise ValueError(f"Invalid sleep-stage label in {recording_name}")

        recording = SleepRecording(
            recording=recording_name,
            subject=subject,
            split=split,
            samples_path=samples_path,
            labels_path=labels_path,
            onsets_path=onsets_path,
            epochs=epochs,
        )
        output[split].extend(
            SleepEpoch(recording, index, int(labels[index]), float(onsets[index]))
            for index in range(epochs)
        )
        observed_subjects[split].add(subject)

    for split, subjects in observed_subjects.items():
        if subjects != declared[split]:
            raise ValueError(
                f"Manifest subjects disagree for {split}: "
                f"observed={sorted(subjects)}, declared={sorted(declared[split])}"
            )
        if not output[split]:
            raise ValueError(f"No epochs found for {split}")
    return output


class SleepEDFDataset(Dataset):
    """Read normalized 2-channel, 30-second epochs from recording memmaps."""

    def __init__(self, epochs: Iterable[SleepEpoch]):
        self.epochs = list(epochs)
        if not self.epochs:
            raise ValueError("SleepEDFDataset requires at least one epoch")
        self._arrays: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.epochs)

    def _samples(self, path: Path) -> np.ndarray:
        array = self._arrays.get(path)
        if array is None:
            array = np.load(path, mmap_mode="r")
            self._arrays[path] = array
        return array

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        reference = self.epochs[index]
        # Copy detaches the tensor from the read-only memory map and permits
        # DataLoader pinning without PyTorch's non-writable-array warning.
        sample = np.array(
            self._samples(reference.recording.samples_path)[reference.index],
            dtype=np.float32,
            copy=True,
        )
        if sample.shape != (len(SLEEP_CHANNELS), EPOCH_SAMPLES):
            raise ValueError(f"Unexpected epoch shape {sample.shape}")
        if not np.isfinite(sample).all():
            raise ValueError(
                f"Non-finite values in {reference.recording.recording} epoch {reference.index}"
            )
        name = f"{reference.recording.recording}@{reference.onset_seconds:.1f}s"
        return torch.from_numpy(sample), torch.tensor(reference.label, dtype=torch.long), name


def class_counts(epochs: Iterable[SleepEpoch]) -> dict[int, int]:
    counts = Counter(epoch.label for epoch in epochs)
    return {label: counts.get(label, 0) for label in range(NUM_CLASSES)}
