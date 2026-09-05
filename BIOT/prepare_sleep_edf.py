#!/usr/bin/env python3
"""Convert official Sleep-EDF Expanded EDFs to fast BIOT-ready arrays.

The default ``cassette`` study matches the SleepPhysionet cohort used by the
upstream EEGPT Sleep-EDF preparation.  Splits are assigned once per subject,
so both nights from a subject always remain in the same split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from sleep_data import EPOCH_SAMPLES, EPOCH_SECONDS, SAMPLING_RATE, SLEEP_CHANNELS


ANNOTATION_TO_LABEL = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}


def recording_key(path: Path) -> str:
    """Return the six-character key shared by a PSG/hypnogram pair."""
    name = path.name
    if len(name) < 6 or not name.endswith(".edf"):
        raise ValueError(f"Unexpected Sleep-EDF filename: {name}")
    key = name[:6]
    if not (key.startswith("SC4") or key.startswith("ST7")):
        raise ValueError(f"Cannot identify Sleep-EDF study from {name}")
    return key


def subject_key(key: str) -> str:
    if len(key) != 6 or not (key.startswith("SC4") or key.startswith("ST7")):
        raise ValueError(f"Unexpected recording key: {key}")
    return f"{key[:2]}{key[3:5]}"


def discover_pairs(raw_dir: Path, study: str) -> list[tuple[str, Path, Path]]:
    psgs: dict[str, Path] = {}
    hypnograms: dict[str, Path] = {}
    for path in raw_dir.rglob("*.edf"):
        if path.name.endswith("-PSG.edf"):
            key = recording_key(path)
            if key in psgs:
                raise ValueError(f"Duplicate PSG key {key}: {psgs[key]} and {path}")
            psgs[key] = path
        elif path.name.endswith("-Hypnogram.edf"):
            key = recording_key(path)
            if key in hypnograms:
                raise ValueError(
                    f"Duplicate hypnogram key {key}: {hypnograms[key]} and {path}"
                )
            hypnograms[key] = path

    allowed_prefixes = {"cassette": ("SC",), "telemetry": ("ST",), "both": ("SC", "ST")}
    prefixes = allowed_prefixes[study]
    keys = sorted(key for key in set(psgs) | set(hypnograms) if key.startswith(prefixes))
    missing = [key for key in keys if key not in psgs or key not in hypnograms]
    if missing:
        raise ValueError(f"Unpaired PSG/hypnogram recordings: {missing}")
    if not keys:
        raise FileNotFoundError(f"No {study} Sleep-EDF pairs found below {raw_dir}")
    return [(key, psgs[key], hypnograms[key]) for key in keys]


def make_subject_split(
    subjects: list[str], seed: int, train_fraction: float, dev_fraction: float
) -> dict[str, list[str]]:
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1:
        raise ValueError("train/dev fractions must be between zero and one")
    if train_fraction + dev_fraction >= 1:
        raise ValueError("train_fraction + dev_fraction must be below one")
    ordered = sorted(set(subjects))
    if len(ordered) < 3:
        raise ValueError("At least three unique subjects are required for train/dev/test")
    rng = np.random.RandomState(seed)
    rng.shuffle(ordered)
    train_stop = max(1, int(len(ordered) * train_fraction))
    dev_stop = max(train_stop + 1, train_stop + int(len(ordered) * dev_fraction))
    dev_stop = min(dev_stop, len(ordered) - 1)
    return {
        "train": sorted(ordered[:train_stop]),
        "dev": sorted(ordered[train_stop:dev_stop]),
        "test": sorted(ordered[dev_stop:]),
    }


def expand_annotations(
    onsets: np.ndarray, durations: np.ndarray, descriptions: np.ndarray
) -> list[tuple[float, int]]:
    epochs: list[tuple[float, int]] = []
    for onset, duration, description in zip(onsets, durations, descriptions):
        label = ANNOTATION_TO_LABEL.get(str(description))
        if label is None:
            continue
        count = int(round(float(duration) / EPOCH_SECONDS))
        if count <= 0 or not np.isclose(duration, count * EPOCH_SECONDS, atol=0.1):
            raise ValueError(
                f"Sleep-stage annotation has non-30-second duration: {description} {duration}"
            )
        epochs.extend(
            (float(onset) + offset * EPOCH_SECONDS, label) for offset in range(count)
        )
    epochs.sort(key=lambda item: item[0])
    return epochs


def crop_wake(epochs: list[tuple[float, int]], wake_minutes: int) -> list[tuple[float, int]]:
    sleeping = [onset for onset, label in epochs if label != 0]
    if not sleeping:
        raise ValueError("Hypnogram contains no scored sleep epoch")
    margin_seconds = wake_minutes * 60
    start = sleeping[0] - margin_seconds
    stop = sleeping[-1] + EPOCH_SECONDS + margin_seconds
    return [(onset, label) for onset, label in epochs if start <= onset < stop]


def read_recording(
    psg_path: Path, hypnogram_path: Path, wake_minutes: int, lowpass_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyedflib
    except ImportError as error:
        raise RuntimeError("Install pyEDFlib to preprocess raw Sleep-EDF files") from error

    with pyedflib.EdfReader(str(hypnogram_path)) as reader:
        annotations = reader.readAnnotations()
    epochs = crop_wake(expand_annotations(*annotations), wake_minutes)

    with pyedflib.EdfReader(str(psg_path)) as reader:
        labels = reader.getSignalLabels()
        missing = [channel for channel in SLEEP_CHANNELS if channel not in labels]
        if missing:
            raise ValueError(f"{psg_path.name} lacks channels {missing}; found {labels}")
        indices = [labels.index(channel) for channel in SLEEP_CHANNELS]
        frequencies = [float(reader.getSampleFrequency(index)) for index in indices]
        if any(not np.isclose(value, SAMPLING_RATE) for value in frequencies):
            raise ValueError(f"{psg_path.name} has EEG rates {frequencies}, expected 100 Hz")
        signals = np.stack([reader.readSignal(index) for index in indices])

    if not np.isfinite(signals).all():
        raise ValueError(f"Raw non-finite EEG values in {psg_path}")
    sos = butter(4, lowpass_hz, btype="lowpass", fs=SAMPLING_RATE, output="sos")
    signals = sosfiltfilt(sos, signals, axis=-1)

    samples: list[np.ndarray] = []
    labels_out: list[int] = []
    onsets_out: list[float] = []
    for onset, label in epochs:
        start = int(round(onset * SAMPLING_RATE))
        stop = start + EPOCH_SAMPLES
        if start < 0 or stop > signals.shape[1]:
            continue
        sample = signals[:, start:stop]
        mean = sample.mean(axis=-1, keepdims=True)
        std = sample.std(axis=-1, keepdims=True)
        if np.any(std < 1e-8):
            raise ValueError(f"Near-constant EEG epoch in {psg_path.name} at {onset}s")
        sample = ((sample - mean) / std).astype(np.float32)
        if not np.isfinite(sample).all():
            raise ValueError(f"Normalization produced non-finite values in {psg_path.name}")
        samples.append(sample)
        labels_out.append(label)
        onsets_out.append(onset)
    if not samples:
        raise ValueError(f"No usable epochs in {psg_path}")
    return (
        np.stack(samples),
        np.asarray(labels_out, dtype=np.int64),
        np.asarray(onsets_out, dtype=np.float64),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF Expanded for BIOT")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study", choices=("cassette", "telemetry", "both"), default="cassette")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--crop-wake-minutes", type=int, default=30)
    parser.add_argument("--lowpass-hz", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.crop_wake_minutes < 0:
        raise ValueError("crop-wake-minutes must be non-negative")
    if not 0 < args.lowpass_hz < SAMPLING_RATE / 2:
        raise ValueError("lowpass-hz must lie below the Nyquist frequency")
    pairs = discover_pairs(raw_dir, args.study)
    splits = make_subject_split(
        [subject_key(key) for key, _, _ in pairs],
        args.seed,
        args.train_fraction,
        args.dev_fraction,
    )
    subject_to_split = {
        subject: split for split, subjects in splits.items() for subject in subjects
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    split_path = output_dir / "subject_split.json"
    if not args.overwrite and (manifest_path.exists() or split_path.exists()):
        raise FileExistsError(
            f"{output_dir} already contains metadata; use --overwrite deliberately"
        )

    records = []
    for key, psg_path, hypnogram_path in tqdm(pairs, desc="Preprocessing nights"):
        samples_path = records_dir / f"{key}_samples.npy"
        labels_path = records_dir / f"{key}_labels.npy"
        onsets_path = records_dir / f"{key}_onsets.npy"
        if not args.overwrite and any(
            path.exists() for path in (samples_path, labels_path, onsets_path)
        ):
            raise FileExistsError(f"Processed arrays already exist for {key}")
        samples, labels, onsets = read_recording(
            psg_path, hypnogram_path, args.crop_wake_minutes, args.lowpass_hz
        )
        np.save(samples_path, samples, allow_pickle=False)
        np.save(labels_path, labels, allow_pickle=False)
        np.save(onsets_path, onsets, allow_pickle=False)
        subject = subject_key(key)
        records.append(
            {
                "recording": key,
                "subject": subject,
                "split": subject_to_split[subject],
                "samples": str(samples_path.relative_to(output_dir)),
                "labels": str(labels_path.relative_to(output_dir)),
                "onsets": str(onsets_path.relative_to(output_dir)),
                "epochs": int(labels.size),
                "class_counts": {
                    str(label): int((labels == label).sum()) for label in range(5)
                },
            }
        )

    manifest = {
        "format_version": 1,
        "dataset": "Sleep-EDF Expanded",
        "study": args.study,
        "channels": list(SLEEP_CHANNELS),
        "sampling_rate": SAMPLING_RATE,
        "epoch_seconds": EPOCH_SECONDS,
        "epoch_samples": EPOCH_SAMPLES,
        "normalization": "per_epoch_per_channel_zscore",
        "lowpass_hz": args.lowpass_hz,
        "crop_wake_minutes": args.crop_wake_minutes,
        "split_seed": args.seed,
        "recordings": records,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
    print(f"Processed {len(records)} recordings into {output_dir}")
    print({split: len(subjects) for split, subjects in splits.items()})


if __name__ == "__main__":
    main()
