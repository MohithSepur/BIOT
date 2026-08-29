"""Resample and channel-order TUSZ EDF recordings as EvoBrain HDF5 files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from math import gcd
import os
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.signal import resample_poly


FREQUENCY = 200
TUSZ_CHANNELS = (
    "EEG FP1", "EEG FP2", "EEG F3", "EEG F4", "EEG C3", "EEG C4",
    "EEG P3", "EEG P4", "EEG O1", "EEG O2", "EEG F7", "EEG F8",
    "EEG T3", "EEG T4", "EEG T5", "EEG T6", "EEG FZ", "EEG CZ", "EEG PZ",
)
MODERN_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
}


def _clean_label(label):
    return label.split("-")[0].strip().upper().replace("EEG ", "")


def ordered_channel_indices(labels):
    cleaned = [_clean_label(label) for label in labels]
    indices = []
    for requested in TUSZ_CHANNELS:
        name = requested.replace("EEG ", "")
        candidates = (name, MODERN_ALIASES.get(name, ""))
        match = next((cleaned.index(candidate) for candidate in candidates if candidate in cleaned), None)
        if match is None:
            raise ValueError(f"Required TUSZ channel {requested!r} is absent")
        indices.append(match)
    return indices


def is_valid_h5(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with h5py.File(path, "r") as handle:
            return (
                "resampled_signal" in handle
                and "resample_freq" in handle
                and handle["resampled_signal"].shape[0] == len(TUSZ_CHANNELS)
                and int(handle["resample_freq"][()]) == FREQUENCY
            )
    except OSError:
        return False


def _default_reader(path):
    import pyedflib

    return pyedflib.EdfReader(str(path))


def _open_with_retry(path, reader_factory, retries=3, delay=1.0):
    error = None
    for _ in range(retries):
        try:
            return reader_factory(path)
        except Exception as exc:  # pragma: no cover - depends on EDF I/O failures
            error = exc
            time.sleep(delay)
    raise RuntimeError(f"Failed to open {path} after {retries} attempts") from error


def _split_name(edf_path, raw_root):
    relative_parts = Path(edf_path).relative_to(raw_root).parts
    for name in ("train", "dev", "test", "eval"):
        if name in relative_parts:
            return "test" if name == "eval" else name
    raise ValueError(f"EDF is not below a train/dev/test split: {edf_path}")


def process_single_file(edf_path, raw_root, save_dir, reader_factory=None):
    """Create one validated, atomic HDF5 file; return its path."""
    edf_path = Path(edf_path)
    raw_root = Path(raw_root)
    destination = Path(save_dir) / _split_name(edf_path, raw_root) / f"{edf_path.stem}.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_valid_h5(destination):
        return destination

    reader_factory = reader_factory or _default_reader
    reader = _open_with_retry(edf_path, reader_factory)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        labels = list(reader.getSignalLabels())
        indices = ordered_channel_indices(labels)
        original_frequency = int(round(reader.getSampleFrequency(0)))
        signal = np.stack([reader.readSignal(index) for index in indices]).astype(np.float32)
        if original_frequency != FREQUENCY:
            common = gcd(original_frequency, FREQUENCY)
            signal = resample_poly(
                signal, FREQUENCY // common, original_frequency // common, axis=-1
            ).astype(np.float32)
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("resampled_signal", data=signal)
            handle.create_dataset("resample_freq", data=FREQUENCY)
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()
        close = getattr(reader, "close", None) or getattr(reader, "_close", None)
        if close is not None:
            close()


def _worker(arguments):
    try:
        path = process_single_file(*arguments)
        return str(path), None
    except Exception as exc:  # pragma: no cover - integration failure logging
        return str(arguments[0]), str(exc)


def resample_all(raw_data_dir, save_dir, num_workers=None):
    raw_root = Path(raw_data_dir)
    files = []
    for split in ("train", "dev", "test"):
        source = raw_root / split
        if not source.is_dir():
            if split == "test" and (raw_root / "eval").is_dir():
                source = raw_root / "eval"
            else:
                raise FileNotFoundError(f"Required TUSZ split directory is missing: {source}")
        files.extend(sorted(source.rglob("*.edf")))

    workers = num_workers if num_workers is not None else min(4, max(1, (os.cpu_count() or 1) // 4))
    failures = []
    if workers == 1:
        results = [_worker((path, raw_root, save_dir)) for path in files]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, (path, raw_root, save_dir)) for path in files]
            results = [future.result() for future in as_completed(futures)]
    for path, error in results:
        if error is not None:
            failures.append({"file": path, "error": error})
    if failures:
        output = Path(save_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "failed_files.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    return {"processed": len(files) - len(failures), "failed": len(failures)}


def main():
    parser = argparse.ArgumentParser("Resample TUSZ EDF signals")
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()
    print(resample_all(args.raw_data_dir, args.save_dir, args.num_workers))


if __name__ == "__main__":
    main()
