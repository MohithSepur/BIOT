"""SEED three-class emotion data loading for BIOT.

This module reads the standard SEED ``Preprocessed_EEG`` MAT layout.  It keeps
subject, session, and trial identity in every window and converts SEED's
monopolar electrodes to the exact 16 bipolar montage used by BIOT's
``EEG-PREST-16-channels.ckpt``.  No model-backbone code is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import gcd
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Iterable

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import lmdb
except ImportError:  # MAT loading remains usable without the optional LMDB binding.
    lmdb = None


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
_LMDB_KEY_PATTERN = re.compile(
    r"^(?P<recording>.+\.mat)-(?P<trial>\d+)-(?P<segment>\d+)$"
)


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


@dataclass(frozen=True)
class SeedLMDBWindow:
    keys: tuple[str, ...]
    split: str
    recording: str
    subject: int
    trial: int
    start_segment: int
    stop_segment: int
    label: int

    @property
    def writeout_fn(self) -> str:
        return (
            f"{Path(self.recording).stem}_trial{self.trial:02d}_"
            f"segments{self.start_segment:06d}-{self.stop_segment:06d}"
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


def _require_lmdb() -> None:
    if lmdb is None:
        raise ModuleNotFoundError(
            "SEED data.mdb loading requires the small 'lmdb' Python package. "
            "Install it in the environment used to launch run_seed_biot.py."
        )


def is_seed_lmdb(data_dir: Path) -> bool:
    return (Path(data_dir).expanduser() / "data.mdb").is_file()


def _decode_lmdb_record(raw: bytes | None, key: str) -> tuple[np.ndarray, int]:
    if raw is None:
        raise KeyError(f"LMDB record is missing: {key}")
    # The supplied preprocessing script deliberately writes trusted pickle
    # dictionaries. Never point this loader at an untrusted LMDB.
    record = pickle.loads(raw)
    if not isinstance(record, dict) or "sample" not in record or "label" not in record:
        raise ValueError(f"LMDB record {key!r} is not a sample/label dictionary")
    sample = np.asarray(record["sample"])
    label = int(record["label"])
    if sample.shape != (len(SEED_CHANNELS), 1, 200):
        raise ValueError(
            f"LMDB record {key!r} has shape {sample.shape}; expected (62, 1, 200)"
        )
    if label not in (0, 1, 2):
        raise ValueError(f"LMDB record {key!r} has invalid label {label}; expected 0, 1, or 2")
    return sample, label


def _subject_from_recording(recording: str) -> int:
    token = Path(recording).name.split("_", 1)[0]
    if not token.isdigit():
        raise ValueError(
            f"Cannot extract SEED subject from LMDB recording name {recording!r}"
        )
    return int(token)


def discover_seed_lmdb_windows(
    data_dir: Path,
    window_seconds: float = 10.0,
    stride_seconds: float = 10.0,
) -> dict[str, list[SeedLMDBWindow]]:
    """Read the stored split and group consecutive one-second LMDB entries."""
    _require_lmdb()
    data_dir = Path(data_dir).expanduser().resolve()
    if not is_seed_lmdb(data_dir):
        raise FileNotFoundError(f"No data.mdb found below {data_dir}")
    # The supplied preprocessing fixes every record to WINDOW=200 at
    # SFREQ=200, so one stored segment is exactly one second.
    segments_per_window_float = window_seconds
    stride_segments_float = stride_seconds
    segments_per_window = int(round(segments_per_window_float))
    stride_segments = int(round(stride_segments_float))
    if (
        segments_per_window <= 0
        or stride_segments <= 0
        or not np.isclose(segments_per_window_float, segments_per_window)
        or not np.isclose(stride_segments_float, stride_segments)
    ):
        raise ValueError(
            "LMDB window-seconds and stride-seconds must be positive integer "
            "multiples of the stored segment duration"
        )

    environment = lmdb.open(
        str(data_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=True,
    )
    try:
        with environment.begin(write=False) as transaction:
            raw_index = transaction.get(b"__keys__")
            if raw_index is None:
                raise KeyError("LMDB has no '__keys__' split index")
            split_index = pickle.loads(raw_index)
            if not isinstance(split_index, dict):
                raise ValueError("LMDB '__keys__' value is not a split dictionary")
            if "dev" in split_index and "val" not in split_index:
                split_index["val"] = split_index["dev"]
            missing = {"train", "val", "test"} - set(split_index)
            if missing:
                raise ValueError(f"LMDB split index is missing {sorted(missing)}")

            result: dict[str, list[SeedLMDBWindow]] = {
                "train": [], "dev": [], "test": []
            }
            all_seen_keys: set[str] = set()
            for stored_split, output_split in (
                ("train", "train"), ("val", "dev"), ("test", "test")
            ):
                grouped: dict[tuple[str, int], list[tuple[int, str]]] = {}
                seen_keys: set[str] = set()
                for raw_key in split_index[stored_split]:
                    key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
                    if key in seen_keys:
                        raise ValueError(f"Duplicate key {key!r} in LMDB split {stored_split}")
                    if key in all_seen_keys:
                        raise ValueError(f"LMDB key {key!r} appears in more than one split")
                    seen_keys.add(key)
                    all_seen_keys.add(key)
                    match = _LMDB_KEY_PATTERN.fullmatch(key)
                    if match is None:
                        raise ValueError(f"Unexpected SEED LMDB key format: {key!r}")
                    recording = match.group("recording")
                    trial = int(match.group("trial"))
                    segment = int(match.group("segment"))
                    grouped.setdefault((recording, trial), []).append((segment, key))

                for (recording, trial), entries in sorted(grouped.items()):
                    entries.sort(key=lambda item: item[0])
                    label_key = entries[0][1]
                    _sample, trial_label = _decode_lmdb_record(
                        transaction.get(label_key.encode("utf-8")), label_key
                    )
                    consecutive_runs: list[list[tuple[int, str]]] = []
                    for entry in entries:
                        if (
                            not consecutive_runs
                            or entry[0] != consecutive_runs[-1][-1][0] + 1
                        ):
                            consecutive_runs.append([entry])
                        else:
                            consecutive_runs[-1].append(entry)
                    for run in consecutive_runs:
                        for offset in range(
                            0, len(run) - segments_per_window + 1, stride_segments
                        ):
                            candidate = run[offset:offset + segments_per_window]
                            first_segment = candidate[0][0]
                            keys = tuple(key for _segment, key in candidate)
                            result[output_split].append(
                                SeedLMDBWindow(
                                    keys=keys,
                                    split=output_split,
                                    recording=recording,
                                    subject=_subject_from_recording(recording),
                                    trial=trial,
                                    start_segment=first_segment,
                                    stop_segment=first_segment + segments_per_window,
                                    label=trial_label,
                                )
                            )
    finally:
        environment.close()

    if any(not windows for windows in result.values()):
        raise ValueError(
            f"At least one LMDB split produced no complete windows: "
            f"{ {name: len(values) for name, values in result.items()} }"
        )
    subjects = {
        name: {window.subject for window in windows}
        for name, windows in result.items()
    }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = subjects[left] & subjects[right]
        if overlap:
            raise ValueError(f"Subject leakage between {left}/{right}: {sorted(overlap)}")

    split_json_path = data_dir / "subject_split.json"
    if split_json_path.is_file():
        with split_json_path.open("r", encoding="utf-8") as handle:
            declared = json.load(handle)
        if "dev" in declared and "val" not in declared:
            declared["val"] = declared["dev"]
        for declared_name, actual_name in (
            ("train", "train"), ("val", "dev"), ("test", "test")
        ):
            if declared_name not in declared:
                raise ValueError(f"subject_split.json has no {declared_name!r} list")
            expected_subjects = {int(subject) for subject in declared[declared_name]}
            if expected_subjects != subjects[actual_name]:
                raise ValueError(
                    f"subject_split.json disagrees with LMDB __keys__ for {declared_name}: "
                    f"json={sorted(expected_subjects)}, lmdb={sorted(subjects[actual_name])}"
                )
    return result


def to_biot_prest16(monopolar: np.ndarray) -> np.ndarray:
    if monopolar.ndim != 2 or monopolar.shape[0] != len(SEED_CHANNELS):
        raise ValueError(f"Expected SEED [62, samples], got {monopolar.shape}")
    return np.stack(
        [monopolar[_CHANNEL_INDEX[left]] - monopolar[_CHANNEL_INDEX[right]]
         for left, right in BIOT_PREST16_PAIRS],
        axis=0,
    )


def _cache_key(path: Path, variable: str) -> str:
    identity = f"{path.name}\0{variable}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:16]
    safe_variable = re.sub(r"[^A-Za-z0-9_.-]+", "_", variable)
    return f"{path.stem}__{safe_variable}__{digest}"


def _cache_paths(cache_dir: Path, path: Path, variable: str) -> tuple[Path, Path]:
    stem = _cache_key(path, variable)
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def _cache_metadata(
    path: Path,
    variable: str,
    source_sampling_rate: int,
) -> dict[str, object]:
    stat = path.stat()
    return {
        "format_version": 1,
        "source_file": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "variable": variable,
        "source_sampling_rate": source_sampling_rate,
        "channels": list(BIOT_PREST16_CHANNELS),
        "dtype": "float32",
    }


def _valid_cache(array_path: Path, metadata_path: Path, expected: dict[str, object]) -> bool:
    if not array_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
        if actual != expected:
            return False
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        return array.ndim == 2 and array.shape[0] == len(BIOT_PREST16_CHANNELS)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array, dtype=np.float32), allow_pickle=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_save_json(path: Path, content: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(content, handle, sort_keys=True)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare_seed_cache(
    windows: Iterable[SeedWindow],
    cache_dir: Path,
    source_sampling_rate: int = 200,
    rebuild: bool = False,
) -> dict[str, int]:
    """Materialize each MAT trial once as a memory-mappable BIOT montage."""
    if source_sampling_rate <= 0:
        raise ValueError("source_sampling_rate must be positive")
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    trials = {
        (reference.mat_path.resolve(), reference.variable)
        for reference in windows
    }
    created = 0
    reused = 0
    for mat_path, variable in tqdm(
        sorted(trials, key=lambda item: (str(item[0]), item[1])),
        desc="Preparing SEED trial cache",
        unit="trial",
        dynamic_ncols=True,
    ):
        array_path, metadata_path = _cache_paths(cache_dir, mat_path, variable)
        metadata = _cache_metadata(mat_path, variable, source_sampling_rate)
        if not rebuild and _valid_cache(array_path, metadata_path, metadata):
            reused += 1
            continue
        # Do not populate the direct-loader LRU while building hundreds of
        # trials; retaining several full 62-channel MAT arrays wastes memory.
        trial = np.asarray(_read_trial(str(mat_path), variable), dtype=np.float32)
        if not np.isfinite(trial).all():
            raise ValueError(f"Non-finite EEG in {mat_path}:{variable}")
        # Cache at the source rate. Resampling is deliberately still done after
        # window slicing so filter-boundary behavior matches uncached loading.
        transformed = to_biot_prest16(trial)
        if not np.isfinite(transformed).all():
            raise ValueError(f"Non-finite cached EEG in {mat_path}:{variable}")
        _atomic_save_array(array_path, transformed)
        _atomic_save_json(metadata_path, metadata)
        created += 1
    _load_trial.cache_clear()
    _load_cached_trial.cache_clear()
    return {"trials": len(trials), "created": created, "reused": reused}


@lru_cache(maxsize=4)
def _load_trial(path: str, variable: str) -> np.ndarray:
    return _read_trial(path, variable)


def _read_trial(path: str, variable: str) -> np.ndarray:
    content = loadmat(path, variable_names=[variable])
    if variable not in content:
        raise KeyError(f"{path} has no variable {variable!r}")
    return np.asarray(content[variable])


@lru_cache(maxsize=32)
def _load_cached_trial(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


class SeedBIOTDataset(Dataset):
    def __init__(
        self,
        windows: Iterable[SeedWindow],
        source_sampling_rate: int = 200,
        target_sampling_rate: int = 200,
        normalize: bool = True,
        cache_dir: Path | None = None,
    ):
        self.windows = list(windows)
        self.source_sampling_rate = int(source_sampling_rate)
        self.target_sampling_rate = int(target_sampling_rate)
        self.normalize = normalize
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        )
        if self.source_sampling_rate <= 0 or self.target_sampling_rate <= 0:
            raise ValueError("sampling rates must be positive")
        self._cached_paths: dict[tuple[Path, str], Path] = {}
        if self.cache_dir is not None:
            for reference in self.windows:
                key = (reference.mat_path.resolve(), reference.variable)
                if key in self._cached_paths:
                    continue
                array_path, metadata_path = _cache_paths(
                    self.cache_dir, reference.mat_path, reference.variable
                )
                expected = _cache_metadata(
                    reference.mat_path,
                    reference.variable,
                    self.source_sampling_rate,
                )
                if not _valid_cache(array_path, metadata_path, expected):
                    raise RuntimeError(
                        f"Missing or stale SEED cache for {reference.mat_path}:"
                        f"{reference.variable}; run prepare_seed_cache first"
                    )
                self._cached_paths[key] = array_path

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        reference = self.windows[index]
        if self.cache_dir is None:
            trial = _load_trial(str(reference.mat_path), reference.variable)
            clip = np.asarray(
                trial[:, reference.start_sample:reference.stop_sample], dtype=np.float32
            )
        else:
            key = (reference.mat_path.resolve(), reference.variable)
            array_path = self._cached_paths[key]
            trial = _load_cached_trial(str(array_path))
            clip = np.asarray(
                trial[:, reference.start_sample:reference.stop_sample], dtype=np.float32
            )
        if not np.isfinite(clip).all():
            raise ValueError(f"Non-finite EEG in {reference.writeout_fn}")
        if self.cache_dir is None:
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


class SeedLMDBDataset(Dataset):
    """Direct reader for the supplied one-second, 62-channel SEED LMDB."""

    def __init__(
        self,
        data_dir: Path,
        windows: Iterable[SeedLMDBWindow],
        normalize: bool = True,
    ):
        _require_lmdb()
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.windows = list(windows)
        self.normalize = normalize
        self._environment = None
        self._environment_pid: int | None = None
        if not self.windows:
            raise ValueError("SeedLMDBDataset requires at least one complete window")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_environment"] = None
        state["_environment_pid"] = None
        return state

    def _db(self):
        process_id = os.getpid()
        if self._environment is None or self._environment_pid != process_id:
            if self._environment is not None:
                self._environment.close()
            self._environment = lmdb.open(
                str(self.data_dir),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
            )
            self._environment_pid = process_id
        return self._environment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        reference = self.windows[index]
        segments: list[np.ndarray] = []
        with self._db().begin(write=False) as transaction:
            for key in reference.keys:
                sample, label = _decode_lmdb_record(
                    transaction.get(key.encode("utf-8")), key
                )
                if label != reference.label:
                    raise ValueError(
                        f"Label changes inside {reference.writeout_fn}: "
                        f"expected {reference.label}, got {label} at {key}"
                    )
                segments.append(np.asarray(sample[:, 0, :], dtype=np.float32))

        monopolar = np.concatenate(segments, axis=-1)
        if not np.isfinite(monopolar).all():
            raise ValueError(f"Non-finite EEG in {reference.writeout_fn}")
        clip = to_biot_prest16(monopolar)
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


def class_counts(
    windows: Iterable[SeedWindow | SeedLMDBWindow],
) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for window in windows:
        counts[window.label] += 1
    return counts
