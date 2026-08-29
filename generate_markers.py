"""Generate EvoBrain-compatible 10-second TUSZ marker files."""

from __future__ import annotations

import argparse
from pathlib import Path


ANNOTATION_SUFFIXES = (".tse_bi", ".csv_bi", ".tse", ".csv")
SEIZURE_KEYWORDS = ("seiz", "fnsz", "gnsz", "cpsz", "spsz", "tcsz")


def annotation_path(edf_path: Path) -> Path | None:
    for suffix in ANNOTATION_SUFFIXES:
        candidate = edf_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def seizure_intervals_and_end(path: Path):
    """Parse the formats accepted by EvoBrain/generate_markers.py:234-254."""
    seizures = []
    max_time = 0.0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            lowered = line.lower()
            if "version" in lowered or line.startswith("#") or not line.strip() or "start_time" in lowered:
                continue
            parts = line.strip().replace(",", " ").split()
            try:
                if len(parts) >= 4 and parts[0].upper() == "TERM":
                    start, end, label = float(parts[1]), float(parts[2]), parts[3]
                elif len(parts) >= 3:
                    start, end, label = float(parts[0]), float(parts[1]), parts[2]
                elif len(parts) >= 2 and any(key in lowered for key in SEIZURE_KEYWORDS):
                    start, end, label = float(parts[0]), float(parts[1]), "seiz"
                else:
                    continue
            except ValueError:
                continue
            max_time = max(max_time, end)
            if label.lower() != "bckg" or "seiz" in lowered:
                seizures.append((start, end))
    return seizures, max_time


def generate_markers_tusz(raw_dir, out_dir, clip_len=10):
    """Write train/dev/test seizure and non-seizure marker manifests."""
    raw_root = Path(raw_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    generated = {}

    for split in ("train", "dev", "test"):
        source = raw_root / split
        if not source.is_dir():
            if split == "test" and (raw_root / "eval").is_dir():
                source = raw_root / "eval"
            else:
                raise FileNotFoundError(f"Required TUSZ split directory is missing: {source}")

        seizure_lines = []
        nonseizure_lines = []
        for edf_path in sorted(source.rglob("*.edf")):
            annotation = annotation_path(edf_path)
            if annotation is None:
                continue
            seizures, max_time = seizure_intervals_and_end(annotation)
            for clip_index in range(int(max_time // clip_len)):
                start = clip_index * clip_len
                end = (clip_index + 1) * clip_len
                label = int(any(max(start, onset) < min(end, offset) for onset, offset in seizures))
                marker = f"{edf_path.stem}.edf_{clip_index}.h5,{label}\n"
                (seizure_lines if label else nonseizure_lines).append(marker)

        (output_root / f"{split}Set_seq2seq_{clip_len}s_sz.txt").write_text(
            "".join(seizure_lines), encoding="utf-8"
        )
        (output_root / f"{split}Set_seq2seq_{clip_len}s_nosz.txt").write_text(
            "".join(nonseizure_lines), encoding="utf-8"
        )
        generated[split] = {
            "seizure": len(seizure_lines),
            "nonseizure": len(nonseizure_lines),
        }
    return generated


def main():
    parser = argparse.ArgumentParser("Generate EvoBrain-compatible TUSZ markers")
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--out_dir", default="./data/file_markers_detection")
    parser.add_argument("--clip_len", type=int, default=10)
    args = parser.parse_args()
    print(generate_markers_tusz(args.raw_data_dir, args.out_dir, args.clip_len))


if __name__ == "__main__":
    main()
