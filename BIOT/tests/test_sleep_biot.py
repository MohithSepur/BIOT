from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_sleep_edf import crop_wake, expand_annotations, make_subject_split, recording_key, subject_key
from run_sleep_biot import load_run_checkpoint, save_run_checkpoint
from sleep_data import EPOCH_SAMPLES, SLEEP_CHANNELS, SleepEDFDataset, class_counts, load_sleep_manifest
from sleep_model import build_sleep_biot


CHECKPOINT = ROOT / "pretrained-models" / "EEG-PREST-16-channels.ckpt"


class SleepBIOTTest(unittest.TestCase):
    def _write_recording(self, root: Path, name: str, subject: str, split: str):
        records = root / "records"
        records.mkdir(exist_ok=True)
        samples = np.random.RandomState(int(subject[-2:])).normal(
            size=(5, len(SLEEP_CHANNELS), EPOCH_SAMPLES)
        ).astype(np.float32)
        labels = np.arange(5, dtype=np.int64)
        onsets = np.arange(5, dtype=np.float64) * 30
        np.save(records / f"{name}_samples.npy", samples, allow_pickle=False)
        np.save(records / f"{name}_labels.npy", labels, allow_pickle=False)
        np.save(records / f"{name}_onsets.npy", onsets, allow_pickle=False)
        return {
            "recording": name,
            "subject": subject,
            "split": split,
            "samples": f"records/{name}_samples.npy",
            "labels": f"records/{name}_labels.npy",
            "onsets": f"records/{name}_onsets.npy",
            "epochs": 5,
        }

    def test_processed_manifest_and_dataset_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declarations = {"train": ["SC00"], "dev": ["SC01"], "test": ["SC02"]}
            records = [
                self._write_recording(root, "SC4001", "SC00", "train"),
                self._write_recording(root, "SC4011", "SC01", "dev"),
                self._write_recording(root, "SC4021", "SC02", "test"),
            ]
            (root / "subject_split.json").write_text(json.dumps(declarations))
            manifest = {
                "format_version": 1,
                "channels": list(SLEEP_CHANNELS),
                "sampling_rate": 100,
                "epoch_samples": EPOCH_SAMPLES,
                "recordings": records,
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            split = load_sleep_manifest(root)
            self.assertEqual({name: len(items) for name, items in split.items()}, {
                "train": 5,
                "dev": 5,
                "test": 5,
            })
            self.assertEqual(class_counts(split["train"]), {0: 1, 1: 1, 2: 1, 3: 1, 4: 1})
            sample, label, name = SleepEDFDataset(split["train"])[3]
            self.assertEqual(tuple(sample.shape), (2, 3000))
            self.assertEqual(sample.dtype, torch.float32)
            self.assertEqual(label.dtype, torch.long)
            self.assertEqual(label.item(), 3)
            self.assertEqual(name, "SC4001@90.0s")

    def test_subject_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = self._write_recording(root, "SC4001", "SC00", "train")
            declarations = {"train": ["SC00"], "dev": ["SC00"], "test": ["SC02"]}
            (root / "subject_split.json").write_text(json.dumps(declarations))
            manifest = {
                "format_version": 1,
                "channels": list(SLEEP_CHANNELS),
                "sampling_rate": 100,
                "epoch_samples": EPOCH_SAMPLES,
                "recordings": [record],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Subject leakage"):
                load_sleep_manifest(root)

    def test_annotation_expansion_crop_and_subject_parsing(self):
        epochs = expand_annotations(
            np.asarray([0.0, 120.0, 180.0]),
            np.asarray([120.0, 60.0, 120.0]),
            np.asarray(["Sleep stage W", "Sleep stage 2", "Sleep stage W"]),
        )
        self.assertEqual(len(epochs), 10)
        self.assertEqual(crop_wake(epochs, wake_minutes=1), epochs[2:8])
        self.assertEqual(recording_key(Path("SC4001E0-PSG.edf")), "SC4001")
        self.assertEqual(recording_key(Path("SC4001EC-Hypnogram.edf")), "SC4001")
        self.assertEqual(subject_key("SC4001"), "SC00")
        split = make_subject_split([f"SC{i:02d}" for i in range(10)], 7, 0.6, 0.2)
        self.assertEqual({name: len(values) for name, values in split.items()}, {
            "train": 6,
            "dev": 2,
            "test": 2,
        })
        self.assertEqual(len(set(split["train"]) & set(split["test"])), 0)

    def test_actual_biot_checkpoint_forward_backward_optimizer_step(self):
        self.assertTrue(CHECKPOINT.is_file())
        model, report = build_sleep_biot(CHECKPOINT)
        self.assertTrue(report["checkpoint_loaded"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        inputs = torch.randn(1, 2, EPOCH_SAMPLES)
        labels = torch.tensor([2])
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1, 5))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_18_channel_checkpoint_is_derived_not_assumed(self):
        checkpoint = ROOT / "pretrained-models" / "EEG-SHHS+PREST-18-channels.ckpt"
        self.assertTrue(checkpoint.is_file())
        model, report = build_sleep_biot(checkpoint)
        self.assertEqual(report["projected_channels"], 18)
        self.assertEqual(model.channel_projection.out_channels, 18)

    def test_run_checkpoint_is_restricted_load_compatible(self):
        model, _ = build_sleep_biot(None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_model.pt"
            save_run_checkpoint(path, model, 3, 0.75)
            checkpoint = load_run_checkpoint(path)
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["dev_macro_f1"], 0.75)


if __name__ == "__main__":
    unittest.main()
