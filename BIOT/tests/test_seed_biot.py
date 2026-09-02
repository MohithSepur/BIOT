from pathlib import Path
import json
import pickle
import tempfile
import unittest

import numpy as np
from scipy.io import savemat
import torch
import seed_data

from seed_data import (
    BIOT_PREST16_CHANNELS,
    SEED_CHANNELS,
    SeedBIOTDataset,
    SeedLMDBDataset,
    discover_seed_lmdb_windows,
    discover_seed_windows,
    prepare_seed_cache,
    split_seed_windows,
    to_biot_prest16,
)
from seed_model import build_seed_biot
from run_seed_biot import load_run_checkpoint, save_run_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "pretrained-models" / "EEG-PREST-16-channels.ckpt"


class SeedBIOTTest(unittest.TestCase):
    def _make_seed(self, root: Path) -> None:
        savemat(root / "label.mat", {"label": np.asarray([[-1, 0, 1]])})
        for subject in (1, 2, 3):
            trials = {}
            for trial in (1, 2, 3):
                signal = np.arange(62 * 400, dtype=np.float64).reshape(62, 400)
                signal = signal + subject * 1000 + trial
                trials[f"subject{subject}_eeg{trial}"] = signal
            savemat(root / f"{subject}_sessionA.mat", trials)

    def test_index_subject_split_and_three_class_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_seed(root)
            windows = discover_seed_windows(
                root,
                source_sampling_rate=200,
                window_seconds=2,
                stride_seconds=2,
            )
            split = split_seed_windows(windows, test_subject=3, dev_subject=2)
            self.assertEqual({window.subject for window in split["train"]}, {1})
            self.assertEqual({window.subject for window in split["dev"]}, {2})
            self.assertEqual({window.subject for window in split["test"]}, {3})
            self.assertEqual({window.label for window in windows}, {0, 1, 2})

    def test_bipolar_derivation_and_dataset_contract(self) -> None:
        monopolar = np.zeros((len(SEED_CHANNELS), 20), dtype=np.float32)
        monopolar[SEED_CHANNELS.index("FP1")] = 3
        monopolar[SEED_CHANNELS.index("F7")] = 1
        bipolar = to_biot_prest16(monopolar)
        self.assertEqual(tuple(bipolar.shape), (16, 20))
        np.testing.assert_array_equal(bipolar[0], np.full(20, 2, dtype=np.float32))
        self.assertEqual(len(BIOT_PREST16_CHANNELS), 16)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_seed(root)
            windows = discover_seed_windows(
                root,
                source_sampling_rate=200,
                window_seconds=2,
                stride_seconds=2,
            )
            x, y, name = SeedBIOTDataset(windows[:1], normalize=False)[0]
            self.assertEqual(tuple(x.shape), (16, 400))
            self.assertEqual(y.dtype, torch.long)
            self.assertTrue(name.startswith("sub01_"))
            self.assertTrue(torch.isfinite(x).all())

    def test_actual_checkpoint_forward_backward_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_seed(root)
            windows = discover_seed_windows(
                root,
                source_sampling_rate=200,
                window_seconds=2,
                stride_seconds=2,
            )
            dataset = SeedBIOTDataset([windows[0], windows[-1]])
            examples = [dataset[index] for index in range(2)]
            x = torch.stack([example[0] for example in examples])
            y = torch.stack([example[1] for example in examples])

            model, report = build_seed_biot(CHECKPOINT)
            self.assertTrue(report["checkpoint_loaded"])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (2, 3))
            loss = torch.nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            self.assertTrue(
                all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
            optimizer.step()

    def test_trial_cache_preserves_output_and_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_seed(root)
            windows = discover_seed_windows(
                root,
                source_sampling_rate=200,
                window_seconds=2,
                stride_seconds=2,
            )
            cache_dir = root / "cache"
            first = prepare_seed_cache(windows, cache_dir)
            second = prepare_seed_cache(windows, cache_dir)
            self.assertEqual(first, {"trials": 9, "created": 9, "reused": 0})
            self.assertEqual(second, {"trials": 9, "created": 0, "reused": 9})

            uncached = SeedBIOTDataset(windows[:1], normalize=True)[0]
            cached = SeedBIOTDataset(
                windows[:1], normalize=True, cache_dir=cache_dir
            )[0]
            torch.testing.assert_close(cached[0], uncached[0], rtol=0, atol=0)
            self.assertEqual(cached[1].item(), uncached[1].item())
            self.assertEqual(cached[2], uncached[2])

            resampled_cache = root / "resampled-cache"
            prepare_seed_cache(
                windows,
                resampled_cache,
                source_sampling_rate=200,
            )
            uncached_resampled = SeedBIOTDataset(
                windows[:1],
                source_sampling_rate=200,
                target_sampling_rate=100,
                normalize=True,
            )[0]
            cached_resampled = SeedBIOTDataset(
                windows[:1],
                source_sampling_rate=200,
                target_sampling_rate=100,
                normalize=True,
                cache_dir=resampled_cache,
            )[0]
            torch.testing.assert_close(
                cached_resampled[0], uncached_resampled[0], rtol=0, atol=0
            )

    def test_lmdb_split_grouping_and_dataset_contract(self) -> None:
        class FakeTransaction:
            def __init__(self, records):
                self.records = records

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def get(self, key):
                return self.records.get(key)

        class FakeEnvironment:
            def __init__(self, records):
                self.records = records

            def begin(self, write=False):
                self.assert_readonly = not write
                return FakeTransaction(self.records)

            def close(self):
                pass

        class FakeLMDB:
            def __init__(self, records):
                self.records = records

            def open(self, *_args, **_kwargs):
                return FakeEnvironment(self.records)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "data.mdb").touch()
            split_keys = {"train": [], "val": [], "test": []}
            records = {}
            for split_name, subject, label, segment_indices in (
                ("train", 1, 0, [0, 1, *range(5, 15)]),
                ("val", 2, 1, range(10)),
                ("test", 3, 2, range(10)),
            ):
                for segment in segment_indices:
                    key = f"{subject}_session.mat-1-{segment}"
                    split_keys[split_name].append(key)
                    sample = np.zeros((62, 1, 200), dtype=np.float64)
                    sample[SEED_CHANNELS.index("FP1")] = 3 + segment
                    sample[SEED_CHANNELS.index("F7")] = 1
                    records[key.encode()] = pickle.dumps(
                        {"sample": sample, "label": label}
                    )
            records[b"__keys__"] = pickle.dumps(split_keys)
            (root / "subject_split.json").write_text(
                json.dumps({"train": ["1"], "val": ["2"], "test": ["3"]}),
                encoding="utf-8",
            )

            original_lmdb = seed_data.lmdb
            seed_data.lmdb = FakeLMDB(records)
            try:
                split = discover_seed_lmdb_windows(root)
                self.assertEqual(
                    {name: len(windows) for name, windows in split.items()},
                    {"train": 1, "dev": 1, "test": 1},
                )
                self.assertEqual(
                    {name: {window.subject for window in windows}
                     for name, windows in split.items()},
                    {"train": {1}, "dev": {2}, "test": {3}},
                )
                dataset = SeedLMDBDataset(root, split["train"], normalize=False)
                x, y, writeout_fn = dataset[0]
                self.assertEqual(tuple(x.shape), (16, 2000))
                self.assertEqual(y.item(), 0)
                expected = torch.cat(
                    [torch.full((200,), float(2 + segment)) for segment in range(5, 15)]
                )
                torch.testing.assert_close(x[0], expected)
                self.assertIn("segments000005-000015", writeout_fn)

                model, report = build_seed_biot(CHECKPOINT)
                self.assertTrue(report["checkpoint_loaded"])
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                logits = model(x.unsqueeze(0))
                self.assertEqual(tuple(logits.shape), (1, 3))
                loss = torch.nn.CrossEntropyLoss()(logits, y.unsqueeze(0))
                loss.backward()
                self.assertTrue(
                    all(
                        torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    )
                )
                optimizer.step()
            finally:
                seed_data.lmdb = original_lmdb

    def test_checkpoint_metadata_and_legacy_numpy_scalar_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = torch.nn.Linear(2, 1)
            current_path = root / "current.pt"
            save_run_checkpoint(current_path, model, 3, np.float64(0.625))
            current = load_run_checkpoint(current_path)
            self.assertIs(type(current["epoch"]), int)
            self.assertIs(type(current["dev_macro_f1"]), float)

            legacy_path = root / "legacy.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": 12,
                    "dev_macro_f1": np.float64(0.6849694153919729),
                },
                legacy_path,
            )
            legacy = load_run_checkpoint(legacy_path)
            self.assertEqual(legacy["epoch"], 12)
            self.assertAlmostEqual(float(legacy["dev_macro_f1"]), 0.6849694153919729)


if __name__ == "__main__":
    unittest.main()
