from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat
import torch

from seed_data import (
    BIOT_PREST16_CHANNELS,
    SEED_CHANNELS,
    SeedBIOTDataset,
    discover_seed_windows,
    prepare_seed_cache,
    split_seed_windows,
    to_biot_prest16,
)
from seed_model import build_seed_biot


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


if __name__ == "__main__":
    unittest.main()
