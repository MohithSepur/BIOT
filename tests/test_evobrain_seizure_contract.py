import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model.biot import BIOTClassifier
from seizure_data import BIOT_CHB16_CHANNELS, CHBMITPklDataset, TUSZDataset
from seizure_model import (
    BIOTSeizureModel,
    BIOTSeizureOutput,
    _may_reuse_checkpoint_channels,
    build_biot_seizure_model,
    checkpoint_channel_count,
)
from seizure_training import (
    evaluate_and_save,
    evaluation_criterion,
    smoothed_pos_weight,
    train_one_batch,
    training_criterion,
)


def synthetic_batch(batch_size, channels, pkl_contract=False):
    # Shapes and dtypes are the accepted Step A raw contracts.
    x = torch.randn(batch_size, 10, channels, 200, dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)[:batch_size]
    seq_len = torch.full((batch_size, 1), 10, dtype=torch.int64)
    supports = (
        torch.empty(batch_size, 0, dtype=torch.float32)
        if pkl_contract
        else torch.zeros(batch_size, 10, 2, channels, channels, dtype=torch.float32)
    )
    adj_mat = torch.eye(channels, dtype=torch.float32).repeat(batch_size, 10, 1, 1)
    names = tuple(f"sample-{index}" for index in range(batch_size))
    return x, y, seq_len, supports, adj_mat, names


class _DatasetWithPosWeight:
    pos_weight = 3.0


class _StaticDataset(Dataset):
    def __init__(self):
        self.samples = []
        for index, label in enumerate((0.0, 1.0, 0.0, 1.0)):
            sample = synthetic_batch(2, 16, pkl_contract=True)
            values = [
                value[index % 2] if torch.is_tensor(value) else value[index % 2]
                for value in sample
            ]
            values[1] = torch.tensor([label], dtype=torch.float32)
            values[5] = f"eval-{index}"
            self.samples.append(tuple(values))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _StaticModel(torch.nn.Module):
    def forward_contract(self, batch):
        x, y, seq_len, supports, adj_mat, names = batch
        logits = torch.where(y.reshape(-1) > 0, torch.tensor(2.0), torch.tensor(-2.0)).to(x)
        return BIOTSeizureOutput(logits, y, seq_len, supports, adj_mat, tuple(names))


class _ChannelDataset:
    def __init__(self, channels, names=None, verified=False):
        self.num_nodes = channels
        self.channel_names = names
        self.channel_order_verified = verified


class EvoBrainContractTest(unittest.TestCase):
    def test_actual_biot_forward_backward_optimizer_step(self):
        torch.manual_seed(123)
        backbone = BIOTClassifier(
            n_classes=1,
            n_channels=19,
            n_fft=200,
            hop_length=100,
        )
        model = BIOTSeizureModel(backbone)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = training_criterion(_DatasetWithPosWeight(), torch.device("cpu"))
        batch = synthetic_batch(2, 19, pkl_contract=False)

        loss, stepped, names = train_one_batch(
            model,
            batch,
            optimizer,
            criterion,
            torch.device("cpu"),
            max_grad_norm=5.0,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(stepped)
        self.assertEqual(names, ("sample-0", "sample-1"))
        self.assertEqual(model.adapter(batch).x.shape, (2, 19, 2000))
        chb_output = model.forward_contract(synthetic_batch(2, 16, pkl_contract=True))
        self.assertEqual(chb_output.logits.shape, (2,))
        self.assertEqual(chb_output.supports.shape, (2, 0))

    def test_chb_pkl_contract_dynamic_weight_and_declared_order(self):
        with tempfile.TemporaryDirectory() as directory:
            train_dir = Path(directory) / "train"
            train_dir.mkdir()
            for index, label in enumerate((1, 0, 0, 0)):
                with (train_dir / f"sample-{index}.pkl").open("wb") as handle:
                    pickle.dump(
                        {"X": np.random.randn(16, 2560).astype(np.float32), "y": label},
                        handle,
                    )
            dataset = CHBMITPklDataset(
                directory,
                "train",
                use_fft=False,
                declared_channel_order="biot_process2",
            )
            self.assertEqual(dataset.pos_weight, 3.0)
            self.assertEqual(smoothed_pos_weight(dataset.pos_weight), np.sqrt(3.0))
            self.assertEqual(dataset.channel_names, BIOT_CHB16_CHANNELS)
            self.assertTrue(dataset.channel_order_verified)
            x, y, seq_len, supports, adj_mat, name = dataset[0]
            self.assertEqual(x.shape, (10, 16, 200))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.shape, (1,))
            self.assertEqual(seq_len.dtype, torch.int64)
            self.assertEqual(supports.shape, (0,))
            self.assertEqual(adj_mat.shape, (10, 16, 16))
            self.assertEqual(name, "sample-0")

    def test_checkpoint_channel_counts_and_alignment_policy(self):
        checkpoint_dir = Path(__file__).parents[1] / "pretrained-models"
        self.assertEqual(
            checkpoint_channel_count(checkpoint_dir / "EEG-PREST-16-channels.ckpt"), 16
        )
        self.assertEqual(
            checkpoint_channel_count(checkpoint_dir / "EEG-SHHS+PREST-18-channels.ckpt"), 18
        )
        verified_chb = _ChannelDataset(16, BIOT_CHB16_CHANNELS, True)
        unknown_chb = _ChannelDataset(16, None, False)
        tusz = _ChannelDataset(19, None, True)
        self.assertTrue(_may_reuse_checkpoint_channels(verified_chb, 16))
        self.assertTrue(_may_reuse_checkpoint_channels(verified_chb, 18))
        self.assertFalse(_may_reuse_checkpoint_channels(unknown_chb, 16))
        self.assertFalse(_may_reuse_checkpoint_channels(tusz, 18))

        args = SimpleNamespace(
            pretrain_model_path=str(
                checkpoint_dir / "EEG-PREST-16-channels.ckpt"
            ),
            token_size=200,
            hop_length=100,
        )
        aligned_model, aligned_policy = build_biot_seizure_model(args, verified_chb)
        self.assertEqual(aligned_policy, "reused_verified_first_16_channel_rows")
        self.assertEqual(aligned_model.backbone.biot.channel_tokens.num_embeddings, 16)
        tusz_model, tusz_policy = build_biot_seizure_model(args, tusz)
        self.assertEqual(tusz_policy, "reinitialized_19_channel_rows")
        self.assertEqual(tusz_model.backbone.biot.channel_tokens.num_embeddings, 19)

    def test_tusz_index_and_fft_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory) / "dev"
            raw_dir.mkdir()
            (raw_dir / "record.edf").touch()
            (raw_dir / "record.tse").write_text(
                "0 5 bckg\n5 8 fnsz\n8 25 bckg\n", encoding="utf-8"
            )
            dataset = TUSZDataset(
                directory, "dev", standardize=False, use_fft=False
            )
            indexed = sorted((clip, label) for _, clip, label, _ in dataset.entries)
            self.assertEqual(indexed, [(0, 1), (1, 0)])
            with self.assertRaisesRegex(ValueError, "use_fft=False"):
                TUSZDataset(directory, "dev", standardize=False, use_fft=True)

    def test_eval_is_unweighted_and_dev_threshold_is_reused(self):
        self.assertIsNone(evaluation_criterion(torch.device("cpu")).pos_weight)
        loader = DataLoader(_StaticDataset(), batch_size=2, shuffle=False)
        model = _StaticModel()
        with tempfile.TemporaryDirectory() as directory:
            dev = evaluate_and_save(model, loader, torch.device("cpu"), directory, "dev")
            test = evaluate_and_save(
                model,
                loader,
                torch.device("cpu"),
                directory,
                "test",
                threshold=dev["threshold"],
            )
            self.assertEqual(test["threshold"], dev["threshold"])
            for split in ("dev", "test"):
                result = np.load(Path(directory) / f"{split}_results.npz")
                self.assertEqual(result["file_names"].tolist(), [f"eval-{i}" for i in range(4)])
                for key in (
                    "labels", "probabilities", "predictions", "accuracy",
                    "balanced_accuracy", "f1", "recall", "precision",
                    "specificity", "auroc", "pr_auc",
                ):
                    self.assertIn(key, result.files)


if __name__ == "__main__":
    unittest.main()
