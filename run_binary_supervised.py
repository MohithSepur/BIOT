import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
except ModuleNotFoundError:
    pl = None
    TensorBoardLogger = None
    EarlyStopping = None

try:
    from pyhealth.metrics import binary_metrics_fn
except ModuleNotFoundError:
    binary_metrics_fn = None

try:
    from model import (
        SPaRCNet,
        ContraWR,
        CNNTransformer,
        FFCL,
        STTransformer,
        BIOTClassifier,
    )
    _MODEL_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    SPaRCNet = ContraWR = CNNTransformer = FFCL = STTransformer = BIOTClassifier = None
    _MODEL_IMPORT_ERROR = error
from utils import TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE


_LightningModule = pl.LightningModule if pl is not None else nn.Module


class LitModel_finetune(_LightningModule):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_outputs = []
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = BCE(prob, y)  # focal_loss(prob, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.validation_outputs.append((step_result, step_gt))

    def on_validation_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        for out in self.validation_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        self.validation_outputs.clear()

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_outputs.append((step_result, step_gt))

    def on_test_epoch_end(self):
        result = np.array([])
        gt = np.array([])
        for out in self.test_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        self.test_outputs.clear()
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def prepare_TUAB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = args.input_dir
    if not root:
        raise ValueError("--input_dir is required for TUAB")

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    # train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_CHB_MIT_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = args.input_dir
    if not root:
        raise ValueError("--input_dir is required for CHB-MIT")

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = args.input_dir
    if not root:
        raise ValueError("--input_dir is required for PTB")

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


def supervised(args):
    if args.dataset in {"TUSZ", "CHB-MIT"}:
        from seizure_training import run_seizure_supervised

        return run_seizure_supervised(args)

    if pl is None:
        raise RuntimeError("Legacy supervised tasks require pytorch_lightning")
    if binary_metrics_fn is None:
        raise RuntimeError("Legacy supervised tasks require pyhealth")
    if _MODEL_IMPORT_ERROR is not None:
        raise RuntimeError("A legacy model dependency is unavailable") from _MODEL_IMPORT_ERROR
    # get data loaders
    if args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)

    else:
        raise NotImplementedError

    # define the model
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sampling_rate * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sampling_rate * args.sample_length),
            shrink_steps=20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            # set the n_channels according to the pretrained model if necessary
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
        )
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            print(f"load pretrain model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError
    lightning_model = LitModel_finetune(args, model)

    # logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        strategy="auto",
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--num_workers", type=int,
                        default=32, help="number of workers")
    parser.add_argument("--dataset", type=str, default="TUAB", help="dataset")
    parser.add_argument(
        "--model", type=str, default="SPaRCNet", help="which supervised model to use"
    )
    parser.add_argument(
        "--in_channels", type=int, default=16, help="number of input channels"
    )
    parser.add_argument(
        "--sample_length", type=int, default=10, help="length (s) of sample"
    )
    parser.add_argument(
        "--n_classes", type=int, default=1, help="number of output classes"
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=200, help="sampling rate (r)"
    )
    parser.add_argument("--token_size", type=int,
                        default=200, help="token size (t)")
    parser.add_argument(
        "--hop_length", type=int, default=100, help="token hop length (t - p)"
    )
    parser.add_argument(
        "--pretrain_model_path", type=str, default="", help="pretrained model path"
    )
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--marker_dir", type=str, default="./data/file_markers_detection")
    parser.add_argument("--result_dir", type=str, default="./seizure-results")
    parser.add_argument(
        "--use_fft", action=argparse.BooleanOptionalAction, default=False,
        help="must remain false for BIOT's raw-input STFT",
    )
    parser.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=True,
        help="apply explicitly supplied raw TUSZ training scalars",
    )
    parser.add_argument("--scaler_mean_path", type=str, default=None)
    parser.add_argument("--scaler_std_path", type=str, default=None)
    parser.add_argument("--data_augment", action="store_true")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--chb_channel_order",
        choices=("unknown", "biot_process2"),
        default="unknown",
        help="only assert biot_process2 when PKLs were made by this repo's process2.py",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    args = parser.parse_args()
    print(args)

    supervised(args)
