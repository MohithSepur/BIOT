#!/usr/bin/env python3
"""Fine-tune and evaluate the full BIOT model on Sleep-EDF Expanded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sleep_data import NUM_CLASSES, SLEEP_STAGE_NAMES, SleepEDFDataset, class_counts, load_sleep_manifest
from sleep_model import build_sleep_biot


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "pretrained-models" / "EEG-PREST-16-channels.ckpt"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_class_weights(counts: dict[int, int], device: torch.device) -> torch.Tensor:
    if any(counts[label] == 0 for label in range(NUM_CLASSES)):
        raise ValueError(f"Cannot balance a training split missing a class: {counts}")
    total = sum(counts.values())
    return torch.tensor(
        [total / (NUM_CLASSES * counts[label]) for label in range(NUM_CLASSES)],
        dtype=torch.float32,
        device=device,
    )


def finite_gradients(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    use_amp,
    max_grad_norm,
):
    model.train()
    total_loss = 0.0
    examples = 0
    skipped = 0
    progress = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for inputs, labels, _names in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = model(inputs)
            loss = criterion(logits, labels)
        if not bool(torch.isfinite(loss)):
            skipped += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not finite_gradients(model):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped += 1
            continue
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        examples += batch_size
        progress.set_postfix(
            loss=float(loss.detach()),
            lr=optimizer.param_groups[0]["lr"],
            skipped=skipped,
        )
    if examples == 0:
        raise RuntimeError("No finite training batch completed")
    return total_loss / examples, skipped


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    labels_out: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    file_names: list[str] = []
    total_loss = 0.0
    examples = 0
    for inputs, labels, names in tqdm(loader, desc="Evaluate", dynamic_ncols=True):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(inputs)
            loss = criterion(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite evaluation output for {list(names)}")
        probability = torch.softmax(logits.float(), dim=-1)
        batch_size = labels.shape[0]
        total_loss += float(loss) * batch_size
        examples += batch_size
        labels_out.append(labels.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        file_names.extend(names)

    y_true = np.concatenate(labels_out)
    y_prob = np.concatenate(probabilities)
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "loss": total_loss / examples,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES)),
    }
    return metrics, y_true, y_prob, y_pred, np.asarray(file_names, dtype=object)


def save_results(path, metrics, labels, probabilities, predictions, file_names):
    np.savez_compressed(
        path,
        labels=labels,
        probabilities=probabilities,
        predictions=predictions,
        file_names=file_names,
        class_names=np.asarray(SLEEP_STAGE_NAMES),
        **metrics,
    )


def metric_log(metrics) -> str:
    printable = {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }
    return json.dumps(printable, sort_keys=True)


def save_run_checkpoint(path: Path, model: nn.Module, epoch: int, dev_macro_f1: float):
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "dev_macro_f1": float(dev_macro_f1),
        },
        path,
    )


def load_run_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        try:
            from numpy._core.multiarray import scalar as numpy_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as numpy_scalar
        safe_types = [numpy_scalar, np.dtype, type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location="cpu", weights_only=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BIOT on five-class Sleep-EDF")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--max-patience", type=int, default=7)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.prefetch_factor <= 0 or not 0 < args.warmup_fraction < 1:
        raise ValueError("prefetch-factor must be positive and warmup-fraction in (0,1)")
    if args.max_grad_norm < 0:
        raise ValueError("max-grad-norm must be non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_sleep_manifest(data_dir)
    datasets = {name: SleepEDFDataset(epochs) for name, epochs in split.items()}
    worker_options = (
        {"persistent_workers": True, "prefetch_factor": args.prefetch_factor}
        if args.num_workers > 0
        else {}
    )
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            **worker_options,
        )
        for name, dataset in datasets.items()
    }
    counts = {name: class_counts(epochs) for name, epochs in split.items()}
    print(f"Device: {device}; AMP: {use_amp}")
    print(f"Epochs: { {name: len(epochs) for name, epochs in split.items()} }")
    print(f"Class counts {list(SLEEP_STAGE_NAMES)}: {counts}")

    pretrained_path = args.checkpoint if args.pretrained else None
    model, model_report = build_sleep_biot(pretrained_path, args.freeze_encoder)
    model.to(device)
    weights = balanced_class_weights(counts["train"], device) if args.balanced_loss else None
    train_criterion = nn.CrossEntropyLoss(weight=weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(loaders["train"]),
        epochs=args.epochs,
        pct_start=args.warmup_fraction,
    )
    scaler = make_grad_scaler(use_amp)

    config = vars(args).copy()
    config.update(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        checkpoint=str(pretrained_path) if pretrained_path else None,
        model_report=model_report,
        class_counts=counts,
        stage_names=list(SLEEP_STAGE_NAMES),
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    checkpoint_path = output_dir / "best_model.pt"
    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    started = time.time()
    if not args.evaluate_only:
        for epoch in range(1, args.epochs + 1):
            train_loss, skipped = train_epoch(
                model,
                loaders["train"],
                train_criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                use_amp,
                args.max_grad_norm,
            )
            dev_metrics, *_ = evaluate(model, loaders["dev"], eval_criterion, device, use_amp)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, skipped={skipped}, "
                f"dev={metric_log(dev_metrics)}"
            )
            dev_f1 = float(dev_metrics["macro_f1"])
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_epoch = epoch
                patience = 0
                save_run_checkpoint(checkpoint_path, model, epoch, dev_f1)
            else:
                patience += 1
                if patience >= args.max_patience:
                    print(f"Early stopping after {patience} epochs without improvement")
                    break
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint to evaluate: {checkpoint_path}")

    best = load_run_checkpoint(checkpoint_path)
    model.load_state_dict(best["model"])
    model.to(device)
    best_epoch = int(best["epoch"])
    best_f1 = float(best["dev_macro_f1"])
    for split_name in ("dev", "test"):
        result = evaluate(model, loaders[split_name], eval_criterion, device, use_amp)
        metrics, labels, probabilities, predictions, names = result
        save_results(
            output_dir / f"{split_name}_results.npz",
            metrics,
            labels,
            probabilities,
            predictions,
            names,
        )
        print(f"{split_name.upper()}: {metric_log(metrics)}")
    print(
        f"Finished in {time.time() - started:.1f}s; "
        f"best epoch={best_epoch}, dev macro-F1={best_f1:.6f}"
    )


if __name__ == "__main__":
    main()
