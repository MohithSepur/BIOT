#!/usr/bin/env python3
"""Train and evaluate BIOT on three-class SEED emotion recognition."""

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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from seed_data import (
    SeedBIOTDataset,
    SeedLMDBDataset,
    class_counts,
    discover_seed_windows,
    discover_seed_lmdb_windows,
    is_seed_lmdb,
    prepare_seed_cache,
    split_seed_windows,
)
from seed_model import build_seed_biot


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
    if any(counts[index] == 0 for index in range(3)):
        raise ValueError(f"Cannot balance a split missing a class: {counts}")
    total = sum(counts.values())
    return torch.tensor(
        [total / (3.0 * counts[index]) for index in range(3)],
        dtype=torch.float32,
        device=device,
    )


def finite_gradients(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def train_epoch(model, loader, criterion, optimizer, device, epoch, max_grad_norm=0.0):
    model.train()
    total_loss = 0.0
    examples = 0
    skipped = 0
    progress = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for x, y, _names in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not bool(torch.isfinite(loss)):
            skipped += 1
            continue
        loss.backward()
        if not finite_gradients(model):
            optimizer.zero_grad(set_to_none=True)
            skipped += 1
            continue
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        batch_size = y.shape[0]
        total_loss += float(loss.detach()) * batch_size
        examples += batch_size
        progress.set_postfix(loss=float(loss.detach()), skipped=skipped)
    if examples == 0:
        raise RuntimeError("No finite training batch completed")
    return total_loss / examples, skipped


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    file_names: list[str] = []
    total_loss = 0.0
    examples = 0
    for x, y, names in tqdm(loader, desc="Evaluate", dynamic_ncols=True):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite evaluation output for {list(names)}")
        probability = torch.softmax(logits, dim=-1)
        batch_size = y.shape[0]
        total_loss += float(loss) * batch_size
        examples += batch_size
        labels.append(y.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        file_names.extend(names)

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "loss": total_loss / examples,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
    }
    return metrics, y_true, y_prob, y_pred, np.asarray(file_names, dtype=object)


def save_results(path, metrics, labels, probabilities, predictions, file_names):
    np.savez_compressed(
        path,
        labels=labels,
        probabilities=probabilities,
        predictions=predictions,
        file_names=file_names,
        loss=metrics["loss"],
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        macro_precision=metrics["macro_precision"],
        macro_recall=metrics["macro_recall"],
        confusion_matrix=metrics["confusion_matrix"],
    )


def metric_log(metrics):
    printable = {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }
    return json.dumps(printable, sort_keys=True)


def save_run_checkpoint(path: Path, model: nn.Module, epoch: int, dev_macro_f1: float):
    """Save only tensors and built-in scalar metadata for restricted loading."""
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "dev_macro_f1": float(dev_macro_f1),
        },
        path,
    )


def load_run_checkpoint(path: Path):
    """Load current checkpoints and safely recover our legacy NumPy metadata."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Older versions of this runner saved sklearn's np.float64 F1 value.
        # Allow only NumPy scalar/dtype constructors, never arbitrary globals.
        try:
            from numpy._core.multiarray import scalar as numpy_scalar
        except ImportError:  # NumPy 1.x
            from numpy.core.multiarray import scalar as numpy_scalar
        safe_types = [numpy_scalar, np.dtype, type(np.dtype(np.float64))]
        with torch.serialization.safe_globals(safe_types):
            return torch.load(path, map_location="cpu", weights_only=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BIOT on three-class SEED")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--data-format",
        choices=("auto", "lmdb", "mat"),
        default="auto",
        help="Input storage; auto selects LMDB when DATA_DIR/data.mdb exists",
    )
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--test-subject", type=int, default=None,
        help="MAT input only; LMDB uses its stored subject split",
    )
    parser.add_argument(
        "--dev-subject", type=int, default=None,
        help="MAT input only; LMDB uses its stored subject split",
    )
    parser.add_argument("--source-sampling-rate", type=int, default=200)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Shared trial cache (default: DATA_DIR/.biot_seed_cache_v1)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Read compressed MAT trials directly (substantially slower)",
    )
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Regenerate all cached trials"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches queued per DataLoader worker",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Optional gradient clipping; 0 disables clipping (default)",
    )
    parser.add_argument("--max-patience", type=int, default=7)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and evaluate OUTPUT_DIR/best_model.pt",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be positive")
    if args.max_grad_norm < 0:
        raise ValueError("max-grad-norm must be non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.expanduser().resolve()
    data_format = args.data_format
    if data_format == "auto":
        data_format = "lmdb" if is_seed_lmdb(data_dir) else "mat"
    cache_dir = None
    cache_report = None
    if data_format == "lmdb":
        if args.source_sampling_rate != 200:
            raise ValueError("This SEED LMDB was created at 200 Hz")
        if args.label_file is not None or args.cache_dir is not None or args.rebuild_cache:
            raise ValueError(
                "--label-file/--cache-dir/--rebuild-cache apply only to MAT input; "
                "LMDB is loaded directly without another cache"
            )
        split = discover_seed_lmdb_windows(
            data_dir,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        datasets = {
            name: SeedLMDBDataset(data_dir, values)
            for name, values in split.items()
        }
    else:
        if args.test_subject is None or args.dev_subject is None:
            raise ValueError("MAT input requires --test-subject and --dev-subject")
        windows = discover_seed_windows(
            data_dir,
            args.label_file,
            source_sampling_rate=args.source_sampling_rate,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        if not args.no_cache:
            cache_dir = (
                args.cache_dir.expanduser().resolve()
                if args.cache_dir is not None
                else data_dir / ".biot_seed_cache_v1"
            )
            cache_report = prepare_seed_cache(
                windows,
                cache_dir,
                source_sampling_rate=args.source_sampling_rate,
                rebuild=args.rebuild_cache,
            )
            print(f"Trial cache: {cache_dir} ({cache_report})")
        split = split_seed_windows(windows, args.test_subject, args.dev_subject)
        datasets = {
            name: SeedBIOTDataset(
                values,
                source_sampling_rate=args.source_sampling_rate,
                target_sampling_rate=200,
                cache_dir=cache_dir,
            )
            for name, values in split.items()
        }
    loader_worker_options = (
        {
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        }
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
            **loader_worker_options,
        )
        for name, dataset in datasets.items()
    }
    counts = {name: class_counts(values) for name, values in split.items()}
    print(f"Device: {device}")
    print(f"Data format: {data_format}")
    print(f"Windows: { {name: len(values) for name, values in split.items()} }")
    print(f"Class counts [negative, neutral, positive]: {counts}")

    checkpoint = args.checkpoint if args.pretrained else None
    model, model_report = build_seed_biot(checkpoint, args.freeze_encoder)
    model.to(device)
    weights = balanced_class_weights(counts["train"], device) if args.balanced_loss else None
    train_criterion = nn.CrossEntropyLoss(weight=weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = vars(args).copy()
    run_config.update(
        data_dir=str(data_dir),
        data_format=data_format,
        label_file=str(args.label_file) if args.label_file else None,
        output_dir=str(output_dir),
        checkpoint=str(checkpoint) if checkpoint else None,
        cache_dir=str(cache_dir) if cache_dir else None,
        cache_report=cache_report,
        model_report=model_report,
        class_counts=counts,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    best_f1 = -1.0
    best_epoch = 0
    patience = 0
    start_time = time.time()
    checkpoint_path = output_dir / "best_model.pt"
    if not args.evaluate_only:
        for epoch in range(1, args.epochs + 1):
            train_loss, skipped = train_epoch(
                model,
                loaders["train"],
                train_criterion,
                optimizer,
                device,
                epoch,
                args.max_grad_norm,
            )
            dev_metrics, *_ = evaluate(model, loaders["dev"], eval_criterion, device)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, skipped={skipped}, "
                f"dev={metric_log(dev_metrics)}"
            )
            dev_macro_f1 = float(dev_metrics["macro_f1"])
            if dev_macro_f1 > best_f1:
                best_f1 = dev_macro_f1
                best_epoch = epoch
                patience = 0
                save_run_checkpoint(checkpoint_path, model, epoch, best_f1)
            else:
                patience += 1
                if patience >= args.max_patience:
                    break
    elif not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint to evaluate: {checkpoint_path}")

    best = load_run_checkpoint(checkpoint_path)
    best_epoch = int(best["epoch"])
    best_f1 = float(best["dev_macro_f1"])
    model.load_state_dict(best["model"])
    model.to(device)
    for split_name in ("dev", "test"):
        result = evaluate(model, loaders[split_name], eval_criterion, device)
        metrics, labels, probabilities, predictions, file_names = result
        save_results(
            output_dir / f"{split_name}_results.npz",
            metrics,
            labels,
            probabilities,
            predictions,
            file_names,
        )
        print(f"{split_name.upper()}: {metric_log(metrics)}")
    print(
        f"Finished in {time.time() - start_time:.1f}s; "
        f"best epoch={best_epoch}, dev macro-F1={best_f1:.6f}"
    )


if __name__ == "__main__":
    main()
