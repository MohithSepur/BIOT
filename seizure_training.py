"""Loss, stability, and traceable evaluation for BIOT seizure detection."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def smoothed_pos_weight(raw_pos_weight: float) -> float:
    """Mirror EvoBrain/main.py:283-288 without replacing data statistics."""
    return float(min(sqrt(raw_pos_weight) if raw_pos_weight > 1.0 else raw_pos_weight, 20.0))


def training_criterion(train_dataset, device: torch.device) -> nn.Module:
    raw_pos_weight = getattr(train_dataset, "pos_weight", None)
    if raw_pos_weight is None:
        return nn.BCEWithLogitsLoss().to(device)
    weight = torch.tensor(
        [smoothed_pos_weight(float(raw_pos_weight))], dtype=torch.float32, device=device
    )
    return nn.BCEWithLogitsLoss(pos_weight=weight).to(device)


def evaluation_criterion(device: torch.device) -> nn.Module:
    """Evaluation is intentionally unweighted (EvoBrain/main.py:490-494)."""
    return nn.BCEWithLogitsLoss().to(device)


def move_contract_batch(batch, device: torch.device):
    if not isinstance(batch, (tuple, list)) or len(batch) != 6:
        raise ValueError("Expected a six-field EvoBrain seizure batch")
    x, y, seq_len, supports, adj_mat, writeout_fn = batch
    return (
        x.to(device),
        y.to(device),
        seq_len.to(device),
        supports.to(device),
        adj_mat.to(device),
        tuple(str(name) for name in writeout_fn),
    )


def train_one_batch(
    model,
    batch,
    optimizer,
    criterion,
    device,
    max_grad_norm,
    scaler=None,
    use_amp=False,
):
    optimizer.zero_grad()
    moved = move_contract_batch(batch, device)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.amp.autocast(autocast_device, enabled=use_amp):
        output = model.forward_contract(moved)
        target = output.y.reshape(-1).float()
        loss = criterion(output.logits.float(), target)
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return None, False, output.writeout_fn
    if use_amp:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    finite_gradients = all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if finite_gradients:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if use_amp:
            scaler.step(optimizer)
        else:
            optimizer.step()
    else:
        optimizer.zero_grad()
    if use_amp:
        scaler.update()
    return float(loss.detach().cpu()), finite_gradients, output.writeout_fn


def max_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    unique_probabilities = np.unique(probabilities)
    if unique_probabilities.size == 0:
        return 0.5  # EvoBrain/main.py:486 default evaluation threshold.
    candidates = np.nextafter(unique_probabilities, -np.inf)
    scores = [
        f1_score(labels, probabilities > candidate, zero_division=0)
        for candidate in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (probabilities > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    both_classes = np.unique(labels).size == 2
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "auroc": float(roc_auc_score(labels, probabilities)) if both_classes else float("nan"),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if np.any(labels == 1)
            else float("nan")
        ),
        "threshold": float(threshold),
    }
    return metrics, predictions


@torch.no_grad()
def evaluate_and_save(model, data_loader, device, output_dir, split, threshold=None):
    model.eval()
    criterion = evaluation_criterion(device)
    labels, probabilities, file_names = [], [], []
    weighted_loss = 0.0
    sample_count = 0
    for batch in data_loader:
        output = model.forward_contract(move_contract_batch(batch, device))
        target = output.y.reshape(-1).float()
        loss = criterion(output.logits.float(), target)
        probability = torch.nan_to_num(torch.sigmoid(output.logits.float()), nan=0.0)
        count = target.numel()
        weighted_loss += float(loss.cpu()) * count
        sample_count += count
        labels.extend(target.long().cpu().numpy().tolist())
        probabilities.extend(probability.cpu().numpy().tolist())
        file_names.extend(output.writeout_fn)
    label_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    selected_threshold = (
        max_f1_threshold(label_array, probability_array)
        if threshold is None
        else float(threshold)
    )
    metrics, predictions = binary_metrics(
        label_array, probability_array, selected_threshold
    )
    metrics["loss"] = weighted_loss / sample_count if sample_count else float("nan")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination / f"{split}_results.npz",
        labels=label_array,
        probabilities=probability_array,
        predictions=predictions,
        file_names=np.asarray(file_names, dtype=str),
        **{name: np.asarray(value) for name, value in metrics.items()},
    )
    return metrics


def run_seizure_supervised(args):
    from seizure_data import build_seizure_dataloaders
    from seizure_model import build_biot_seizure_model

    if args.model != "BIOT":
        raise ValueError("TUSZ/CHB-MIT contract integration is supported for model=BIOT")
    if args.use_fft:
        raise ValueError("BIOT requires raw input; set --no-use_fft")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_seizure_dataloaders(args)
    model, channel_policy = build_biot_seizure_model(args, loaders["train"].dataset)
    model.to(device)
    print(f"BIOT pretrained channel policy: {channel_policy}")
    criterion = training_criterion(loaders["train"].dataset, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    best_state = None
    for _ in range(args.epochs):
        model.train()
        for batch in loaders["train"]:
            train_one_batch(
                model,
                batch,
                optimizer,
                criterion,
                device,
                args.max_grad_norm,
                scaler=scaler,
                use_amp=use_amp,
            )
        dev_metrics = evaluate_and_save(
            model, loaders["val"], device, args.result_dir, "dev"
        )
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    dev_metrics = evaluate_and_save(
        model, loaders["val"], device, args.result_dir, "dev"
    )
    test_metrics = evaluate_and_save(
        model,
        loaders["test"],
        device,
        args.result_dir,
        "test",
        threshold=dev_metrics["threshold"],
    )
    return {"dev": dev_metrics, "test": test_metrics, "channel_policy": channel_policy}
