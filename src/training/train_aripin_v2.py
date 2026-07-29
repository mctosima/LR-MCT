"""Train and evaluate ConvNeXt-inspired AripinV2 image model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.data.aripin_dataset import AripinROIDataset, collect_samples_aripin, split_train_val
from src.models.aripin_v2 import AripinV2, model_summary


BATCH_SIZE = 4
WARMUP_EPOCHS = 5
PATIENCE = 20


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_fn(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
    """Pad ROI sequences to the longest sequence in a batch."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    rois, labels = zip(*batch)
    if any(roi.ndim != 4 or roi.shape[0] != 1 or tuple(roi.shape[-2:]) != (80, 80) for roi in rois):
        raise ValueError("ROI samples must have shape (1, time, 80, 80)")
    if any(roi.shape[1] <= 0 for roi in rois):
        raise ValueError("ROI sequences must be non-empty")
    max_steps = max(int(roi.shape[1]) for roi in rois)
    roi_batch = torch.zeros(len(rois), 1, max_steps, 80, 80, dtype=rois[0].dtype)
    for index, roi in enumerate(rois):
        roi_batch[index, :, : roi.shape[1]] = roi
    return roi_batch, torch.tensor(labels, dtype=torch.long)


def augment_training_batch(img: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Apply ROI augmentation while preserving black tail-padding frames."""
    if img.ndim != 5 or img.shape[1] != 1 or tuple(img.shape[-2:]) != (80, 80):
        raise ValueError(f"Expected image batch (B, 1, T, 80, 80), got {tuple(img.shape)}")
    if img.shape[0] == 0 or img.shape[2] == 0:
        raise ValueError("Image batch and time dimension must be non-empty")
    augmented = img.clone()
    device = augmented.device
    batch = augmented.shape[0]
    padding_mask = augmented.abs().sum(dim=(1, 3, 4)) == 0.0
    contrast = 0.85 + 0.30 * torch.rand(batch, 1, 1, 1, 1, device=device, generator=generator)
    frame_mean = augmented.mean(dim=(-1, -2), keepdim=True)
    augmented = (augmented - frame_mean) * contrast + frame_mean
    brightness = -0.08 + 0.16 * torch.rand(batch, 1, 1, 1, 1, device=device, generator=generator)
    augmented = augmented + brightness
    noise = 0.015 * torch.randn(augmented.shape, device=device, dtype=augmented.dtype, generator=generator)
    augmented = (augmented + noise).clamp(0.0, 1.0)
    for index in range(batch):
        if padding_mask[index].any():
            augmented[index, :, padding_mask[index]] = 0.0
    return augmented


def _resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested not in {"cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    return torch.device(requested)


def _set_epoch_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, total_epochs: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        scale = epoch / WARMUP_EPOCHS
    else:
        progress = (epoch - WARMUP_EPOCHS) / max(total_epochs - WARMUP_EPOCHS, 1)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    lr = base_lr * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _run_epoch(
    model: AripinV2,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    rng: torch.Generator | None = None,
) -> tuple[float, float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                features = augment_training_batch(features, generator=rng)
                optimizer.zero_grad(set_to_none=True)
            if scaler is not None and training:
                with torch.amp.autocast("cuda"):
                    logits = model(features)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(features)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()
            count = labels.shape[0]
            total_loss += loss.item() * count
            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if not all_true:
        raise ValueError("DataLoader produced no samples")
    return (
        total_loss / len(all_true),
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro", zero_division=0),
        all_true,
        all_pred,
    )


def train_aripin_v2(
    train_ds: Dataset,
    val_ds: Dataset,
    num_classes: int,
    *,
    epochs: int = 100,
    batch_size: int = BATCH_SIZE,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str | Path = "output/checkpoints",
    log_dir: str | Path = "output/logs",
    run_name: str = "aripin_v2",
    metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one AripinV2 model and return validation metrics."""
    if num_classes <= 0 or epochs <= 0 or batch_size <= 0 or lr <= 0 or weight_decay < 0:
        raise ValueError("num_classes, epochs, batch_size, and lr must be positive; weight_decay cannot be negative")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train_ds and val_ds must be non-empty")
    _set_seed(seed)
    torch_device = _resolve_device(device)
    use_amp = torch_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    rng = torch.Generator(device=torch_device.type).manual_seed(seed) if use_amp else torch.Generator().manual_seed(seed)
    effective_epochs = min(epochs, 5) if dry_run else epochs
    model = AripinV2(num_classes=num_classes).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_amp,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_amp,
        collate_fn=_collate_fn,
    )
    output_dir = Path(output_dir)
    log_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}_best.pth"
    csv_path = log_dir / f"{run_name}.csv"
    summary_path = log_dir / f"{run_name}.json"
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    with csv_path.open("w", newline="") as handle:
        fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro", "lr"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, effective_epochs + 1):
            current_lr = _set_epoch_lr(optimizer, lr, epoch, effective_epochs)
            train_loss, train_acc, _, _, _ = _run_epoch(
                model, train_loader, criterion, torch_device, optimizer=optimizer, scaler=scaler, rng=rng
            )
            val_loss, val_acc, val_f1, y_true, y_pred = _run_epoch(model, val_loader, criterion, torch_device)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "lr": current_lr,
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()
            improved = val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss)
            if improved:
                best_f1 = val_f1
                best_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": {
                            "num_classes": num_classes,
                            "input_channels": 1,
                            "input_size": (80, 80),
                        },
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1_macro": val_f1,
                        "seed": seed,
                        "n_params": model.n_params,
                        "metadata": metadata or {},
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
            print(
                f"Epoch {epoch:03d}/{effective_epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} lr={current_lr:.6g}",
                flush=True,
            )
            if stale_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_loss, final_acc, final_f1, y_true, y_pred = _run_epoch(model, val_loader, criterion, torch_device)
    sequence_length = 30 if num_classes == 10 else 40
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary = {
        "run_name": run_name,
        "best_val_f1_macro": best_f1,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "final_val_loss": final_loss,
        "final_val_acc": final_acc,
        "final_val_f1_macro": final_f1,
        "n_params": model.n_params,
        "model_summary": model_summary(model, (batch_size, 1, sequence_length, 80, 80)),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "classification_report": report,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "metadata": metadata or {},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "run_name",
                    "best_val_f1_macro",
                    "final_val_acc",
                    "final_val_f1_macro",
                    "n_params",
                    "train_size",
                    "val_size",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["random", "grouped"], required=True)
    parser.add_argument("--scope", choices=["words", "phrases"], required=True)
    parser.add_argument("--precomputed-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sanity", action="store_true", help="Override to 3 epochs for quick smoke")
    parser.add_argument("--dry-run", action="store_true", help="Limit run to five epochs")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    effective_epochs = 3 if args.sanity else args.epochs
    if args.sanity:
        print(f"SANITY MODE: epochs {args.epochs} -> {effective_epochs}", flush=True)
    elif args.dry_run:
        print(f"DRY RUN: epochs {args.epochs} -> {min(args.epochs, 5)}", flush=True)
    samples = collect_samples_aripin(args.precomputed_root, args.scope)
    labels = [class_name for _, class_name, _ in samples]
    speakers = [speaker for _, _, speaker in samples]
    unique_speakers = sorted(set(speakers))
    actual_protocol = args.protocol
    if args.protocol == "grouped" and len(unique_speakers) < 2:
        print(f"WARNING: only {len(unique_speakers)} speaker(s); falling back to random split", flush=True)
        actual_protocol = "random"
    train_ds, val_ds = split_train_val(
        samples,
        labels,
        speakers,
        protocol=actual_protocol,
        val_ratio=0.15,
        seed=args.seed,
    )
    train_aripin_v2(
        train_ds,
        val_ds,
        num_classes=len(train_ds.classes),
        epochs=effective_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        run_name=f"aripin_v2_{args.protocol}_{args.scope}_seed{args.seed}",
        metadata={"protocol": args.protocol, "actual_protocol": actual_protocol, "scope": args.scope, "seed": args.seed},
        dry_run=args.dry_run and not args.sanity,
    )


if __name__ == "__main__":
    main()
