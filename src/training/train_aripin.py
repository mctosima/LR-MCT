"""Train and evaluate Aripin & Setiawan LRCN-3Conv."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.aripin_dataset import collect_samples_aripin, split_train_val
from src.models.lrcn import LRCN3Conv, model_summary


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: LRCN3Conv,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true: list[int] = []
    predicted: list[int] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            true.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
    if not true:
        raise ValueError("DataLoader produced no samples")
    return total_loss / len(true), accuracy_score(true, predicted), true, predicted


def train_aripin(
    train_ds: Dataset,
    val_ds: Dataset,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 0.0005,
    weight_decay: float = 1e-5,
    dropout_cnn: float = 0.25,
    dropout_lstm: float = 0.5,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str | Path = "output/checkpoints",
    log_dir: str | Path = "output/logs",
    run_name: str = "aripin",
) -> dict[str, Any]:
    """Train one paper-configured image LRCN and return validation metrics."""
    if not 0 <= dropout_cnn <= 1 or not 0 <= dropout_lstm <= 1:
        raise ValueError("dropout values must be between 0 and 1")
    if dropout_cnn != 0.25 or dropout_lstm != 0.5:
        raise ValueError("Aripin replication requires dropout_cnn=0.25 and dropout_lstm=0.5")
    _set_seed(seed)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train_ds and val_ds must be non-empty")
    requested_device = device.lower()
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested_device not in {"cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    torch_device = torch.device(requested_device)
    pin_memory = torch_device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    model = LRCN3Conv(num_classes=num_classes).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(output_dir)
    log_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_name}_best.pth"
    csv_path = log_dir / f"{run_name}.csv"
    summary_path = log_dir / f"{run_name}.json"
    history: list[dict[str, float | int]] = []
    best_acc = -1.0
    best_epoch = 0
    best_loss = float("inf")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, _, _ = _run_epoch(model, train_loader, criterion, torch_device, optimizer)
            val_loss, val_acc, _, _ = _run_epoch(model, val_loader, criterion, torch_device)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_loss = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": {
                            "num_classes": num_classes,
                            "input_channels": 1,
                            "input_size": (80, 80),
                            "dropout_cnn": dropout_cnn,
                            "dropout_lstm": dropout_lstm,
                        },
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "seed": seed,
                    },
                    checkpoint_path,
                )
            print(
                f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_loss, final_acc, y_true, y_pred = _run_epoch(model, val_loader, criterion, torch_device)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    summary = {
        "run_name": run_name,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "final_val_loss": final_loss,
        "final_val_acc": final_acc,
        "final_val_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "n_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_summary": model_summary(model, (batch_size, 1, 30 if num_classes == 10 else 40, 80, 80)),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "classification_report": report,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ("run_name", "best_val_acc", "final_val_acc", "final_val_f1_macro", "n_params", "train_size", "val_size")}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["random", "grouped"], required=True)
    parser.add_argument("--scope", choices=["words", "phrases"], required=True)
    parser.add_argument("--precomputed-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    samples = collect_samples_aripin(args.precomputed_root, args.scope)
    labels = [class_name for _, class_name, _ in samples]
    speakers = [speaker for _, _, speaker in samples]
    train_ds, val_ds = split_train_val(samples, labels, speakers, protocol=args.protocol, val_ratio=0.15, seed=args.seed)
    train_aripin(
        train_ds,
        val_ds,
        num_classes=len(train_ds.classes),
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        run_name=f"aripin_{args.protocol}_{args.scope}",
    )


if __name__ == "__main__":
    main()
