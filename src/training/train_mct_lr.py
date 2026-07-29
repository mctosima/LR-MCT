"""Train and evaluate MCT-LR fusion model."""

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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

from src.data.mct_dataset import MCTFusionDataset, collect_samples, split_train_val
from src.models.mct_lr import MCTLR, model_summary


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to batch max.

    ROI: zero-pad. Landmarks: replicate last valid frame.
    """
    rois, lms, labels = zip(*batch)
    T_img = max(r.shape[1] for r in rois)
    T_lm = max(l.shape[0] for l in lms)
    B = len(rois)
    H, W = rois[0].shape[-2], rois[0].shape[-1]
    lm_dim = lms[0].shape[-1]

    roi_batch = torch.zeros(B, 1, T_img, H, W)
    lm_batch = torch.zeros(B, T_lm, lm_dim)
    label_batch = torch.tensor(labels, dtype=torch.long)

    for i, (r, l) in enumerate(zip(rois, lms)):
        t_img = r.shape[1]
        t_lm = l.shape[0]
        roi_batch[i, :, :t_img] = r
        lm_batch[i, :t_lm] = l
        # replicate last frame for landmark padding
        if t_lm < T_lm:
            lm_batch[i, t_lm:] = l[-1]

    return roi_batch, lm_batch, label_batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(
    model: MCTLR,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: CosineAnnealingWarmRestarts | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true: list[int] = []
    predicted: list[int] = []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img, lm, labels in loader:
            img = img.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(img, lm)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            true.extend(labels.detach().cpu().tolist())
            predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if not true:
        raise ValueError("DataLoader produced no samples")
    if training and scheduler is not None:
        scheduler.step()
    n = len(true)
    return total_loss / n, accuracy_score(true, predicted), true, predicted


def train_mct_lr(
    train_ds: Dataset,
    val_ds: Dataset,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 0.0005,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str | Path = "output/checkpoints",
    log_dir: str | Path = "output/logs",
    run_name: str = "mct_lr",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one MCT-LR model and return validation metrics."""
    _set_seed(seed)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train_ds and val_ds must be non-empty")

    requested = device.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested not in {"cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    torch_device = torch.device(requested)
    print(f"Device: {torch_device}", flush=True)

    pin = torch_device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=pin, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=pin, collate_fn=_collate_fn,
    )

    model = MCTLR(num_classes=num_classes).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path(output_dir)
    log_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{run_name}_best.pth"
    csv_path = log_dir / f"{run_name}.csv"
    summary_path = log_dir / f"{run_name}.json"

    history: list[dict[str, float | int]] = []
    best_acc = -1.0
    best_epoch = 0
    best_loss = float("inf")
    patience = 15
    patience_counter = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, _, _ = _run_epoch(
                model, train_loader, criterion, torch_device, optimizer, scheduler,
            )
            val_loss, val_acc, _, _ = _run_epoch(model, val_loader, criterion, torch_device)

            row: dict[str, float | int] = {
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }
            history.append(row)
            writer.writerow(row)
            f.flush()

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": {"num_classes": num_classes},
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                        "seed": seed,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                flush=True,
            )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
                break

            if dry_run and epoch >= 2:
                print("DRY RUN: stopping after 2 epochs", flush=True)
                break

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
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
        "n_params": model.n_params,
        "model_summary": model_summary(model, (batch_size, 1, 30, 80, 80), (batch_size, 40, 80)),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "classification_report": report,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {k: summary[k] for k in ("run_name", "best_val_acc", "final_val_acc", "final_val_f1_macro", "n_params", "train_size", "val_size")},
        indent=2,
    ))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CONFIG: dict[str, dict[str, Any]] = {
    "words":   {"batch_size": 4, "lr": 0.0005, "epochs": 100},
    "phrases": {"batch_size": 4, "lr": 0.0005, "epochs": 100},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["random", "grouped"], required=True)
    parser.add_argument("--scope", choices=["words", "phrases"], required=True)
    parser.add_argument("--roi-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--lm-root", type=Path, default=Path("precomputed_utama"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sanity", action="store_true", help="3-epoch smoke test")
    parser.add_argument("--dry-run", action="store_true", help="2 epochs, save nothing")
    args = parser.parse_args()

    cfg = CONFIG[args.scope]
    effective_epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    if args.sanity:
        effective_epochs = 3
        print(f"SANITY MODE: {effective_epochs} epochs", flush=True)

    samples = collect_samples(args.roi_root, args.lm_root, args.scope)
    labels = [class_name for _, _, class_name, _, _ in samples]
    speakers = [speaker for _, _, _, speaker, _ in samples]
    unique_speakers = sorted(set(speakers))

    protocol = args.protocol
    if protocol == "grouped" and len(unique_speakers) < 2:
        print(f"WARNING: only {len(unique_speakers)} speaker(s); falling back to random", flush=True)
        protocol = "random"

    train_ds, val_ds = split_train_val(samples, labels, speakers, protocol=protocol, val_ratio=0.15, seed=args.seed)
    print(f"Split: train={len(train_ds)} val={len(val_ds)} protocol={protocol}", flush=True)

    train_mct_lr(
        train_ds,
        val_ds,
        num_classes=len(train_ds.class_to_idx),
        epochs=effective_epochs,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        seed=args.seed,
        device=args.device,
        run_name=f"mct_lr_{args.protocol}_{args.scope}",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
