"""Train and evaluate AripinModifiedV2 two-stream image model."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.aripin_dataset import (
    AripinROIDataset,
    collect_samples_aripin,
    make_loso_folds_aripin,
    split_train_val,
)
from src.models.aripin_modified_v2 import AripinModifiedV2, model_summary

BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested not in {"cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    return torch.device(requested)


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


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------


def augment_training_batch(img: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Apply photometric + affine augmentation, preserving black tail padding."""
    if img.ndim != 5 or img.shape[1] != 1 or tuple(img.shape[-2:]) != (80, 80):
        raise ValueError(f"Expected image batch (B, 1, T, 80, 80), got {tuple(img.shape)}")
    if img.shape[0] == 0 or img.shape[2] == 0:
        raise ValueError("Image batch and time dimension must be non-empty")
    augmented = img.clone()
    device = augmented.device
    batch = augmented.shape[0]

    # --- photometric (copied from V1) ---
    padding_mask = augmented.abs().sum(dim=(1, 3, 4)) == 0.0
    contrast = 0.85 + 0.30 * torch.rand(batch, 1, 1, 1, 1, device=device, generator=generator)
    frame_mean = augmented.mean(dim=(-1, -2), keepdim=True)
    augmented = (augmented - frame_mean) * contrast + frame_mean
    brightness = -0.08 + 0.16 * torch.rand(batch, 1, 1, 1, 1, device=device, generator=generator)
    augmented = augmented + brightness
    noise = 0.015 * torch.randn(augmented.shape, device=device, dtype=augmented.dtype, generator=generator)
    # --- per-sample time-consistent affine jitter ---
    # Flatten time into batch for per-frame 2D affine
    B, C, T, H, W = augmented.shape
    flat = augmented.reshape(B * T, C, H, W)
    for i in range(batch):
        scale = 0.95 + 0.10 * torch.rand(1, device=device, generator=generator).item()
        tx = (-0.05 + 0.10 * torch.rand(1, device=device, generator=generator).item()) * 2.0
        ty = (-0.05 + 0.10 * torch.rand(1, device=device, generator=generator).item()) * 2.0
        theta = torch.tensor(
            [[scale, 0.0, tx], [0.0, scale, ty]],
            device=device,
            dtype=augmented.dtype,
        ).unsqueeze(0)
        for t_idx in range(T):
            idx = i * T + t_idx
            grid = F.affine_grid(theta, flat[idx:idx + 1].shape, align_corners=False)
            flat[idx:idx + 1] = F.grid_sample(
                flat[idx:idx + 1], grid, mode="bilinear",
                padding_mode="zeros", align_corners=False,
            )
    augmented = flat.reshape(B, C, T, H, W)
    for i in range(batch):
        if padding_mask[i].any():
            augmented[i, :, padding_mask[i]] = 0.0

    return augmented


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _run_epoch(
    model: AripinModifiedV2,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    rng: torch.Generator | None = None,
    max_norm: float = 5.0,
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
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(features)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm)
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


def train_aripin_modified_v2(
    train_ds: Dataset,
    val_ds: Dataset,
    num_classes: int,
    *,
    test_ds: Dataset | None = None,
    epochs: int = 100,
    batch_size: int = BATCH_SIZE,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str | Path = "output/checkpoints",
    log_dir: str | Path = "output/logs",
    run_name: str = "aripin_modified_v2",
    metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one AripinModifiedV2 model and return validation metrics."""
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

    model = AripinModifiedV2(num_classes=num_classes).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
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
    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_f1 = 0.0

    with csv_path.open("w", newline="") as handle:
        fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1_macro"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, effective_epochs + 1):
            train_loss, train_acc, _, _, _ = _run_epoch(
                model, train_loader, criterion, torch_device, optimizer=optimizer, scaler=scaler, rng=rng
            )
            val_loss, val_acc, val_f1, y_true, y_pred = _run_epoch(
                model, val_loader, criterion, torch_device
            )
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
            history.append(row)
            writer.writerow(row)
            handle.flush()

            improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
            if improved:
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch
                best_f1 = val_f1
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
            print(
                f"Epoch {epoch:03d}/{effective_epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}",
                flush=True,
            )

    # load best checkpoint for final eval
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_loss, final_acc, final_f1, y_true, y_pred = _run_epoch(model, val_loader, criterion, torch_device)
    sequence_length = 30 if num_classes == 10 else 40
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "best_val_acc": best_acc,
        "best_val_loss": best_loss,
        "best_val_f1_macro": best_f1,
        "best_epoch": best_epoch,
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

    # --- test evaluation (optional) ---
    if test_loader is not None:
        model.eval()
        test_true: list[int] = []
        test_pred: list[int] = []
        test_total_loss = 0.0
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(torch_device, non_blocking=True)
                labels = labels.to(torch_device, non_blocking=True)
                logits = model(features)
                test_loss = criterion(logits, labels)
                test_total_loss += test_loss.item() * labels.size(0)
                test_true.extend(labels.cpu().tolist())
                test_pred.extend(logits.argmax(dim=1).cpu().tolist())
        test_count = len(test_true)
        summary["final_test_loss"] = test_total_loss / test_count if test_count else 0.0
        summary["final_test_acc"] = accuracy_score(test_true, test_pred) if test_count else 0.0
        summary["final_test_f1_macro"] = f1_score(test_true, test_pred, average="macro", zero_division=0) if test_count else 0.0
        summary["test_y_true"] = test_true
        summary["test_y_pred"] = test_pred
        summary["test_size"] = test_count
        if test_count:
            summary["test_classification_report"] = classification_report(test_true, test_pred, output_dict=True, zero_division=0)

    summary_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "run_name",
                    "best_val_acc",
                    "final_val_acc",
                    "final_val_f1_macro",
                    "best_val_f1_macro",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["random", "grouped", "loso"], required=True)
    parser.add_argument("--scope", choices=["words", "phrases"], required=True)
    parser.add_argument("--precomputed-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--fold", type=int, default=None, help="LOSO fold index 0..7")
    parser.add_argument("--sanity", action="store_true", help="Override to 3 epochs for quick smoke")
    parser.add_argument("--dry-run", action="store_true", help="Limit run to 5 epochs")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    if args.fold is not None and args.protocol != "loso":
        raise SystemExit("--fold requires --protocol loso")

    effective_epochs = 3 if args.sanity else args.epochs
    if args.sanity:
        print(f"SANITY MODE: epochs {args.epochs} -> {effective_epochs}", flush=True)
    elif args.dry_run:
        print(f"DRY RUN: epochs {args.epochs} -> {min(args.epochs, 5)}", flush=True)

    samples = collect_samples_aripin(args.precomputed_root, args.scope)
    labels = [class_name for _, class_name, _ in samples]
    speakers = [speaker for _, _, speaker in samples]
    unique_speakers = sorted(set(speakers))

    if args.protocol == "loso":
        if len(unique_speakers) != 8:
            raise SystemExit(
                f"LOSO requires exactly 8 speakers, found {len(unique_speakers)}: {unique_speakers}"
            )
        folds = make_loso_folds_aripin(samples)
        fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))
        fold_results: list[dict[str, Any]] = []
        for fi in fold_indices:
            if fi < 0 or fi >= len(folds):
                raise ValueError(f"Fold {fi} out of range [0, {len(folds) - 1}]")
            test_sp, val_sp, tr_s, va_s, te_s = folds[fi]
            train_ds = AripinROIDataset(tr_s)
            val_ds = AripinROIDataset(va_s)
            test_ds = AripinROIDataset(te_s)
            safe_sp = test_sp.replace(" ", "_")
            run_name = f"aripin_modified_v2_loso_{args.scope}_fold{fi}_{safe_sp}_seed{args.seed}"
            print(f"\n{'=' * 60}\nFold {fi}: test={test_sp}, val={val_sp}\n{'=' * 60}", flush=True)
            result = train_aripin_modified_v2(
                train_ds,
                val_ds,
                num_classes=len(train_ds.classes),
                test_ds=test_ds,
                epochs=effective_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                device=args.device,
                run_name=run_name,
                metadata={
                    "protocol": args.protocol,
                    "scope": args.scope,
                    "fold": fi,
                    "test_speaker": test_sp,
                    "val_speaker": val_sp,
                    "seed": args.seed,
                },
                dry_run=args.dry_run and not args.sanity,
            )
            result["fold"] = fi
            result["test_speaker"] = test_sp
            result["val_speaker"] = val_sp
            # Persist fold/speaker/test into per-fold JSON on disk
            fold_json_path = Path("output/logs") / f"{run_name}.json"
            if fold_json_path.exists():
                fold_data = json.loads(fold_json_path.read_text())
                fold_data["fold"] = fi
                fold_data["test_speaker"] = test_sp
                fold_data["val_speaker"] = val_sp
                fold_data["test_y_true"] = result.get("test_y_true", [])
                fold_data["test_y_pred"] = result.get("test_y_pred", [])
                fold_data["final_test_acc"] = result.get("final_test_acc")
                fold_data["final_test_f1_macro"] = result.get("final_test_f1_macro")
                fold_json_path.write_text(json.dumps(fold_data, indent=2))
            fold_results.append(result)

        if args.fold is not None:
            print(f"\nSingle fold {args.fold} complete — no aggregation.", flush=True)
            return

        # aggregate across folds
        pooled_true: list[int] = []
        pooled_pred: list[int] = []
        fold_accs: list[float] = []
        fold_f1s: list[float] = []
        for fr in fold_results:
            if fr.get("test_y_true") and fr.get("test_y_pred"):
                pooled_true.extend(fr["test_y_true"])
                pooled_pred.extend(fr["test_y_pred"])
            fold_accs.append(fr.get("final_test_acc", 0.0))
            fold_f1s.append(fr.get("final_test_f1_macro", 0.0))

        pooled_acc = accuracy_score(pooled_true, pooled_pred) if pooled_true else 0.0
        pooled_f1 = f1_score(pooled_true, pooled_pred, average="macro", zero_division=0) if pooled_true else 0.0
        acc_mean = float(np.mean(fold_accs)) if fold_accs else 0.0
        acc_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0
        f1_mean = float(np.mean(fold_f1s)) if fold_f1s else 0.0
        f1_std = float(np.std(fold_f1s, ddof=1)) if len(fold_f1s) > 1 else 0.0

        aggregate_path = Path(f"output/logs/aripin_modified_v2_loso_{args.scope}_seed{args.seed}_aggregate.json")
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate: dict[str, Any] = {
            "run_name": f"aripin_modified_v2_loso_{args.scope}_seed{args.seed}",
            "protocol": args.protocol,
            "scope": args.scope,
            "seed": args.seed,
            "n_params": fold_results[0]["n_params"] if fold_results else 0,
            "folds": [
                {
                    "fold": fr.get("fold", fi),
                    "test_speaker": fr.get("test_speaker", "?"),
                    "val_speaker": fr.get("val_speaker", "?"),
                    "train_size": fr["train_size"],
                    "val_size": fr["val_size"],
                    "test_size": fr.get("test_size", len(fr.get("test_y_true", []))),
                    "best_val_acc": fr.get("best_val_acc"),
                    "best_epoch": fr.get("best_epoch"),
                    "final_test_acc": fr.get("final_test_acc"),
                    "final_test_f1_macro": fr.get("final_test_f1_macro"),
                    "checkpoint": fr.get("checkpoint"),
                }
                for fi, fr in enumerate(fold_results)
            ],
            "pooled_accuracy": pooled_acc,
            "pooled_f1_macro": pooled_f1,
            "fold_accuracy_mean": acc_mean,
            "fold_accuracy_std": acc_std,
            "fold_f1_macro_mean": f1_mean,
            "fold_f1_macro_std": f1_std,
            "pooled_classification_report": classification_report(
                pooled_true, pooled_pred, output_dict=True, zero_division=0,
            ) if pooled_true else None,
            "pooled_y_true": pooled_true,
            "pooled_y_pred": pooled_pred,
        }
        aggregate_path.write_text(json.dumps(aggregate, indent=2))
        print(f"\nAggregate written to {aggregate_path}", flush=True)
        print(f"  Pooled acc: {pooled_acc:.4f}, Pooled F1: {pooled_f1:.4f}")
        print(f"  Fold mean acc: {acc_mean:.4f} ± {acc_std:.4f}")
        return

    # --- random/grouped path ---
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
    train_aripin_modified_v2(
        train_ds,
        val_ds,
        num_classes=len(train_ds.classes),
        epochs=effective_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        run_name=f"aripin_modified_v2_{args.protocol}_{args.scope}_seed{args.seed}",
        metadata={
            "protocol": args.protocol,
            "actual_protocol": actual_protocol,
            "scope": args.scope,
            "seed": args.seed,
        },
        dry_run=args.dry_run and not args.sanity,
    )


if __name__ == "__main__":
    main()
