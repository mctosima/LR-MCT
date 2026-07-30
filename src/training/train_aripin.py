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

from src.data.aripin_dataset import AripinROIDataset, collect_samples_aripin, split_train_val
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
    parser.add_argument("--protocol", choices=["random", "grouped", "loso"], required=True)
    parser.add_argument("--scope", choices=["words", "phrases"], required=True)
    parser.add_argument("--precomputed-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="LOSO fold index 0..7")
    parser.add_argument("--sanity", action="store_true", help="Override to 3 epochs for quick smoke")
    args = parser.parse_args()
    if args.fold is not None and args.protocol != "loso":
        raise SystemExit("--fold requires --protocol loso")
    if args.sanity:
        effective_epochs = 3
        print(f"SANITY MODE: epochs {args.epochs} -> {effective_epochs}", flush=True)
    else:
        effective_epochs = args.epochs
    samples = collect_samples_aripin(args.precomputed_root, args.scope)
    labels = [class_name for _, class_name, _ in samples]
    speakers = [speaker for _, _, speaker in samples]
    unique_speakers = sorted(set(speakers))

    if args.protocol == "loso":
        if len(unique_speakers) != 8:
            raise SystemExit(
                f"LOSO requires exactly 8 speakers, found {len(unique_speakers)}: {unique_speakers}"
            )
        from src.data.aripin_dataset import make_loso_folds_aripin
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
            run_name = f"aripin_loso_{args.scope}_fold{fi}_{safe_sp}_seed{args.seed}"
            print(f"\n{'='*60}\nFold {fi}: test={test_sp}, val={val_sp}\n{'='*60}", flush=True)
            result = train_aripin(
                train_ds, val_ds,
                num_classes=len(train_ds.classes),
                epochs=effective_epochs,
                seed=args.seed,
                device=args.device,
                run_name=run_name,
            )
            # Evaluate on test set
            _device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
            model = LRCN3Conv(num_classes=len(train_ds.classes)).to(_device)
            ckpt = torch.load(result["checkpoint"], map_location=_device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
            true: list[int] = []
            pred: list[int] = []
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(_device)
                    logits = model(features)
                    true.extend(labels.tolist())
                    pred.extend(logits.argmax(dim=1).tolist())
            test_acc = accuracy_score(true, pred)
            test_f1 = f1_score(true, pred, average="macro", zero_division=0)
            print(f"Fold {fi} test_acc={test_acc:.4f} test_f1_macro={test_f1:.4f}", flush=True)
            result["test_y_true"] = true
            result["test_y_pred"] = pred
            result["final_test_acc"] = test_acc
            result["final_test_f1_macro"] = test_f1
            result["test_speaker"] = test_sp
            result["val_speaker"] = val_sp
            result["fold"] = fi
            # Persist test predictions back into the per-fold JSON
            fold_json_path = Path("output/logs") / f"{run_name}.json"
            if fold_json_path.exists():
                fold_data = json.loads(fold_json_path.read_text())
                fold_data["test_y_true"] = true
                fold_data["test_y_pred"] = pred
                fold_data["final_test_acc"] = test_acc
                fold_data["final_test_f1_macro"] = test_f1
                fold_data["test_speaker"] = test_sp
                fold_data["val_speaker"] = val_sp
                fold_data["fold"] = fi
                fold_json_path.write_text(json.dumps(fold_data, indent=2))
            fold_results.append(result)

        if args.fold is not None:
            print(f"\nSingle fold {args.fold} complete — no aggregation.", flush=True)
            return

        # Aggregate across folds
        pooled_true: list[int] = []
        pooled_pred: list[int] = []
        fold_accs: list[float] = []
        fold_f1s: list[float] = []
        for fr in fold_results:
            if fr.get("test_y_true") and fr.get("test_y_pred"):
                pooled_true.extend(fr["test_y_true"])
                pooled_pred.extend(fr["test_y_pred"])
            fold_accs.append(fr["final_test_acc"] or 0.0)
            fold_f1s.append(fr["final_test_f1_macro"] or 0.0)

        pooled_acc = accuracy_score(pooled_true, pooled_pred) if pooled_true else 0.0
        pooled_f1 = f1_score(pooled_true, pooled_pred, average="macro", zero_division=0) if pooled_true else 0.0
        acc_mean = float(np.mean(fold_accs))
        acc_std = float(np.std(fold_accs, ddof=1))
        f1_mean = float(np.mean(fold_f1s))
        f1_std = float(np.std(fold_f1s, ddof=1))

        aggregate_path = Path(f"output/logs/aripin_loso_{args.scope}_seed{args.seed}_aggregate.json")
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate: dict[str, Any] = {
            "run_name": f"aripin_loso_{args.scope}_seed{args.seed}",
            "protocol": args.protocol,
            "scope": args.scope,
            "seed": args.seed,
            "n_params": fold_results[0]["n_params"] if fold_results else 0,
            "folds": [
                {
                    "fold": fr["fold"],
                    "test_speaker": fr["test_speaker"],
                    "val_speaker": fr["val_speaker"],
                    "train_size": fr["train_size"],
                    "val_size": fr["val_size"],
                    "test_size": len(fr.get("test_y_true", [])),
                    "best_val_acc": fr["best_val_acc"],
                    "best_epoch": fr["best_epoch"],
                    "final_test_acc": fr["final_test_acc"],
                    "final_test_f1_macro": fr["final_test_f1_macro"],
                }
                for fr in fold_results
            ],
            "pooled_accuracy": pooled_acc,
            "pooled_f1_macro": pooled_f1,
            "fold_acc_mean": acc_mean,
            "fold_acc_std": acc_std,
            "fold_f1_mean": f1_mean,
            "fold_f1_std": f1_std,
        }
        aggregate_path.write_text(json.dumps(aggregate, indent=2))
        print(f"\nAggregate written to {aggregate_path}", flush=True)
        print(f"  Pooled acc: {pooled_acc:.4f}, Pooled F1: {pooled_f1:.4f}")
        print(f"  Fold mean acc: {acc_mean:.4f} ± {acc_std:.4f}")
        return

    # --- existing random/grouped path ---
    actual_protocol = args.protocol
    if args.protocol == "grouped" and len(unique_speakers) < 2:
        print(f"WARNING: only {len(unique_speakers)} speaker(s) in sanity mode; falling back to random split", flush=True)
        actual_protocol = "random"
    train_ds, val_ds = split_train_val(samples, labels, speakers, protocol=actual_protocol, val_ratio=0.15, seed=args.seed)
    train_aripin(
        train_ds,
        val_ds,
        num_classes=len(train_ds.classes),
        epochs=effective_epochs,
        seed=args.seed,
        device=args.device,
        run_name=f"aripin_{args.protocol}_{args.scope}",
    )


if __name__ == "__main__":
    main()
