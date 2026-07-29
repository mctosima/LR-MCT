"""Train and evaluate MCT-LR2 fusion model with speaker-adversarial regularisation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.data.mct_dataset import (
    MCTFusionDataset,
    collect_samples,
    make_loso_folds,
    split_train_val,
)
from src.models.mct_lr2 import MCTLR2, model_summary


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def _collate_fn(
    batch: list[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad variable-length ROI + landmark sequences per batch.

    ROI: zero-pad to batch max T_img.
    Landmarks: zero-pad, then replicate last valid frame.
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
    rois, lms, class_labels, speaker_labels = zip(*batch)
    T_img = max(r.shape[1] for r in rois)
    T_lm = max(l.shape[0] for l in lms)
    B = len(rois)
    _H, _W = rois[0].shape[-2], rois[0].shape[-1]
    lm_dim = lms[0].shape[-1]

    roi_batch = torch.zeros(B, 1, T_img, _H, _W)
    lm_batch = torch.zeros(B, T_lm, lm_dim)
    class_batch = torch.tensor(class_labels, dtype=torch.long)
    speaker_batch = torch.tensor(speaker_labels, dtype=torch.long)

    for i, (r, l) in enumerate(zip(rois, lms)):
        if l.shape[0] == 0:
            raise ValueError("Landmark sequence is empty")
        t_r = r.shape[1]
        t_l = l.shape[0]
        roi_batch[i, :, :t_r] = r
        lm_batch[i, :t_l] = l
        if t_l < T_lm:
            lm_batch[i, t_l:] = l[-1]
    return roi_batch, lm_batch, class_batch, speaker_batch


# ---------------------------------------------------------------------------
# Augmentation helper
# ---------------------------------------------------------------------------

def augment_training_batch(
    img: Tensor,
    lm: Tensor,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Augment a training batch in-place.

    ``img``: ``(B, 1, T, 80, 80)`` float, ``lm``: ``(B, T, 80)`` float.
    Returns modified copies.
    """
    img_aug = img.clone()
    lm_aug = lm.clone()
    B = img_aug.shape[0]
    device = img_aug.device
    if generator is not None:
        g_state = generator.get_state()
    else:
        g_state = torch.default_generator.get_state()

    # --- ROI augmentations ---
    # Capture zero-padding mask (frames where all pixels are exactly zero)
    padding_mask = img_aug.abs().sum(dim=(1, 3, 4)) == 0.0  # (B, T)

    # Contrast  [0.8, 1.2]
    contrast = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, 1, device=device, generator=generator)
    mean_img = img_aug.mean(dim=(-1, -2), keepdim=True)
    img_aug = (img_aug - mean_img) * contrast + mean_img

    # Brightness [-0.1, 0.1]
    brightness = (-0.1) + 0.2 * torch.rand(B, 1, 1, 1, 1, device=device, generator=generator)
    img_aug = img_aug + brightness

    # Gaussian noise σ=0.02
    noise = 0.02 * torch.randn_like(img_aug, generator=generator)
    img_aug = img_aug + noise

    img_aug.clamp_(0.0, 1.0)

    # Restore zero-padding frames
    for i in range(B):
        pad_frames = padding_mask[i]  # (T,)
        if pad_frames.any():
            img_aug[i, :, pad_frames] = 0.0

    # --- Landmark augmentations ---
    lm_reshaped = lm_aug.reshape(B, -1, 40, 2)  # (B, T, 40, 2)
    T_lm = lm_reshaped.shape[1]

    # Scale around centre 0.5
    scale = 0.95 + 0.1 * torch.rand(B, 1, device=device, generator=generator)
    centre = 0.5
    lm_reshaped = (lm_reshaped - centre) * scale.view(B, 1, 1, 1) + centre

    # Translation [-0.025, 0.025]
    transl = (-0.025) + 0.05 * torch.rand(B, 1, 2, device=device, generator=generator)
    lm_reshaped = lm_reshaped + transl.view(B, 1, 1, 2)

    # Independent Gaussian noise σ=0.005
    noise_lm = 0.005 * torch.randn(B, T_lm, 40, 2, device=device, generator=generator)
    lm_reshaped = lm_reshaped + noise_lm

    lm_reshaped.clamp_(0.0, 1.0)
    lm_aug = lm_reshaped.reshape(B, -1, 80)

    # Restore generator state if supplied
    if generator is not None:
        generator.set_state(g_state)

    return img_aug, lm_aug


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_epoch(
    model: MCTLR2,
    loader: DataLoader,
    class_criterion: nn.Module,
    speaker_criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_accum: int = 1,
    progress_start: float = 0.0,
    progress_end: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    rng: torch.Generator | None = None,
    accumulator_step: int = 0,
) -> tuple[float, float, float, float, list[int], list[int], int]:
    training = optimizer is not None
    model.train(training)
    total_class_loss = 0.0
    total_speaker_loss = 0.0
    total_n = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    step_local = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for img, lm, class_labels, speaker_labels in loader:
            img = img.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            class_labels = class_labels.to(device, non_blocking=True)
            speaker_labels = speaker_labels.to(device, non_blocking=True)

            # augment in training
            if training:
                img, lm = augment_training_batch(img, lm, generator=rng)

            # GRL progress interpolation
            if training:
                local_progress = progress_start + (progress_end - progress_start) * \
                    ((accumulator_step + step_local) / max(
                        (accumulator_step + len(loader)), 1))
                grl_scale = 2.0 / (1.0 + math.exp(-10.0 * min(local_progress, 1.0))) - 1.0
                grl_scale = max(grl_scale, 0.0)
            else:
                grl_scale = 0.0

            # forward
            if training and scaler is not None:
                with torch.amp.autocast("cuda"):
                    class_logits, sp_logits = model(img, lm, return_aux=True, grl_scale=float(grl_scale))
            else:
                if training:
                    class_logits, sp_logits = model(img, lm, return_aux=True, grl_scale=float(grl_scale))
                else:
                    class_logits = model(img, lm)

            class_loss = class_criterion(class_logits, class_labels)
            if training:
                sp_loss = speaker_criterion(sp_logits, speaker_labels)
                loss = class_loss + 0.1 * sp_loss
                loss = loss / grad_accum
            else:
                sp_loss = torch.tensor(float('nan'))
                loss = class_loss / grad_accum

            # backward
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            n = img.shape[0]
            total_class_loss += class_loss.item() * n
            if training and not torch.isnan(sp_loss):
                total_speaker_loss += sp_loss.item() * n
            total_n += n

            if training:
                all_true.extend(class_labels.detach().cpu().tolist())
                all_pred.extend(class_logits.argmax(dim=1).detach().cpu().tolist())

                # step after accumulation
                if (step_local + 1) % grad_accum == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step_local += 1
            else:
                all_true.extend(class_labels.detach().cpu().tolist())
                all_pred.extend(class_logits.argmax(dim=1).detach().cpu().tolist())

    # finalise incomplete accumulation
    if training and step_local % grad_accum != 0:
        remaining = step_local % grad_accum
        remaining_params = [p for p in model.parameters() if p.grad is not None]
        if remaining_params:
            for p in remaining_params:
                p.grad.mul_(remaining / grad_accum)
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if not all_true:
        raise ValueError("DataLoader produced no samples")

    avg_class_loss = total_class_loss / total_n
    avg_speaker_loss = total_speaker_loss / total_n if training else 0.0
    return (
        avg_class_loss,
        avg_speaker_loss,
        accuracy_score(all_true, all_pred),
        f1_score(all_true, all_pred, average="macro", zero_division=0),
        all_true,
        all_pred,
        accumulator_step + step_local,
    )


# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------

def train_mct_lr2(
    train_ds: Dataset,
    val_ds: Dataset,
    num_classes: int,
    num_speakers: int,
    *,
    test_ds: Dataset | None = None,
    epochs: int = 150,
    batch_size: int = 4,
    grad_accum: int = 1,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str | Path = "output/checkpoints",
    log_dir: str | Path = "output/logs",
    run_name: str = "mct_lr2",
    metadata: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one MCT-LR2 model and return validation metrics."""
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
    use_amp = torch_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    rng = torch.Generator(torch_device).manual_seed(seed) if torch_device.type == "cuda" else torch.Generator()

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
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=pin, collate_fn=_collate_fn,
        )

    model = MCTLR2(num_classes=num_classes, num_speakers=num_speakers).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    speaker_criterion = nn.CrossEntropyLoss()

    output_dir = Path(output_dir)
    log_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{run_name}_best.pth"
    csv_path = log_dir / f"{run_name}.csv"
    summary_path = log_dir / f"{run_name}.json"

    total_steps_per_epoch = math.ceil(len(train_ds) / batch_size / grad_accum)
    total_optimizer_steps = epochs * total_steps_per_epoch

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    patience = 20
    patience_counter = 0
    global_step = 0

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_class_loss", "train_speaker_loss",
            "train_acc", "train_f1", "val_loss", "val_acc", "val_f1_macro",
            "grl_scale",
        ])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            progress_start = global_step / max(total_optimizer_steps, 1)
            progress_end = (global_step + total_steps_per_epoch) / max(total_optimizer_steps, 1)

            train_class_loss, train_sp_loss, train_acc, train_f1, _, _, global_step = _run_epoch(
                model, train_loader, class_criterion, speaker_criterion,
                torch_device, optimizer, grad_accum,
                progress_start=progress_start, progress_end=progress_end,
                scaler=scaler, rng=rng, accumulator_step=global_step,
            )
            train_loss = train_class_loss + 0.1 * train_sp_loss

            val_class_loss, _, val_acc, val_f1, _, _, _ = _run_epoch(
                model, val_loader, class_criterion, speaker_criterion,
                torch_device, None,
            )

            progress = min(global_step / max(total_optimizer_steps, 1), 1.0)
            current_grl = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_class_loss": train_class_loss,
                "train_speaker_loss": train_sp_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_class_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
                "grl_scale": round(current_grl, 6),
            }
            history.append(row)
            writer.writerow(row)
            f.flush()

            if val_f1 > best_f1 or (val_f1 == best_f1 and val_class_loss < best_loss):
                best_f1 = val_f1
                best_loss = val_class_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": {
                            "num_classes": num_classes,
                            "num_speakers": num_speakers,
                            "img_size": [80, 80],
                            "modality_dropout": model.modality_dropout,
                        },
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "val_f1_macro": val_f1,
                        "val_loss": val_class_loss,
                        "seed": seed,
                        "class_to_idx": val_ds.class_to_idx if hasattr(val_ds, "class_to_idx") else {},
                        "speaker_to_idx": val_ds.speaker_to_idx if hasattr(val_ds, "speaker_to_idx") else {},
                        "metadata": metadata or {},
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                f"val_loss={val_class_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
                f"grl={current_grl:.4f}",
                flush=True,
            )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
                break

            if dry_run and epoch >= 5:
                print("DRY RUN: stopping after 5 epochs", flush=True)
                break

    # reload best checkpoint
    checkpoint = torch.load(ckpt_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # final validation evaluation
    _vl, _, final_val_acc, final_val_f1, val_true, val_pred, _ = _run_epoch(
        model, val_loader, class_criterion, speaker_criterion,
        torch_device, None,
    )
    val_report = classification_report(val_true, val_pred, output_dict=True, zero_division=0)
    final_val_loss = float(_vl)

    # final test evaluation (if available)
    test_result: dict[str, Any] = {
        "final_test_loss": None,
        "final_test_acc": None,
        "final_test_f1_macro": None,
        "test_classification_report": None,
        "test_y_true": None,
        "test_y_pred": None,
    }
    if test_loader is not None:
        t_loss, _, t_acc, t_f1, test_true, test_pred, _ = _run_epoch(
            model, test_loader, class_criterion, speaker_criterion,
            torch_device, None,
        )
        test_result = {
            "final_test_loss": t_loss,
            "final_test_acc": t_acc,
            "final_test_f1_macro": t_f1,
            "test_classification_report": classification_report(
                test_true, test_pred, output_dict=True, zero_division=0,
            ),
            "test_y_true": test_true,
            "test_y_pred": test_pred,
        }

    # inference timing
    model.eval()
    if test_loader is not None:
        timing_loader = test_loader
    else:
        timing_loader = val_loader
    inf_n = 0
    with torch.no_grad():
        # warm-up
        for img, lm, _, _ in timing_loader:
            _img = img.to(torch_device, non_blocking=True)
            _lm = lm.to(torch_device, non_blocking=True)
            model(_img, _lm)
            break
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for img, lm, _, _ in timing_loader:
            _img = img.to(torch_device, non_blocking=True)
            _lm = lm.to(torch_device, non_blocking=True)
            model(_img, _lm)
            inf_n += _img.shape[0]
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    inference_seconds_per_sample = (t1 - t0) / max(inf_n, 1)

    # model summary
    img_shape = (1, 1, 30 if num_classes == 10 else 40, 80, 80)
    lm_shape = (1, 40 if num_classes == 10 else 60, 80)

    summary: dict[str, Any] = {
        **test_result,
        "run_name": run_name,
        "best_val_f1_macro": best_f1,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "final_val_f1_macro": final_val_f1,
        "n_params": model.n_params,
        "model_summary": model_summary(model, img_shape, lm_shape),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds) if test_ds else 0,
        "classification_report": val_report,
        "checkpoint": str(ckpt_path),
        "history": history,
        "inference_seconds_per_sample": inference_seconds_per_sample,
        "seed": seed,
    }
    if metadata:
        summary.update(metadata)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {k: summary[k] for k in ("run_name", "best_val_f1_macro", "final_val_acc",
                                 "final_val_f1_macro", "n_params", "train_size", "val_size",
                                 "inference_seconds_per_sample")},
        indent=2,
    ))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CONFIG: dict[str, dict[str, Any]] = {
    "words":   {"batch_size": 4, "lr": 3e-4, "epochs": 150},
    "phrases": {"batch_size": 4, "lr": 3e-4, "epochs": 150},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["random", "grouped", "loso"], required=True)
    parser.add_argument("--scope", choices=["words", "phrases"], required=True)
    parser.add_argument("--roi-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--lm-root", type=Path, default=Path("precomputed_utama"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="LOSO fold index 0..7")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--sanity", action="store_true", help="3-epoch smoke test")
    parser.add_argument("--dry-run", action="store_true", help="5 epochs only")
    args = parser.parse_args()

    cfg = CONFIG[args.scope]
    effective_epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else cfg["batch_size"]
    grad_accum = args.grad_accum

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if grad_accum <= 0:
        raise ValueError("grad_accum must be positive")

    if args.sanity:
        effective_epochs = 3
        print(f"SANITY MODE: {effective_epochs} epochs", flush=True)
    if args.sanity and args.dry_run:
        effective_epochs = 3

    samples = collect_samples(args.roi_root, args.lm_root, args.scope)
    print(f"Matched samples: {len(samples)}", flush=True)

    # build class/speaker mappings
    all_classes = sorted({class_name for _, _, class_name, _, _ in samples})
    all_speakers = sorted({speaker for _, _, _, speaker, _ in samples})
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    speaker_to_idx = {s: i for i, s in enumerate(all_speakers)}

    if args.protocol == "loso":
        if len(all_speakers) != 8:
            raise SystemExit(
                f"LOSO requires exactly 8 speakers, found {len(all_speakers)}: {all_speakers}"
            )
    elif args.fold is not None:
        raise SystemExit("--fold requires --protocol loso")
    num_speakers = 8

    if args.protocol == "loso":
        folds = make_loso_folds(samples)
        fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))
        fold_results: list[dict[str, Any]] = []
        for fi in fold_indices:
            if fi < 0 or fi >= len(folds):
                raise ValueError(f"Fold {fi} out of range [0, {len(folds) - 1}]")
            test_sp, val_sp, tr_samples, va_samples, te_samples = folds[fi]
            train_ds = MCTFusionDataset(
                tr_samples, include_speaker=True,
                class_to_idx=class_to_idx, speaker_to_idx=speaker_to_idx,
            )
            val_ds = MCTFusionDataset(
                va_samples, include_speaker=True,
                class_to_idx=class_to_idx, speaker_to_idx=speaker_to_idx,
            )
            test_ds = MCTFusionDataset(
                te_samples, include_speaker=True,
                class_to_idx=class_to_idx, speaker_to_idx=speaker_to_idx,
            )
            safe_sp = _safe_filename(test_sp)
            run_name = f"mct_lr2_loso_{args.scope}_fold{fi}_{safe_sp}_seed{args.seed}"
            print(f"\n{'='*60}\nFold {fi}: test={test_sp}, val={val_sp}\n{'='*60}", flush=True)
            result = train_mct_lr2(
                train_ds, val_ds, num_classes=len(class_to_idx),
                num_speakers=num_speakers,
                test_ds=test_ds,
                epochs=effective_epochs, batch_size=batch_size,
                grad_accum=grad_accum,
                lr=cfg["lr"], seed=args.seed, device=args.device,
                run_name=run_name,
                metadata={
                    "protocol": args.protocol,
                    "scope": args.scope,
                    "fold": fi,
                    "test_speaker": test_sp,
                    "val_speaker": val_sp,
                },
                dry_run=args.dry_run,
            )
            fold_results.append(result)

        # aggregate
        pooled_true: list[int] = []
        pooled_pred: list[int] = []
        fold_accs: list[float] = []
        fold_f1s: list[float] = []
        fold_inftimes: list[float] = []
        for fr in fold_results:
            if fr.get("test_y_true") and fr.get("test_y_pred"):
                pooled_true.extend(fr["test_y_true"])
                pooled_pred.extend(fr["test_y_pred"])
            fold_accs.append(fr["final_test_acc"] or 0.0)
            fold_f1s.append(fr["final_test_f1_macro"] or 0.0)
            fold_inftimes.append(fr["inference_seconds_per_sample"])

        pooled_acc = accuracy_score(pooled_true, pooled_pred) if pooled_true else 0.0
        pooled_f1 = f1_score(pooled_true, pooled_pred, average="macro", zero_division=0) if pooled_true else 0.0
        acc_mean = float(np.mean(fold_accs))
        acc_std = float(np.std(fold_accs, ddof=1))
        f1_mean = float(np.mean(fold_f1s))
        f1_std = float(np.std(fold_f1s, ddof=1))
        inf_mean = float(np.mean(fold_inftimes))
        inf_std = float(np.std(fold_inftimes, ddof=1))

        aggregate_path = Path(f"output/logs/mct_lr2_loso_{args.scope}_seed{args.seed}_aggregate.json")
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate: dict[str, Any] = {
            "run_name": f"mct_lr2_loso_{args.scope}_seed{args.seed}",
            "protocol": args.protocol,
            "scope": args.scope,
            "seed": args.seed,
            "n_params": fold_results[0]["n_params"] if fold_results else 0,
            "folds": [
                {
                    "fold": r["fold"],
                    "test_speaker": r["test_speaker"],
                    "val_speaker": r["val_speaker"],
                    "train_size": r["train_size"],
                    "val_size": r["val_size"],
                    "test_size": r["test_size"],
                    "final_test_acc": r["final_test_acc"],
                    "final_test_f1_macro": r["final_test_f1_macro"],
                    "inference_seconds_per_sample": r["inference_seconds_per_sample"],
                    "checkpoint": r["checkpoint"],
                }
                for r in fold_results
            ],
            "pooled_accuracy": pooled_acc,
            "pooled_f1_macro": pooled_f1,
            "fold_accuracy_mean": acc_mean,
            "fold_accuracy_std": acc_std,
            "fold_f1_macro_mean": f1_mean,
            "fold_f1_macro_std": f1_std,
            "inference_seconds_mean": inf_mean,
            "inference_seconds_std": inf_std,
            "pooled_classification_report": classification_report(
                pooled_true, pooled_pred, output_dict=True, zero_division=0,
            ) if pooled_true else None,
            "pooled_y_true": pooled_true,
            "pooled_y_pred": pooled_pred,
        }
        aggregate_path.write_text(json.dumps(aggregate, indent=2))
        print(f"\nAggregate written to {aggregate_path}", flush=True)

    else:
        # random or grouped
        ci = [class_name for _, _, class_name, _, _ in samples]
        si = [speaker for _, _, _, speaker, _ in samples]
        train_ds, val_ds = split_train_val(
            samples, ci, si, protocol=args.protocol,
            val_ratio=0.15, seed=args.seed, include_speaker=True,
        )
        run_name = f"mct_lr2_{args.protocol}_{args.scope}_seed{args.seed}"
        print(f"Split: train={len(train_ds)} val={len(val_ds)} protocol={args.protocol}", flush=True)
        train_mct_lr2(
            train_ds, val_ds, num_classes=len(class_to_idx),
            num_speakers=num_speakers,
            epochs=effective_epochs, batch_size=batch_size,
            grad_accum=grad_accum,
            lr=cfg["lr"], seed=args.seed, device=args.device,
            run_name=run_name,
            metadata={
                "protocol": args.protocol,
                "scope": args.scope,
            },
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
