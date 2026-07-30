"""Post-evaluate saved LOSO checkpoints against their test sets.

Reads per-fold JSON from train_aripin (which has no test predictions),
loads the checkpoint, runs inference on the test fold, and writes
test predictions back into the JSON.

Usage:
    python -m src.training.evaluate_loso_checkpoints \
        --prefix output/logs/aripin_loso_words_fold \
        --precomputed-root precomputed_aripin \
        --scope words
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.data.aripin_dataset import (
    AripinROIDataset,
    collect_samples_aripin,
    make_loso_folds_aripin,
)
from src.models.lrcn import LRCN3Conv


def evaluate_single(fold_json_path: str, samples: list, folds: list) -> None:
    """Evaluate one fold checkpoint on its test set and update the JSON."""
    jpath = Path(fold_json_path)
    data = json.loads(jpath.read_text())
    ckpt_path = data.get("checkpoint")
    if not ckpt_path:
        print(f"SKIP {jpath.name}: no checkpoint field")
        return

    fold_idx = data.get("fold", -1)
    if fold_idx < 0:
        # Fallback: parse from filename pattern fold{N}_
        import re
        m = re.search(r'fold(\d+)_', jpath.name)
        fold_idx = int(m.group(1)) if m else -1
    if fold_idx < 0 or fold_idx >= len(folds):
        print(f"SKIP {jpath.name}: invalid fold {fold_idx}")
        return

    test_sp, val_sp, _, _, te_s = folds[fold_idx]
    test_ds = AripinROIDataset(te_s)
    num_classes = len(test_ds.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LRCN3Conv(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    true: list[int] = []
    pred: list[int] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            true.extend(labels.tolist())
            pred.extend(logits.argmax(dim=1).tolist())

    test_acc = float(accuracy_score(true, pred))
    test_f1 = float(f1_score(true, pred, average="macro", zero_division=0))

    data["test_y_true"] = true
    data["test_y_pred"] = pred
    data["final_test_acc"] = test_acc
    data["final_test_f1_macro"] = test_f1
    data["test_speaker"] = test_sp
    data["val_speaker"] = val_sp
    data["fold"] = fold_idx

    jpath.write_text(json.dumps(data, indent=2))
    print(f"  {jpath.name}: test={test_sp}, acc={test_acc:.4f}, f1={test_f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Prefix for per-fold JSONs")
    parser.add_argument("--precomputed-root", required=True, help="Path to precomputed_aripin")
    parser.add_argument("--scope", required=True, choices=["words", "phrases"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    # Build LOSO folds from the dataset
    print(f"Collecting {args.scope} samples from {args.precomputed_root}...")
    samples = collect_samples_aripin(args.precomputed_root, args.scope)
    folds = make_loso_folds_aripin(samples)
    print(f"  {len(folds)} folds built")

    pattern = f"{args.prefix}*.json"
    files = sorted(glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        return

    print(f"Found {len(files)} fold JSON files\n")
    for fpath in files:
        evaluate_single(fpath, samples, folds)


if __name__ == "__main__":
    main()
