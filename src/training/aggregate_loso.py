"""Aggregate per-fold LOSO JSON files into a single summary.

Usage:
    python -m src.training.aggregate_loso \\
        --prefix output/logs/aripin_loso_words_fold \\
        --output output/logs/aripin_loso_words_seed42_aggregate.json
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def aggregate_loso(
    prefix: str,
    output: str | Path,
) -> dict[str, Any]:
    """Read per-fold JSON files matching *prefix* and write aggregate."""
    pattern = f"{prefix}*.json"
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    print(f"Found {len(files)} fold files for prefix {prefix}")

    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    folds: list[dict[str, Any]] = []
    fold_accs: list[float] = []
    fold_f1s: list[float] = []
    n_params = 0
    run_name = ""
    protocol = "loso"
    scope = "unknown"
    seed = 0

    for fpath in files:
        with open(fpath) as fh:
            data = json.load(fh)
        y_true = data.get("test_y_true")
        y_pred = data.get("test_y_pred")
        if y_true is None or y_pred is None:
            print(f"WARNING: {fpath} missing test_y_true/test_y_pred — skipping")
            continue
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)
        fold_acc = data.get("final_test_acc", float(accuracy_score(y_true, y_pred)))
        fold_f1 = data.get("final_test_f1_macro", float(f1_score(y_true, y_pred, average="macro", zero_division=0)))
        fold_accs.append(fold_acc)
        fold_f1s.append(fold_f1)
        if not n_params:
            n_params = data.get("n_params", 0)
        if not run_name:
            run_name = data.get("run_name", "").rsplit("_fold", 1)[0]
        if scope == "unknown":
            scope = data.get("scope", data.get("metadata", {}).get("scope", "unknown"))
        if not seed:
            seed = data.get("seed", data.get("metadata", {}).get("seed", 0))

        folds.append({
            "fold": data.get("fold", len(folds)),
            "test_speaker": data.get("test_speaker", "?"),
            "val_speaker": data.get("val_speaker", "?"),
            "train_size": data.get("train_size", 0),
            "val_size": data.get("val_size", 0),
            "test_size": len(y_true),
            "best_val_acc": data.get("best_val_acc"),
            "best_epoch": data.get("best_epoch"),
            "final_test_acc": fold_acc,
            "final_test_f1_macro": fold_f1,
            "checkpoint": data.get("checkpoint"),
        })

    if not pooled_true:
        raise ValueError("No fold data had test predictions")

    pooled_acc = float(accuracy_score(pooled_true, pooled_pred))
    pooled_f1 = float(f1_score(pooled_true, pooled_pred, average="macro", zero_division=0))
    acc_mean = float(np.mean(fold_accs))
    acc_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0
    f1_mean = float(np.mean(fold_f1s))
    f1_std = float(np.std(fold_f1s, ddof=1)) if len(fold_f1s) > 1 else 0.0

    aggregate: dict[str, Any] = {
        "run_name": f"{run_name}_seed{seed}",
        "protocol": protocol,
        "scope": scope,
        "seed": seed,
        "n_params": n_params,
        "folds": folds,
        "pooled_accuracy": pooled_acc,
        "pooled_f1_macro": pooled_f1,
        "fold_accuracy_mean": acc_mean,
        "fold_accuracy_std": acc_std,
        "fold_f1_macro_mean": f1_mean,
        "fold_f1_macro_std": f1_std,
        "pooled_classification_report": classification_report(
            pooled_true, pooled_pred, output_dict=True, zero_division=0,
        ),
        "pooled_y_true": pooled_true,
        "pooled_y_pred": pooled_pred,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2))
    print(f"Aggregate written to {out}")
    print(f"  Files: {len(folds)} folds")
    print(f"  Pooled accuracy: {pooled_acc:.4f}")
    print(f"  Pooled F1 macro: {pooled_f1:.4f}")
    print(f"  Fold accuracy:   {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Fold F1 macro:   {f1_mean:.4f} ± {f1_std:.4f}")

    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Path prefix for per-fold JSONs, e.g. output/logs/aripin_loso_words_fold")
    parser.add_argument("--output", required=True, help="Output aggregate JSON path")
    args = parser.parse_args()
    aggregate_loso(args.prefix, args.output)


if __name__ == "__main__":
    main()
