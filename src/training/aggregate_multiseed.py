"""Aggregate per-seed benchmark JSONs into a multi-seed summary.

Usage:
    python -m src.training.aggregate_multiseed \
        --prefix output/logs/aripin_random_words \
        --output output/logs/aripin_random_words_multiseed.json

Reads files matching {prefix}_seed{seed}.json for seeds 42, 123, 2024 and writes
a summary with mean ± std of key metrics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

SEEDS = (42, 123, 2024)


def aggregate_multiseed(
    prefix: str,
    output: str | Path,
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        fpath = Path(f"{prefix}_seed{seed}.json")
        if not fpath.exists():
            raise FileNotFoundError(f"Missing seed file: {fpath}")
        with open(fpath) as fh:
            data = json.load(fh)
        per_seed.append({
            "seed": seed,
            "best_val_acc": data.get("best_val_acc", data.get("final_val_acc")),
            "final_val_f1_macro": data.get("final_val_f1_macro"),
            "best_epoch": data.get("best_epoch"),
        })

    accs = [d["best_val_acc"] for d in per_seed]
    f1s = [d["final_val_f1_macro"] for d in per_seed]
    epochs = [d["best_epoch"] for d in per_seed if d["best_epoch"] is not None]

    # n_params from seed 42 (same across seeds)
    with open(Path(f"{prefix}_seed42.json")) as fh:
        base_data = json.load(fh)

    summary: dict[str, Any] = {
        "run_name": f"{Path(prefix).stem}_multiseed",
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "best_val_acc_mean": statistics.mean(accs),
        "best_val_acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "final_val_f1_macro_mean": statistics.mean(f1s),
        "final_val_f1_macro_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        "best_epoch_mean": statistics.mean(epochs) if epochs else None,
        "n_params": base_data.get("n_params"),
        "per_seed": per_seed,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Aggregate written to {out}")
    print(f"  Acc:  {summary['best_val_acc_mean']:.4f} ± {summary['best_val_acc_std']:.4f}")
    print(f"  F1:   {summary['final_val_f1_macro_mean']:.4f} ± {summary['final_val_f1_macro_std']:.4f}")
    if summary["best_epoch_mean"] is not None:
        print(f"  Epoch mean: {summary['best_epoch_mean']:.1f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="Path prefix, e.g. output/logs/aripin_random_words")
    parser.add_argument("--output", required=True, help="Output aggregate JSON path")
    args = parser.parse_args()
    aggregate_multiseed(args.prefix, args.output)


if __name__ == "__main__":
    main()
