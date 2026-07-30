#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

BATCH_SIZE="${BATCH_SIZE:-4}"
if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE must be a positive integer, got '$BATCH_SIZE'" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p output/logs
SUMMARY="output/logs/aripin_modified_v2_summary_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$SUMMARY") 2>&1

if [[ ! -d precomputed_aripin ]]; then
  echo "ERROR: precomputed_aripin/ not found. Run preprocessing first." >&2
  exit 1
fi

usage() {
  echo "Usage: $0 [--sanity] [--loso words|phrases]"
  echo ""
  echo "  (no args)     Run random/grouped benchmark (4 jobs, 100 epochs each)"
  echo "  --sanity      Run random words sanity check (3 epochs)"
  echo "  --loso words   Run 8-fold LOSO for words, then aggregate"
  echo "  --loso phrases Run 8-fold LOSO for phrases, then aggregate"
  exit 1
}

MODE="${1:-}"
SCOPE="${2:-}"

if [[ -z "$MODE" ]]; then
  echo "=== AripinModifiedV2 Benchmark ==="
  for config in "random words" "grouped words" "random phrases" "grouped phrases"; do
    read -r protocol scope <<< "$config"
    echo ""
    echo "--- $protocol $scope ---"
    .venv/bin/python -u -m src.training.train_aripin_modified_v2 \
      --protocol "$protocol" \
      --scope "$scope" \
      --device auto \
      --epochs 100 \
      --seed 42 \
      --batch-size "$BATCH_SIZE"
  done
elif [[ "$MODE" == "--sanity" ]]; then
  echo "=== AripinModifiedV2 Sanity ==="
  .venv/bin/python -u -m src.training.train_aripin_modified_v2 \
    --protocol random --scope words --device auto --sanity --batch-size "$BATCH_SIZE"
elif [[ "$MODE" == "--loso" ]]; then
  if [[ "$SCOPE" != "words" && "$SCOPE" != "phrases" ]]; then
    usage
  fi
  echo "=== AripinModifiedV2 LOSO $SCOPE ==="
  for fold in 0 1 2 3 4 5 6 7; do
    echo ""
    echo "--- LOSO $SCOPE fold $fold ---"
    .venv/bin/python -u -m src.training.train_aripin_modified_v2 \
      --protocol loso \
      --scope "$SCOPE" \
      --fold "$fold" \
      --device auto \
      --epochs 100 \
      --seed 42 \
      --batch-size "$BATCH_SIZE" \
      2>&1 | tee "output/logs/aripin_modified_v2_loso_${SCOPE}_fold${fold}.log"
  done
  echo ""
  echo "--- Aggregating $SCOPE LOSO ---"
  .venv/bin/python -m src.training.aggregate_loso \
    --prefix "output/logs/aripin_modified_v2_loso_${SCOPE}_fold" \
    --output "output/logs/aripin_modified_v2_loso_${SCOPE}_seed42_aggregate.json"
else
  usage
fi

echo ""
echo "=== AripinModifiedV2 complete: $SUMMARY ==="
