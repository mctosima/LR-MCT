#!/usr/bin/env bash
# run_multiseed_benchmark.sh — 3-seed random-words + random-phrases for Aripin & AripinModifiedV2
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p output/logs
[[ -d precomputed_aripin ]] || { echo "ERROR: precomputed_aripin/ missing. Run preprocessing first." >&2; exit 1; }

BATCH_SIZE="${BATCH_SIZE:-4}"
MODE="${1:-full}"   # full | sanity
SEEDS="42 123 2024"

run_one () {  # model protocol scope seed
  local model="$1" protocol="$2" scope="$3" seed="$4"
  if [[ "$model" == "aripin" ]]; then
    .venv/bin/python -u -m src.training.train_aripin \
      --protocol "$protocol" --scope "$scope" --device auto \
      --epochs 100 --seed "$seed" \
      2>&1 | tee "output/logs/aripin_${protocol}_${scope}_seed${seed}.log"
  else
    .venv/bin/python -u -m src.training.train_aripin_modified_v2 \
      --protocol "$protocol" --scope "$scope" --device auto \
      --epochs 100 --seed "$seed" --batch-size "$BATCH_SIZE" \
      2>&1 | tee "output/logs/aripin_modified_v2_${protocol}_${scope}_seed${seed}.log"
  fi
}

if [[ "$MODE" == "sanity" ]]; then
  echo "=== Multi-seed Sanity ==="
  .venv/bin/python -u -m src.training.train_aripin \
    --protocol random --scope words --device auto --sanity --seed 42
  .venv/bin/python -u -m src.training.train_aripin_modified_v2 \
    --protocol random --scope words --device auto --sanity --seed 42 --batch-size "$BATCH_SIZE"
  echo "SANITY OK"
  exit 0
fi

echo "=== Multi-seed Benchmark (random only) ==="
for model in aripin aripin_modified_v2; do
  for seed in $SEEDS; do
    for scope in words phrases; do
      echo ""
      echo "--- $model random $scope seed=$seed ---"
      run_one "$model" random "$scope" "$seed"
    done
  done
done
echo ""
echo "MULTI-SEED BENCHMARK COMPLETE"
