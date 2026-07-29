#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p output/logs
SUMMARY="output/logs/aripin_v2_summary_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$SUMMARY") 2>&1

if [[ "${1:-}" == "--sanity" ]]; then
  echo "Running AripinV2 sanity check"
  .venv/bin/python -u -m src.training.train_aripin_v2 \
    --protocol random --scope words --device auto --sanity
  exit 0
fi

for protocol in random grouped; do
  for scope in words phrases; do
    echo "Running AripinV2: protocol=$protocol scope=$scope"
    .venv/bin/python -u -m src.training.train_aripin_v2 \
      --protocol "$protocol" \
      --scope "$scope" \
      --device auto \
      --epochs 100 \
      --seed 42
  done
done

echo "AripinV2 runs complete"
