#!/bin/bash
# run_all_aripin.sh — preprocess → four Aripin LRCN-3Conv training jobs.
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON=".venv/bin/python"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
NOW=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/aripin_summary_$NOW.txt"
exec > >(tee -a "$SUMMARY") 2>&1

echo "======================================================================="
echo "Aripin & Setiawan LRCN-3Conv — started at $(date)"
echo "======================================================================="

echo "[1/5] Preprocessing (resumable) — $(date)"
$PYTHON -u -m src.preprocessing.aripin_roi \
  --video-root data \
  --output-root precomputed_aripin \
  --model-path face_landmarker.task \
  --log-level INFO \
  2>&1 | tee -a "$LOG_DIR/aripin_preprocessing.log"

for CONFIG in \
  "random words" \
  "grouped words" \
  "random phrases" \
  "grouped phrases"
do
  read PROTOCOL SCOPE <<<"$CONFIG"
  echo ""
  echo "======================================================================="
  echo "[Training] protocol=$PROTOCOL scope=$SCOPE — $(date)"
  echo "======================================================================="
  $PYTHON -u -m src.training.train_aripin \
    --protocol "$PROTOCOL" \
    --scope "$SCOPE" \
    --precomputed-root precomputed_aripin \
    --device auto \
    --epochs 100 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/aripin_${PROTOCOL}_${SCOPE}.log"
done

echo ""
echo "======================================================================="
echo "All done — $(date)"
echo "Summary: $SUMMARY"
echo "======================================================================="
