#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SANITY=""
if [[ "${1:-}" == "--sanity" ]]; then
    SANITY="--sanity"
    echo ">>> SANITY MODE: 1 speaker, 4 classes, 3 epochs <<<"
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON=".venv/bin/python"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
NOW=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/mct_lr2_summary_$NOW.txt"
exec > >(tee -a "$SUMMARY") 2>&1

echo "======================================================================="
echo "MCT-LR2 Fusion — started at $(date)"
echo "Sanity: ${SANITY:-off}"
echo "======================================================================="

echo ""
echo "[1/5] Preprocessing ROI (resumable) — $(date)"
$PYTHON -u -m src.preprocessing.aripin_roi \
  --video-root data \
  --output-root precomputed_aripin \
  --model-path face_landmarker.task \
  --log-level INFO \
  $SANITY \
  2>&1 | tee -a "$LOG_DIR/mct_lr2_preprocess_roi.log" || true

echo ""
echo "[2/5] Preprocessing landmarks (resumable) — $(date)"
$PYTHON -u -m src.preprocessing.utama_landmarks \
  --video-root data \
  --output-root precomputed_utama \
  --model-path face_landmarker.task \
  --log-level INFO \
  $SANITY \
  2>&1 | tee -a "$LOG_DIR/mct_lr2_preprocess_lm.log" || true

N_ROI=$($PYTHON -c "from pathlib import Path; print(sum(1 for _ in Path('precomputed_aripin').rglob('*.npy')))")
N_LM=$($PYTHON -c "from pathlib import Path; print(sum(1 for _ in Path('precomputed_utama').rglob('*.npy')))")
echo "Preprocessing: ROI=$N_ROI  landmarks=$N_LM"

if [[ -n "$SANITY" ]]; then
    echo ""
    echo "======================================================================="
    echo "[Sanity] Random words — 3 epochs — $(date)"
    echo "======================================================================="
    $PYTHON -u -m src.training.train_mct_lr2 \
      --protocol random --scope words --device auto --seed 42 $SANITY
    echo ""
    echo "Sanity done — $(date)"
    echo "======================================================================="
    exit 0
fi

echo ""
echo "======================================================================="
echo "[Training] 6 jobs — $(date)"
echo "======================================================================="

for PROTO in random grouped; do
  for SCOPE in words phrases; do
    echo ""
    echo "--- $PROTO $SCOPE --- $(date)"
    RAW_LOG="$LOG_DIR/mct_lr2_${PROTO}_${SCOPE}_raw.log"
    set +e
    $PYTHON -u -m src.training.train_mct_lr2 \
      --protocol "$PROTO" \
      --scope "$SCOPE" \
      --device auto \
      --seed 42 \
      > "$RAW_LOG" 2>&1
    RC=$?
    set -e
    cat "$RAW_LOG" | tee -a "$LOG_DIR/mct_lr2_${PROTO}_${SCOPE}.log"
    if [ "$RC" -ne 0 ]; then
      echo "TRAINING FAILED (exit $RC) for $PROTO $SCOPE — check $RAW_LOG"
    fi
  done
done

for SCOPE in words phrases; do
  echo ""
  echo "--- LOSO $SCOPE --- $(date)"
  RAW_LOG="$LOG_DIR/mct_lr2_loso_${SCOPE}_raw.log"
  set +e
  $PYTHON -u -m src.training.train_mct_lr2 \
    --protocol loso \
    --scope "$SCOPE" \
    --device auto \
    --seed 42 \
    > "$RAW_LOG" 2>&1
  RC=$?
  set -e
  cat "$RAW_LOG" | tee -a "$LOG_DIR/mct_lr2_loso_${SCOPE}.log"
  if [ "$RC" -ne 0 ]; then
    echo "TRAINING FAILED (exit $RC) for LOSO $SCOPE — check $RAW_LOG"
  fi
done

echo ""
echo "======================================================================="
echo "All done — $(date)"
echo "Summary: $SUMMARY"
echo "======================================================================="
