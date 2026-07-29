#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

SANITY=""
if [[ "${1:-}" == "--sanity" ]]; then
    SANITY="--sanity"
    echo ">>> SANITY MODE: 1 speaker, 4 classes, 3 epochs each <<<"
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHON=".venv/bin/python"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
NOW=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/mct_lr_summary_$NOW.txt"
exec > >(tee -a "$SUMMARY") 2>&1

echo "======================================================================="
echo "MCT-LR Fusion — started at $(date)"
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
  2>&1 | tee -a "$LOG_DIR/mct_lr_preprocess_roi.log" || true

echo ""
echo "[2/5] Preprocessing landmarks (resumable) — $(date)"
$PYTHON -u -m src.preprocessing.utama_landmarks \
  --video-root data \
  --output-root precomputed_utama \
  --model-path face_landmarker.task \
  --log-level INFO \
  $SANITY \
  2>&1 | tee -a "$LOG_DIR/mct_lr_preprocess_lm.log" || true

N_ROI=$($PYTHON -c "from pathlib import Path; print(sum(1 for _ in Path('precomputed_aripin').rglob('*.npy')))")
N_LM=$($PYTHON -c "from pathlib import Path; print(sum(1 for _ in Path('precomputed_utama').rglob('*.npy')))")
echo "Preprocessing: ROI=$N_ROI  landmarks=$N_LM"

echo ""
echo "======================================================================="
echo "[Training] 4 configs — $(date)"
echo "======================================================================="

for CONFIG in \
  "random words" \
  "grouped words" \
  "random phrases" \
  "grouped phrases"
do
  read PROTO SCOPE <<<"$CONFIG"
  echo ""
  echo "--- $PROTO $SCOPE --- $(date)"
  RAW_LOG="$LOG_DIR/mct_lr_${PROTO}_${SCOPE}_raw.log"
  set +e
  $PYTHON -u -m src.training.train_mct_lr \
    --protocol "$PROTO" \
    --scope "$SCOPE" \
    --device auto \
    --epochs 100 \
    --seed 42 \
    $SANITY \
    > "$RAW_LOG" 2>&1
  RC=$?
  set -e
  cat "$RAW_LOG" | tee -a "$LOG_DIR/mct_lr_${PROTO}_${SCOPE}.log"
  if [ "$RC" -ne 0 ]; then
    echo "TRAINING FAILED (exit $RC) for $PROTO $SCOPE — check $RAW_LOG"
  fi
done

echo ""
echo "======================================================================="
echo "All done — $(date)"
echo "Summary: $SUMMARY"
echo "======================================================================="
