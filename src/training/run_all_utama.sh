#!/bin/bash
# run_all_utama.sh — single-pass Utama replication: preprocess → 4 training jobs.
# Usage:
#   Full run:    bash src/training/run_all_utama.sh
#   Sanity run:  bash src/training/run_all_utama.sh --sanity
set -euo pipefail

export PYTHONUNBUFFERED=1

SANITY=""
if [[ "${1:-}" == "--sanity" ]]; then
    SANITY="--sanity"
    echo ">>> SANITY MODE: 1 speaker, 2 classes, 3 epochs each <<<"
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON=".venv/bin/python"

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

NOW=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/utama_summary_$NOW.txt"
exec > >(tee -a "$SUMMARY") 2>&1

echo "======================================================================="
echo "Utama et al. Replication — started at $(date)"
echo "Sanity: ${SANITY:-off}"
echo "======================================================================="

# ── 1. Preprocessing (resumable) ──────────────────────────────────────
echo ""
echo "[1/5] Preprocessing (resumable) — $(date)"

$PYTHON -u -m src.preprocessing.utama_landmarks \
  --video-root data \
  --output-root precomputed_utama \
  --model-path face_landmarker.task \
  --log-level INFO \
  $SANITY \
  2>&1 | tee -a "$LOG_DIR/utama_preprocessing.log" || true

N_WORDS=$($PYTHON -c "from pathlib import Path; print(sum(1 for p in Path('precomputed_utama').rglob('*.npy') if int(p.parent.name.split()[0])<=10))")
N_PHRASES=$($PYTHON -c "from pathlib import Path; print(sum(1 for p in Path('precomputed_utama').rglob('*.npy') if int(p.parent.name.split()[0])>10))")
echo "Preprocessing done: words=$N_WORDS  phrases=$N_PHRASES  total=$((N_WORDS+N_PHRASES))"

# ── 2–5. Training ─────────────────────────────────────────────────────
for CONFIG in \
  "random words" \
  "grouped words" \
  "random phrases" \
  "grouped phrases"
do
  read PROTOCOL SCOPE <<<"$CONFIG"
  echo ""
  echo "======================================================================="
  echo "[Training] protocol=$PROTOCOL  scope=$SCOPE — $(date)"
  echo "======================================================================="
  $PYTHON -u -m src.training.train_utama \
    --protocol "$PROTOCOL" \
    --scope "$SCOPE" \
    --precomputed-root precomputed_utama \
    --device auto \
    --epochs 100 \
    --seed 42 \
    $SANITY \
    2>&1 | tee -a "$LOG_DIR/utama_${PROTOCOL}_${SCOPE}.log"
done

echo ""
echo "======================================================================="
echo "All done — $(date)"
echo "Summary:  $SUMMARY"
echo "Logs:     $LOG_DIR/utama_*.log"
echo "JSON:     $LOG_DIR/utama_*.json"
echo "======================================================================="
