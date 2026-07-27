#!/usr/bin/env bash
set -euo pipefail

# Download IndoLR dataset + MediaPipe Face Landmarker model
# Usage: bash download_dataset.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
MODEL_PATH="${SCRIPT_DIR}/face_landmarker.task"

echo "============================================"
echo "  IndoLR Dataset Download Script"
echo "============================================"
echo "Data dir : ${DATA_DIR}"
echo "Model    : ${MODEL_PATH}"
echo ""

# ---- 1. Python + pip check ----
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+ first."
    exit 1
fi

# ---- 2. Install kagglehub ----
echo "[1/3] Installing kagglehub..."
pip3 install --quiet kagglehub

# ---- 3. Download IndoLR dataset from Kaggle ----
echo "[2/3] Downloading IndoLR dataset (abasset/indolr)..."
python3 -c "
import os, shutil, kagglehub

data_dir = os.environ.get('DATA_DIR', '${DATA_DIR}')
kaggle_path = kagglehub.dataset_download('abasset/indolr')
print(f'Downloaded to cache: {kaggle_path}')

os.makedirs(data_dir, exist_ok=True)
shutil.copytree(kaggle_path, data_dir, dirs_exist_ok=True)

# Count downloaded videos
n = 0
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.endswith('.mp4'):
            n += 1
print(f'Total videos copied: {n}')
" DATA_DIR="${DATA_DIR}"

# Fix nested folder issue (matching notebook cell workaround)
if [ -d "${DATA_DIR}/8 Glamor ( Bella )/8 Glamor ( Bella ) - Copy - Copy" ]; then
    echo "Fixing nested folder for '8 Glamor ( Bella )'..."
    mv "${DATA_DIR}/8 Glamor ( Bella )/8 Glamor ( Bella ) - Copy - Copy/"* "${DATA_DIR}/8 Glamor ( Bella )/" 2>/dev/null || true
    rmdir "${DATA_DIR}/8 Glamor ( Bella )/8 Glamor ( Bella ) - Copy - Copy" 2>/dev/null || true
fi

# ---- 4. Download MediaPipe Face Landmarker model ----
echo "[3/3] Downloading MediaPipe Face Landmarker model..."
if command -v wget &>/dev/null; then
    wget -q -O "${MODEL_PATH}" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
elif command -v curl &>/dev/null; then
    curl -sL -o "${MODEL_PATH}" \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
else
    echo "WARNING: neither wget nor curl found. Skipping model download."
    echo "Download manually from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
fi

if [ -f "${MODEL_PATH}" ]; then
    echo "Model saved to: ${MODEL_PATH} ($(du -h "${MODEL_PATH}" | cut -f1))"
fi

echo ""
echo "============================================"
echo "  Download complete!"
echo "  Data  : ${DATA_DIR}"
echo "  Model : ${MODEL_PATH}"
echo "============================================"
