#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== MCT-LR: preprocessing ==="
.venv/bin/python -m src.preprocessing.aripin_roi --all-speakers
.venv/bin/python -m src.preprocessing.utama_landmarks --all-speakers

echo "=== MCT-LR: training ==="
for job in \
  "random words" \
  "grouped words" \
  "random phrases" \
  "grouped phrases"; do
  read -r proto scope <<< "$job"
  echo "--- $proto $scope ---"
  .venv/bin/python -m src.training.train_mct_lr --protocol "$proto" --scope "$scope"
done

echo "=== MCT-LR: done ==="
