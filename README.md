# IndoLR Visual Speech Recognition

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/pytorch-2.13-red?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-ready-green?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/dataset-IndoLR-orange)](https://www.kaggle.com/datasets/abasset/indolr)
[![uv](https://img.shields.io/badge/uv-venv-lightgrey?logo=astral)](https://docs.astral.sh/uv/)

This is a fork of a student's final project. My goal is to help them reproduce and improve their experiments on a proper GPU setup (RTX 5070).

---

## Two Approaches

| | V5 — Image Pipeline | V6 — Landmark Pipeline |
|---|---|---|
| **Input** | Grayscale mouth crops (80×80) | 81-dim vector: 40 facial landmarks × (x,y) + MAR |
| **Model** | LRCN (CNN + LSTM) — ~1.26M params | BiGRU + Attention — ~70K params |
| **Preprocessing** | Bounding box crop → square pad → resize | Landmark normalization + speech segment detection via MAR |
| **Sequence handling** | Center-crop / tail-pad | MAR-thresholded segment extraction + linear resampling |

Both share: 40 MediaPipe face mesh landmarks (full outer + inner lip rings), LOSO 8-fold cross-validation, IndoLR dataset (10 word classes + 4 phrase classes).

---

## Quick Start

### 1. Download the dataset

```bash
bash download_dataset.sh
```

Downloads ~4.3 GB of videos to `./data/` plus the MediaPipe face landmarker model.

### 2. Set up the environment

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

### 3. Run the notebooks

Open `LipReading_v5.ipynb` (image-based) or `LipReading_v6.ipynb` (landmark-based). The notebooks were originally written for Google Colab — adjust paths in the `CONFIG` cell for local use:

```python
CONFIG["ROOT_VIDEO"] = "./data"
CONFIG["OUT_ROOT"] = "./precomputed"
CONFIG["MODEL_TASK_PATH"] = "./face_landmarker.task"
```

---

## Repo Layout

```
.
├── download_dataset.sh   # Fetches IndoLR + MediaPipe model
├── requirements.txt      # Python deps (uv-compatible)
├── LipReading_v5.ipynb   # Image-based pipeline
├── LipReading_v6.ipynb   # Landmark-based pipeline
├── data/                 # Downloaded dataset (gitignored)
└── skills/               # Local notes (gitignored)
```
