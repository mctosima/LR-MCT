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

## Repo Layout

```
.
├── notebooks/               # Reference notebooks (Colab originals)
├── src/                     # Python source
│   ├── preprocessing/       # Landmark / ROI extraction
│   ├── models/              # BiGRU, LRCN architectures
│   ├── training/            # Training loop, LOSO CV
│   └── data/                # Dataset, dataloaders
├── output/                  # Checkpoints, logs, figures
├── docs/                    # Additional documentation
├── data/                    # Downloaded dataset (gitignored)
├── download_dataset.sh      # Fetches IndoLR + MediaPipe model
└── requirements.txt         # Python deps (uv-compatible)
```

## Documentation

- [Project documentation home](docs/index.html)
- [Literature review](docs/Literature-Review/index.html) — recent IndoLR studies, architecture comparison, metrics, and BibTeX
- [Experiment report](docs/Experiment-Report/index.html) — Utama et al. replication results under random and speaker-grouped validation

Latest replication results:

| Protocol | Words | Phrases |
|---|---:|---:|
| Random 85:15 validation | **94.70%** | **99.20%** |
| Speaker-grouped 85:15 validation | 50.65% | 25.00% |
