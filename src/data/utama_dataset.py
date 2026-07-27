"""Dataset loading and train/validation protocols for Utama replication."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import Tensor
from torch.utils.data import Dataset

Sample = tuple[str, str, str]


class UtamaLandmarkDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("Dataset cannot be empty")
        self.classes = sorted({class_name for _, class_name, _ in self.samples})
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[class_name] for _, class_name, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, _, _ = self.samples[index]
        array = np.load(path)
        if array.ndim != 2 or array.shape[1] != 80:
            raise ValueError(f"Expected ({array.shape[0]}, 80) landmark array: {path}; got {array.shape}")
        return torch.from_numpy(array).float(), self.labels[index]


def collect_samples(output_root: str | Path, scope: str) -> list[Sample]:
    """Collect precomputed samples for ``words`` or ``phrases``."""
    if scope not in {"words", "phrases"}:
        raise ValueError("scope must be 'words' or 'phrases'")
    samples: list[Sample] = []
    for class_dir in sorted(Path(output_root).iterdir()):
        if not class_dir.is_dir():
            continue
        match = re.match(r"^\s*(\d+)", class_dir.name)
        if not match:
            continue
        class_number = int(match.group(1))
        is_word = class_number <= 10
        if (scope == "words") != is_word:
            continue
        for path in sorted(class_dir.glob("*.npy")):
            speaker = path.name.split("__", 1)[0]
            samples.append((str(path), class_dir.name, speaker))
    if not samples:
        raise ValueError(f"No {scope} samples found in {output_root}")
    return samples


def split_train_val(
    samples: Sequence[Sample],
    classes: Sequence[int] | None = None,
    speakers: Sequence[str] | None = None,
    protocol: str = "random",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[UtamaLandmarkDataset, UtamaLandmarkDataset]:
    """Split samples with paper-style random or speaker-grouped protocol."""
    samples = list(samples)
    labels = np.asarray(classes if classes is not None else [class_name for _, class_name, _ in samples])
    groups = np.asarray(speakers if speakers is not None else [speaker for _, _, speaker in samples])
    if len(samples) != len(labels) or len(samples) != len(groups):
        raise ValueError("samples, classes, and speakers must have equal lengths")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    indices = np.arange(len(samples))
    if protocol == "random":
        train_indices, val_indices = train_test_split(
            indices, test_size=val_ratio, random_state=seed, stratify=labels
        )
    elif protocol == "grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_indices, val_indices = next(splitter.split(indices, labels, groups=groups))
    else:
        raise ValueError("protocol must be 'random' or 'grouped'")

    train_samples = [samples[int(index)] for index in train_indices]
    val_samples = [samples[int(index)] for index in val_indices]
    if not train_samples or not val_samples:
        raise ValueError("Split produced an empty train or validation set")
    return UtamaLandmarkDataset(train_samples), UtamaLandmarkDataset(val_samples)
