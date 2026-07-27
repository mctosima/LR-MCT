"""Dataset loading and train/validation protocols for Aripin LRCN."""

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


class AripinROIDataset(Dataset[tuple[Tensor, int]]):
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
        if array.ndim != 3 or array.shape[1:] != (80, 80):
            raise ValueError(f"Expected (time, 80, 80) ROI array: {path}; got {array.shape}")
        tensor = torch.from_numpy(array).float().unsqueeze(0)
        return tensor, self.labels[index]


def collect_samples_aripin(output_root: str | Path, scope: str) -> list[Sample]:
    """Collect precomputed Aripin samples for words or phrases."""
    if scope not in {"words", "phrases"}:
        raise ValueError("scope must be 'words' or 'phrases'")
    root = Path(output_root)
    if not root.exists():
        raise ValueError(f"Precomputed root does not exist: {root}")
    samples: list[Sample] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        match = re.match(r"^\s*(\d+)", class_dir.name)
        if not match:
            continue
        is_word = int(match.group(1)) <= 10
        if (scope == "words") != is_word:
            continue
        for path in sorted(class_dir.glob("*.npy")):
            speaker = path.name.split("__", 1)[0]
            samples.append((str(path), class_dir.name, speaker))
    if not samples:
        raise ValueError(f"No {scope} samples found in {root}")
    return samples


def split_train_val(
    samples: Sequence[Sample],
    classes: Sequence[str] | None = None,
    speakers: Sequence[str] | None = None,
    protocol: str = "random",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[AripinROIDataset, AripinROIDataset]:
    """Split samples using random stratification or speaker grouping."""
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
        train_speakers = set(groups[train_indices])
        val_speakers = set(groups[val_indices])
        if train_speakers & val_speakers:
            raise AssertionError("Grouped split has overlapping speakers")
    else:
        raise ValueError("protocol must be 'random' or 'grouped'")
    train_samples = [samples[int(index)] for index in train_indices]
    val_samples = [samples[int(index)] for index in val_indices]
    if not train_samples or not val_samples:
        raise ValueError("Split produced an empty train or validation set")
    return AripinROIDataset(train_samples), AripinROIDataset(val_samples)
