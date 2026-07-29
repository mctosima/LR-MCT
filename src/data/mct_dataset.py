"""MCT-LR fusion dataset — matched ROI + landmark samples.

Loads precomputed ROI (aripin) and landmark (utama) arrays from
``precomputed_aripin/`` and ``precomputed_utama/``, which use
identical ``speaker___video.npy`` basenames inside class directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import Tensor
from torch.utils.data import Dataset

Sample = tuple[str, str, str, str, str]  # (roi_path, lm_path, class_name, speaker, video_name)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MCTFusionDataset(Dataset):
    """Yields ``(roi, landmarks, label)`` or ``(roi, landmarks, label, speaker_index)``
    for matched fusion samples.

    Set ``include_speaker=True`` to receive a 4-tuple.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        include_speaker: bool = False,
        class_to_idx: dict[str, int] | None = None,
        speaker_to_idx: dict[str, int] | None = None,
    ) -> None:
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("Dataset cannot be empty")
        self.include_speaker = include_speaker

        if class_to_idx is not None:
            classes_present = {class_name for _, _, class_name, _, _ in self.samples}
            if not classes_present.issubset(class_to_idx.keys()):
                raise ValueError("class_to_idx is missing one or more classes present in samples")
            self.class_to_idx = class_to_idx
        else:
            classes = sorted({class_name for _, _, class_name, _, _ in self.samples})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.labels = [self.class_to_idx[c] for _, _, c, _, _ in self.samples]

        if include_speaker:
            if speaker_to_idx is not None:
                speakers_present = {sp for _, _, _, sp, _ in self.samples}
                if not speakers_present.issubset(speaker_to_idx.keys()):
                    raise ValueError("speaker_to_idx is missing one or more speakers present in samples")
                self.speaker_to_idx = speaker_to_idx
            else:
                speakers = sorted({sp for _, _, _, sp, _ in self.samples})
                self.speaker_to_idx = {sp: i for i, sp in enumerate(speakers)}
            self.speaker_labels = [self.speaker_to_idx[sp] for _, _, _, sp, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        roi_path, lm_path, _, _, _ = self.samples[index]
        roi = np.load(roi_path).astype(np.float32)
        lm = np.load(lm_path).astype(np.float32)
        if roi.ndim != 3 or roi.shape[1:] != (80, 80):
            raise ValueError(f"ROI shape mismatch in {roi_path}: {roi.shape}")
        if lm.ndim != 2 or lm.shape[1] != 80:
            raise ValueError(f"LM shape mismatch in {lm_path}: {lm.shape}")
        result = [
            torch.from_numpy(roi).unsqueeze(0),
            torch.from_numpy(lm),
            self.labels[index],
        ]
        if self.include_speaker:
            result.append(self.speaker_labels[index])
        return tuple(result)


# ---------------------------------------------------------------------------
# Sample collection — match across precomputed trees
# ---------------------------------------------------------------------------

def collect_samples(roi_root: str | Path, lm_root: str | Path, scope: str) -> list[Sample]:
    """Collect matched (ROI, landmark) samples for ``words`` or ``phrases``.

    Args:
        roi_root: Root of ``precomputed_aripin/``.
        lm_root: Root of ``precomputed_utama/``.
        scope: ``"words"`` or ``"phrases"``.

    Returns:
        List of ``(roi_path, lm_path, class_name, speaker, video_name)``.
    """
    if scope not in {"words", "phrases"}:
        raise ValueError("scope must be 'words' or 'phrases'")

    roi_root = Path(roi_root)
    lm_root = Path(lm_root)
    if not roi_root.exists():
        raise FileNotFoundError(f"ROI precomputed root missing: {roi_root}")
    if not lm_root.exists():
        raise FileNotFoundError(f"Landmark precomputed root missing: {lm_root}")

    matched: list[Sample] = []
    skipped_roi_only = 0
    skipped_lm_only = 0

    import re

    for class_dir in sorted(roi_root.iterdir()):
        if not class_dir.is_dir():
            continue
        match = re.match(r"^\s*(\d+)", class_dir.name)
        if not match:
            continue
        is_word = int(match.group(1)) <= 10
        if (scope == "words") != is_word:
            continue

        lm_class_dir = lm_root / class_dir.name
        if not lm_class_dir.is_dir():
            skipped_roi_only += len(list(class_dir.glob("*.npy")))
            continue

        roi_files = {p.name for p in class_dir.glob("*.npy")}
        lm_files = {p.name for p in lm_class_dir.glob("*.npy")}
        common = roi_files & lm_files

        skipped_roi_only += len(roi_files - lm_files)
        skipped_lm_only += len(lm_files - roi_files)

        for fname in sorted(common):
            speaker = fname.split("__", 1)[0]
            video = fname.replace(".npy", "")
            matched.append((
                str(class_dir / fname),
                str(lm_class_dir / fname),
                class_dir.name,
                speaker,
                video,
            ))

    total_skip = skipped_roi_only + skipped_lm_only
    if total_skip:
        pct = total_skip / max(len(matched) + total_skip, 1) * 100
        print(
            f"collect_samples matched={len(matched)} "
            f"skipped(roi_only={skipped_roi_only} lm_only={skipped_lm_only}) "
            f"({pct:.1f}% unmatched)",
            flush=True,
        )

    if not matched:
        raise ValueError(f"No matched {scope} samples between {roi_root} and {lm_root}")
    return matched


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_train_val(
    samples: Sequence[Sample],
    classes: Sequence[str] | None = None,
    speakers: Sequence[str] | None = None,
    protocol: str = "random",
    val_ratio: float = 0.15,
    seed: int = 42,
    *,
    include_speaker: bool = False,
) -> tuple[MCTFusionDataset, MCTFusionDataset]:
    """Split matched samples with random stratification or speaker grouping."""
    samples = list(samples)
    labels = np.asarray(classes if classes is not None else [class_name for _, _, class_name, _, _ in samples])
    groups = np.asarray(speakers if speakers is not None else [speaker for _, _, _, speaker, _ in samples])
    if len(samples) != len(labels) or len(samples) != len(groups):
        raise ValueError("classes / speakers lengths must match samples")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    indices = np.arange(len(samples))
    if protocol == "random":
        train_idx, val_idx = train_test_split(
            indices, test_size=val_ratio, random_state=seed, stratify=labels,
        )
    elif protocol == "grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(splitter.split(indices, labels, groups=groups))
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise AssertionError("Grouped split has overlapping speakers")
    else:
        raise ValueError("protocol must be 'random' or 'grouped'")

    train_samples = [samples[int(i)] for i in train_idx]
    val_samples = [samples[int(i)] for i in val_idx]
    if not train_samples or not val_samples:
        raise ValueError("Split produced empty set")

    if include_speaker:
        full_class_to_idx: dict[str, int] = {}
        full_speaker_to_idx: dict[str, int] = {}
        classes_sorted = sorted({class_name for _, _, class_name, _, _ in samples})
        speakers_sorted = sorted({speaker for _, _, _, speaker, _ in samples})
        full_class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
        full_speaker_to_idx = {s: i for i, s in enumerate(speakers_sorted)}
        return (
            MCTFusionDataset(train_samples, include_speaker=True, class_to_idx=full_class_to_idx, speaker_to_idx=full_speaker_to_idx),
            MCTFusionDataset(val_samples, include_speaker=True, class_to_idx=full_class_to_idx, speaker_to_idx=full_speaker_to_idx),
        )

    return MCTFusionDataset(train_samples), MCTFusionDataset(val_samples)


# ---------------------------------------------------------------------------
# LOSO folds
# ---------------------------------------------------------------------------

def make_loso_folds(
    samples: Sequence[Sample],
) -> list[tuple[str, str, list[Sample], list[Sample], list[Sample]]]:
    """Create leave-one-speaker-out folds from matched samples.

    Fold ``i`` holds out speaker ``i`` as test, speaker ``(i+1)%N`` as
    validation, and the remaining speakers as training.  Order is
    lexicographic by speaker name, so folds are deterministic.

    Returns:
        List of ``(test_speaker, val_speaker, train_samples, val_samples, test_samples)``.

    Raises:
        ValueError: if fewer than 3 unique speakers are present.
        AssertionError: if any partition is empty or speaker sets overlap.
    """
    samples = list(samples)
    speakers = sorted({sp for _, _, _, sp, _ in samples})
    n = len(speakers)
    if n < 3:
        raise ValueError(f"make_loso_folds requires at least 3 speakers, found {n}")

    folds: list[tuple[str, str, list[Sample], list[Sample], list[Sample]]] = []
    for i, test_sp in enumerate(speakers):
        val_sp = speakers[(i + 1) % n]
        train_sps = {s for j, s in enumerate(speakers) if j != i and j != (i + 1) % n}
        train_samples = [s for s in samples if s[3] in train_sps]
        val_samples = [s for s in samples if s[3] == val_sp]
        test_samples = [s for s in samples if s[3] == test_sp]
        assert train_samples, f"Fold {i}: train partition is empty"
        assert val_samples, f"Fold {i}: val partition is empty"
        assert test_samples, f"Fold {i}: test partition is empty"
        train_sp_set = {s[3] for s in train_samples}
        val_sp_set = {s[3] for s in val_samples}
        test_sp_set = {s[3] for s in test_samples}
        assert train_sp_set.isdisjoint(val_sp_set), f"Fold {i}: train/val overlap"
        assert train_sp_set.isdisjoint(test_sp_set), f"Fold {i}: train/test overlap"
        assert val_sp_set.isdisjoint(test_sp_set), f"Fold {i}: val/test overlap"
        folds.append((test_sp, val_sp, train_samples, val_samples, test_samples))

    return folds
