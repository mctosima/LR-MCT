"""Aripin et al. image-based lip ROI preprocessing."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from src.preprocessing.utama_landmarks import (
    extract_landmarks_per_frame,
    init_detector,
)

LOGGER = logging.getLogger(__name__)
TARGET_SIZE = (80, 80)
MAX_FRAMES = 600
MIN_BOX_SIZE = 20
PAD_RATIO = 0.15


def compute_lip_bbox(
    lip_pts: np.ndarray,
    w: int,
    h: int,
    pad_ratio: float = PAD_RATIO,
) -> tuple[int, int, int, int] | None:
    """Return padded, image-clamped lip bounding box."""
    points = np.asarray(lip_pts, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError(f"Expected non-empty (N, 2) lip points, got {points.shape}")
    if w <= 0 or h <= 0 or pad_ratio < 0:
        raise ValueError("w, h must be positive and pad_ratio must be non-negative")

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    pad_w = (x_max - x_min) * pad_ratio
    pad_h = (y_max - y_min) * pad_ratio
    x1 = max(0, int(np.floor(x_min - pad_w)))
    y1 = max(0, int(np.floor(y_min - pad_h)))
    x2 = min(w, int(np.ceil(x_max + pad_w)))
    y2 = min(h, int(np.ceil(y_max + pad_h)))
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return None
    return x1, y1, x2, y2


def crop_to_square(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Crop ROI, square-pad, resize to 80x80, and convert to grayscale."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = crop.shape[:2]
    side = max(height, width)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = crop
    return cv2.resize(canvas, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def fix_sequence_center_pad(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    """Center-crop long ROI sequences and black-pad short sequences at tail."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    black = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
    if not frames:
        selected = [black] * seq_len
    elif len(frames) >= seq_len:
        start = (len(frames) - seq_len) // 2
        selected = frames[start : start + seq_len]
    else:
        selected = frames + [black] * (seq_len - len(frames))
    normalized = []
    for frame in selected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        normalized.append(resized.astype(np.float32) / 255.0)
    return np.stack(normalized).astype(np.float32, copy=False)


def process_video_roi(video_path: str | Path, seq_len: int, detector: Any) -> np.ndarray | None:
    """Extract fixed-length grayscale lip ROI sequence from one video."""
    capture = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    last_valid: np.ndarray | None = None
    valid_count = 0
    total = 0
    try:
        while capture.isOpened() and total < MAX_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            total += 1
            current = last_valid.copy() if last_valid is not None else np.zeros((*TARGET_SIZE[::-1], 3), dtype=np.uint8)
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if result.face_landmarks:
                    height, width = frame.shape[:2]
                    points = extract_landmarks_per_frame(result.face_landmarks[0], width, height)
                    bbox = compute_lip_bbox(points, width, height)
                    if bbox is not None:
                        roi = crop_to_square(frame, bbox)
                        if roi is not None:
                            current = roi
                            last_valid = roi.copy()
                            valid_count += 1
            except Exception as exc:
                LOGGER.debug("Frame %d failed for %s: %s", total, video_path, exc)
            frames.append(current)
    finally:
        capture.release()
    if valid_count == 0:
        return None
    return fix_sequence_center_pad(frames, seq_len)


def _class_number(class_name: str) -> int | None:
    match = re.match(r"^\s*(\d+)", class_name)
    return int(match.group(1)) if match else None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def run_preprocessing_aripin(
    video_root: str | Path,
    output_root: str | Path,
    detector: Any,
) -> dict[str, list[tuple[str, str, str]]]:
    """Process IndoLR videos into resumable Aripin ROI arrays."""
    video_root = Path(video_root)
    output_root = Path(output_root)
    result: dict[str, list[tuple[str, str, str]]] = {"word_samples": [], "phrase_samples": []}
    for video_path in sorted(video_root.glob("*/*/*.mp4")):
        speaker_dir, class_dir = video_path.parent.parent, video_path.parent
        class_number = _class_number(class_dir.name)
        if class_number is None:
            LOGGER.warning("Skipping class with no numeric prefix: %s", class_dir)
            continue
        scope = "word_samples" if class_number <= 10 else "phrase_samples"
        seq_len = 30 if class_number <= 10 else 40
        output_dir = output_root / class_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{_safe_name(speaker_dir.name)}__{video_path.stem}.npy"
        sample = (str(output_path), class_dir.name, speaker_dir.name)
        if output_path.exists():
            result[scope].append(sample)
            continue
        sequence = process_video_roi(video_path, seq_len, detector)
        if sequence is None:
            LOGGER.warning("No valid lip ROI detected; skipping %s", video_path)
            continue
        np.save(output_path, sequence)
        result[scope].append(sample)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("precomputed_aripin"))
    parser.add_argument("--model-path", type=Path, default=Path("face_landmarker.task"))
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    parser.add_argument("--sanity", action="store_true", help="Process only 1 speaker + 4 classes")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    detector = init_detector(args.model_path)
    if args.sanity:
        print(">>> SANITY preprocessing (1 speaker, 4 classes)", flush=True)
        vroot = Path(args.video_root)
        oroot = Path(args.output_root)
        speakers = sorted(p for p in vroot.iterdir() if p.is_dir())
        classes = sorted(p for p in speakers[0].iterdir() if p.is_dir())[:4]
        oroot.mkdir(parents=True, exist_ok=True)
        word_samples: list[tuple[str, str, str]] = []
        phrase_samples: list[tuple[str, str, str]] = []
        for class_dir in classes:
            cn = _class_number(class_dir.name)
            if cn is None:
                continue
            scope = "word" if cn <= 10 else "phrase"
            seq_len = 30 if cn <= 10 else 40
            out_dir = oroot / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for vp in sorted(class_dir.glob("*.mp4")):
                out_path = out_dir / f"{_safe_name(speakers[0].name)}__{vp.stem}.npy"
                if out_path.exists():
                    (word_samples if scope == "word" else phrase_samples).append((str(out_path), class_dir.name, speakers[0].name))
                    continue
                seq = process_video_roi(vp, seq_len, detector)
                if seq is None:
                    continue
                np.save(out_path, seq)
                (word_samples if scope == "word" else phrase_samples).append((str(out_path), class_dir.name, speakers[0].name))
        print(f"Processed words: {len(word_samples)}")
        print(f"Processed phrases: {len(phrase_samples)}")
    else:
        counts = run_preprocessing_aripin(args.video_root, args.output_root, detector)
        print(f"Processed words: {len(counts['word_samples'])}")
        print(f"Processed phrases: {len(counts['phrase_samples'])}")


if __name__ == "__main__":
    main()
