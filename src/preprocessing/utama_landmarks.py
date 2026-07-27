"""Utama et al. landmark preprocessing baseline.

Extracts 40 MediaPipe mouth landmarks, normalizes each frame to its mouth
bounding box, and stores fixed-length 80-dimensional coordinate sequences.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LOGGER = logging.getLogger(__name__)
MAX_FRAMES = 600

LIPS_PAIRS: list[tuple[int, int]] = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
    (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
]


def _unique_indices_from_pairs(pairs: list[tuple[int, int]]) -> list[int]:
    seen: list[int] = []
    for left, right in pairs:
        if left not in seen:
            seen.append(left)
        if right not in seen:
            seen.append(right)
    return seen


LIPS_IDX = sorted(_unique_indices_from_pairs(LIPS_PAIRS))
N_LIP_POINTS = len(LIPS_IDX)
FEATURE_DIM = N_LIP_POINTS * 2
assert N_LIP_POINTS == 40, f"Expected 40 lip points, got {N_LIP_POINTS}"


def init_detector(model_path: str | Path) -> Any:
    """Create MediaPipe FaceLandmarker detector used by preprocessing."""
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_landmarks_per_frame(landmarks: list[Any], w: int, h: int) -> np.ndarray:
    """Return 40 mouth points as pixel-space float32 coordinates."""
    return np.asarray(
        [[landmarks[index].x * w, landmarks[index].y * h] for index in LIPS_IDX],
        dtype=np.float32,
    )


def normalize_landmarks(lip_pts: np.ndarray) -> np.ndarray:
    """Min-max normalize x/y independently inside each frame's mouth box."""
    points = np.asarray(lip_pts, dtype=np.float32)
    if points.shape != (N_LIP_POINTS, 2):
        raise ValueError(f"Expected {(N_LIP_POINTS, 2)} points, got {points.shape}")

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if np.isclose(x_max, x_min) or np.isclose(y_max, y_min):
        return np.full_like(points, 0.5, dtype=np.float32)

    normalized = np.empty_like(points, dtype=np.float32)
    normalized[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
    normalized[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
    return normalized


def _fix_sequence(frames: list[np.ndarray], seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if not frames:
        raise ValueError("Cannot fix empty frame sequence")

    if len(frames) >= seq_len:
        start = (len(frames) - seq_len) // 2
        selected = frames[start : start + seq_len]
    else:
        selected = frames + [frames[-1].copy() for _ in range(seq_len - len(frames))]
    return np.stack(selected).astype(np.float32, copy=False)


def process_video(video_path: str | Path, seq_len: int, detector: Any) -> np.ndarray | None:
    """Extract one fixed-size coordinate sequence from a video."""
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
            current = last_valid.copy() if last_valid is not None else np.zeros(FEATURE_DIM, dtype=np.float32)
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(image)
                if result.face_landmarks:
                    height, width = frame.shape[:2]
                    points = extract_landmarks_per_frame(result.face_landmarks[0], width, height)
                    current = normalize_landmarks(points).reshape(-1)
                    last_valid = current.copy()
                    valid_count += 1
            except Exception as exc:  # one bad frame must not discard video
                LOGGER.debug("Frame %d failed for %s: %s", total, video_path, exc)
            frames.append(current)
    finally:
        capture.release()

    if valid_count == 0:
        return None
    return _fix_sequence(frames, seq_len)


def _class_number(class_name: str) -> int | None:
    match = re.match(r"^\s*(\d+)", class_name)
    return int(match.group(1)) if match else None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def run_preprocessing(
    video_root: str | Path,
    output_root: str | Path,
    detector: Any,
) -> dict[str, list[tuple[str, str, str]]]:
    """Process all IndoLR videos and return word/phrase sample metadata."""
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
        seq_len = 40 if scope == "word_samples" else 60
        output_dir = output_root / class_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{_safe_name(speaker_dir.name)}__{video_path.stem}.npy"
        speaker_name = speaker_dir.name

        if output_path.exists():
            result[scope].append((str(output_path), class_dir.name, speaker_name))
            continue

        sequence = process_video(video_path, seq_len, detector)
        if sequence is None:
            LOGGER.warning("No landmarks detected; skipping %s", video_path)
            continue
        np.save(output_path, sequence)
        result[scope].append((str(output_path), class_dir.name, speaker_name))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("precomputed_utama"))
    parser.add_argument("--model-path", type=Path, default=Path("face_landmarker.task"))
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--sanity", action="store_true", help="Process only 1 speaker + 2 classes")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    detector = init_detector(args.model_path)
    if args.sanity:
        print(">>> SANITY preprocessing starting (1 speaker, 2 classes)", flush=True)
        vroot = Path(args.video_root)
        oroot = Path(args.output_root)
        speakers = sorted(p for p in vroot.iterdir() if p.is_dir())
        classes = sorted(p for p in speakers[0].iterdir() if p.is_dir())[:2]
        oroot.mkdir(parents=True, exist_ok=True)
        word_samples: list[tuple[str, str, str]] = []
        phrase_samples: list[tuple[str, str, str]] = []
        for class_dir in classes:
            cn = _class_number(class_dir.name)
            if cn is None:
                continue
            scope = "word" if cn <= 10 else "phrase"
            seq_len = 40 if scope == "word" else 60
            out_dir = oroot / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for vp in sorted(class_dir.glob("*.mp4")):
                out_path = out_dir / f"{_safe_name(speakers[0].name)}__{vp.stem}.npy"
                if out_path.exists():
                    (word_samples if scope == "word" else phrase_samples).append((str(out_path), class_dir.name, speakers[0].name))
                    continue
                seq = process_video(vp, seq_len, detector)
                if seq is None:
                    continue
                np.save(out_path, seq)
                (word_samples if scope == "word" else phrase_samples).append((str(out_path), class_dir.name, speakers[0].name))
        print(f"Processed words: {len(word_samples)}")
        print(f"Processed phrases: {len(phrase_samples)}")
    else:
        counts = run_preprocessing(args.video_root, args.output_root, detector)
        print(f"Processed words: {len(counts['word_samples'])}")
        print(f"Processed phrases: {len(counts['phrase_samples'])}")


if __name__ == "__main__":
    main()
