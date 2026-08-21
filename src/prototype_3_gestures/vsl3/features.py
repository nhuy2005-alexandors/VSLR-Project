from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


N_POSE = 25
N_HAND = 21
N_LANDMARKS = N_POSE + N_HAND * 2
FEATURE_DIM = N_LANDMARKS * 3
SEQUENCE_LENGTH = 60


@dataclass(frozen=True)
class FrameObservation:
    features: np.ndarray
    hands_present: bool


def _to_landmark_array(results) -> tuple[np.ndarray, bool]:
    pose = np.zeros((N_POSE, 3), dtype=np.float32)
    left = np.zeros((N_HAND, 3), dtype=np.float32)
    right = np.zeros((N_HAND, 3), dtype=np.float32)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for idx in range(min(N_POSE, len(landmarks))):
            lm = landmarks[idx]
            pose[idx] = (lm.x, lm.y, lm.z)

    if results.left_hand_landmarks:
        left[:] = np.asarray(
            [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        )
    if results.right_hand_landmarks:
        right[:] = np.asarray(
            [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        )

    points = np.concatenate((pose, left, right), axis=0)
    hands_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
    return points, hands_present


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    """Normalize translation/scale while preserving missing landmarks as zeros."""
    points = np.asarray(points, dtype=np.float32).reshape(N_LANDMARKS, 3).copy()
    valid = np.any(np.abs(points) > 1e-8, axis=1)
    if not np.any(valid):
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    left_shoulder = points[11]
    right_shoulder = points[12]
    shoulders_valid = valid[11] and valid[12]

    if shoulders_valid:
        center_xy = (left_shoulder[:2] + right_shoulder[:2]) / 2.0
        scale = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    else:
        valid_xy = points[valid, :2]
        center_xy = np.median(valid_xy, axis=0)
        span = np.ptp(valid_xy, axis=0)
        scale = float(max(span.max(), 1e-3))

    scale = max(scale, 1e-3)
    normalized = np.zeros_like(points, dtype=np.float32)
    normalized[valid, 0] = (points[valid, 0] - center_xy[0]) / scale
    normalized[valid, 1] = (points[valid, 1] - center_xy[1]) / scale
    normalized[valid, 2] = points[valid, 2] / scale
    np.clip(normalized, -6.0, 6.0, out=normalized)
    return normalized.reshape(-1)


def resample_sequence(sequence: np.ndarray, target_len: int = SEQUENCE_LENGTH) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [frames, {FEATURE_DIM}], got {sequence.shape}")
    if len(sequence) == 0:
        raise ValueError("Cannot resample an empty sequence")
    if len(sequence) == 1:
        return np.repeat(sequence, target_len, axis=0)

    source_t = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    target_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    output = np.empty((target_len, FEATURE_DIM), dtype=np.float32)
    for feature_idx in range(FEATURE_DIM):
        output[:, feature_idx] = np.interp(target_t, source_t, sequence[:, feature_idx])
    return output


def augment_sequence(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild landmark augmentation; intentionally avoids left/right mirroring."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [frames, {FEATURE_DIM}], got {sequence.shape}")

    n_frames = len(sequence)
    max_crop = max(1, int(round(n_frames * 0.08)))
    crop_left = int(rng.integers(0, max_crop + 1))
    crop_right = int(rng.integers(0, max_crop + 1))
    end = n_frames - crop_right
    cropped = sequence[crop_left:end] if end - crop_left >= 8 else sequence
    augmented = resample_sequence(cropped, n_frames)

    pts = augmented.reshape(n_frames, N_LANDMARKS, 3).copy()
    valid = np.any(np.abs(pts) > 1e-8, axis=2)

    angle = np.deg2rad(rng.uniform(-5.0, 5.0))
    scale = float(rng.uniform(0.94, 1.06))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    x = pts[:, :, 0].copy()
    y = pts[:, :, 1].copy()
    pts[:, :, 0] = scale * (x * cos_a - y * sin_a)
    pts[:, :, 1] = scale * (x * sin_a + y * cos_a)
    pts[:, :, 2] *= scale

    jitter = rng.normal(0.0, 0.012, size=pts.shape).astype(np.float32)
    jitter[:, :, 2] *= 0.5
    pts[valid] += jitter[valid]
    pts[~valid] = 0.0

    if n_frames >= 12 and rng.random() < 0.45:
        drop_count = int(rng.integers(1, max(2, n_frames // 18)))
        candidates = np.arange(1, n_frames - 1)
        dropped = rng.choice(candidates, size=min(drop_count, len(candidates)), replace=False)
        for idx in dropped:
            pts[idx] = (pts[idx - 1] + pts[idx + 1]) / 2.0

    np.clip(pts, -6.0, 6.0, out=pts)
    return pts.reshape(n_frames, FEATURE_DIM).astype(np.float32)


class HolisticExtractor:
    def __init__(self, min_detection_confidence: float = 0.45, min_tracking_confidence: float = 0.45):
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def process_frame(self, frame_bgr: np.ndarray) -> FrameObservation:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)
        points, hands_present = _to_landmark_array(results)
        return FrameObservation(normalize_landmarks(points), hands_present)

    def extract_video(self, video_path: str | Path, target_len: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, dict]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total_frames // 240) if total_frames > 0 else 1
        frames: list[np.ndarray] = []
        hand_flags: list[bool] = []
        hand_frames = 0
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                obs = self.process_frame(frame)
                frames.append(obs.features)
                hand_flags.append(obs.hands_present)
                hand_frames += int(obs.hands_present)
            frame_index += 1
        cap.release()

        if len(frames) < 8:
            raise ValueError(f"Video {video_path} has too few readable frames ({len(frames)})")
        hand_ratio = hand_frames / len(frames)
        if hand_ratio < 0.10:
            raise ValueError(
                f"MediaPipe detected hands in only {hand_ratio:.1%} of sampled frames for {video_path}. "
                "Re-record with both hands/body clearly visible."
            )

        active_indices = np.flatnonzero(np.asarray(hand_flags, dtype=bool))
        margin = 4
        start = max(0, int(active_indices[0]) - margin)
        stop = min(len(frames), int(active_indices[-1]) + margin + 1)
        trimmed_frames = frames[start:stop]
        if len(trimmed_frames) < 8:
            trimmed_frames = frames

        raw = np.stack(trimmed_frames).astype(np.float32)
        return resample_sequence(raw, target_len), {
            "video": str(video_path),
            "sampled_frames": len(frames),
            "trimmed_frames": len(trimmed_frames),
            "hand_frame_ratio": hand_ratio,
        }
