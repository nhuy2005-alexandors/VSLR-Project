from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


N_POSE = 25
N_HAND = 21
N_LANDMARKS = N_POSE + N_HAND * 2
LANDMARK_FEATURE_DIM = N_LANDMARKS * 3
PRESENCE_DIM = 2
FEATURE_DIM = LANDMARK_FEATURE_DIM + PRESENCE_DIM
LEFT_PRESENCE_INDEX = LANDMARK_FEATURE_DIM
RIGHT_PRESENCE_INDEX = LANDMARK_FEATURE_DIM + 1
SEQUENCE_LENGTH = 60
MAX_BRIDGE_GAP_SECONDS = 0.20

# Version 3 adds explicit left/right presence channels and presence-aware bridge/resampling.  A v2
# checkpoint has the same-looking landmark coordinates but a different meaning for gaps, so it
# must not load.
FEATURES_VERSION = 3
FEATURE_CONTRACT = "holistic-landmarks-v3-presence-aware-group-local-z"

# Bump whenever augment_sequence's operations, ranges, probabilities, or composition change.
AUGMENTATION_VERSION = 2


class ClipExtractionError(ValueError):
    """An expected source-video/recording failure that may be isolated to one clip."""


def require_valid_video_fps(value: float, source: str | Path) -> float:
    """Reject missing/invalid FPS instead of silently switching timestamp semantics."""

    fps = float(value)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ClipExtractionError(
            f"Video {source} has invalid FPS ({fps!r}); cannot build timestamp contract"
        )
    return fps


@dataclass(frozen=True)
class FrameObservation:
    """One frame's features plus independent observation masks for the two hands."""

    features: np.ndarray
    left_hand_present: bool
    right_hand_present: bool

    @property
    def hands_present(self) -> bool:
        return self.left_hand_present or self.right_hand_present


def _to_landmark_array(results) -> tuple[np.ndarray, bool, bool]:
    pose = np.zeros((N_POSE, 3), dtype=np.float32)
    left = np.zeros((N_HAND, 3), dtype=np.float32)
    right = np.zeros((N_HAND, 3), dtype=np.float32)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for idx in range(min(N_POSE, len(landmarks))):
            lm = landmarks[idx]
            pose[idx] = (lm.x, lm.y, lm.z)

    left_present = bool(results.left_hand_landmarks)
    if left_present:
        landmarks = results.left_hand_landmarks.landmark
        for idx in range(min(N_HAND, len(landmarks))):
            lm = landmarks[idx]
            left[idx] = (lm.x, lm.y, lm.z)

    right_present = bool(results.right_hand_landmarks)
    if right_present:
        landmarks = results.right_hand_landmarks.landmark
        for idx in range(min(N_HAND, len(landmarks))):
            lm = landmarks[idx]
            right[idx] = (lm.x, lm.y, lm.z)

    return np.concatenate((pose, left, right), axis=0), left_present, right_present


def normalize_landmarks(
    points: np.ndarray,
    left_hand_present: bool | None = None,
    right_hand_present: bool | None = None,
) -> np.ndarray:
    """Normalize landmarks and append independent left/right presence bits.

    MediaPipe pose ``z`` and hand ``z`` have different local origins (hip and wrist respectively).
    We subtract the observed origin of each group (hip midpoint for pose, wrist landmark for each
    hand) before applying the shared shoulder-width scale. Thus z is group-local and is never
    compared as if pose and hand coordinates came from one physical camera frame.
    """

    raw = np.asarray(points, dtype=np.float32).reshape(-1)
    if raw.size not in (LANDMARK_FEATURE_DIM, FEATURE_DIM):
        raise ValueError(
            f"Expected {LANDMARK_FEATURE_DIM} landmark values (or {FEATURE_DIM} with presence), "
            f"got {raw.shape}"
        )
    points = raw[:LANDMARK_FEATURE_DIM].reshape(N_LANDMARKS, 3).copy()
    coordinate_valid = np.any(np.abs(points) > 1e-8, axis=1)
    if left_hand_present is None:
        left_hand_present = bool(np.any(coordinate_valid[N_POSE : N_POSE + N_HAND]))
    if right_hand_present is None:
        right_hand_present = bool(np.any(coordinate_valid[N_POSE + N_HAND :]))
    # A presence flag is authoritative. This also handles malformed partial hand observations such
    # as a non-zero wrist-z with all other hand coordinates zero: an explicitly missing group is
    # removed before it can affect body center, scale, or depth origins.
    valid = coordinate_valid.copy()
    if not left_hand_present:
        valid[N_POSE : N_POSE + N_HAND] = False
    if not right_hand_present:
        valid[N_POSE + N_HAND :] = False
    if not np.any(valid):
        normalized = np.zeros(LANDMARK_FEATURE_DIM, dtype=np.float32)
    else:
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
        normalized_points = np.zeros_like(points, dtype=np.float32)
        normalized_points[valid, 0] = (points[valid, 0] - center_xy[0]) / scale
        normalized_points[valid, 1] = (points[valid, 1] - center_xy[1]) / scale
        # Make depth explicitly group-local.  This keeps the useful within-pose/within-hand
        # relative depth while avoiding a false cross-group coordinate system.
        pose_origin_z = (
            float(np.mean(points[[23, 24], 2]))
            if valid[23] and valid[24]
            else (float(points[0, 2]) if valid[0] else 0.0)
        )
        left_origin_z = float(points[N_POSE, 2]) if valid[N_POSE] else 0.0
        right_origin_z = float(points[N_POSE + N_HAND, 2]) if valid[N_POSE + N_HAND] else 0.0
        pose_valid = valid[:N_POSE]
        left_valid = valid[N_POSE : N_POSE + N_HAND]
        right_valid = valid[N_POSE + N_HAND :]
        pose_indices = np.flatnonzero(pose_valid)
        left_indices = N_POSE + np.flatnonzero(left_valid)
        right_indices = N_POSE + N_HAND + np.flatnonzero(right_valid)
        normalized_points[pose_indices, 2] = (points[pose_indices, 2] - pose_origin_z) / scale
        normalized_points[left_indices, 2] = (points[left_indices, 2] - left_origin_z) / scale
        normalized_points[right_indices, 2] = (points[right_indices, 2] - right_origin_z) / scale
        np.clip(normalized_points, -6.0, 6.0, out=normalized_points)
        normalized = normalized_points.reshape(-1)

    if not left_hand_present:
        normalized[N_POSE * 3 : (N_POSE + N_HAND) * 3] = 0.0
    if not right_hand_present:
        normalized[(N_POSE + N_HAND) * 3 : LANDMARK_FEATURE_DIM] = 0.0
    presence = np.asarray([float(left_hand_present), float(right_hand_present)], dtype=np.float32)
    return np.concatenate((normalized, presence)).astype(np.float32, copy=False)


def _validate_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [frames, {FEATURE_DIM}], got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("sequence contains NaN or infinity")
    return sequence


def _presence_arrays(
    sequence: np.ndarray,
    left_hand_present: np.ndarray | list[bool] | None,
    right_hand_present: np.ndarray | list[bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = len(sequence)
    left = (
        np.asarray(sequence[:, LEFT_PRESENCE_INDEX] >= 0.5, dtype=bool)
        if left_hand_present is None
        else np.asarray(left_hand_present, dtype=bool)
    )
    right = (
        np.asarray(sequence[:, RIGHT_PRESENCE_INDEX] >= 0.5, dtype=bool)
        if right_hand_present is None
        else np.asarray(right_hand_present, dtype=bool)
    )
    if left.shape != (n_frames,) or right.shape != (n_frames,):
        raise ValueError(
            f"presence arrays must have shape {(n_frames,)}, got {left.shape} and {right.shape}"
        )
    return left.copy(), right.copy()


def _time_axis(timestamps: np.ndarray | list[float] | None, n_frames: int) -> np.ndarray:
    if timestamps is None:
        return np.arange(n_frames, dtype=np.float64)
    axis = np.asarray(timestamps, dtype=np.float64)
    if axis.shape != (n_frames,):
        raise ValueError(f"timestamps must have shape {(n_frames,)}, got {axis.shape}")
    if not np.isfinite(axis).all() or np.any(np.diff(axis) < 0.0):
        raise ValueError("timestamps must be finite and non-decreasing")
    return axis


def trim_active_frames(
    sequence: np.ndarray | list[np.ndarray],
    left_hand_present: np.ndarray | list[bool],
    right_hand_present: np.ndarray | list[bool],
    timestamps: np.ndarray | list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trim only no-hand padding at the two ends, preserving internal missing gaps."""

    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected frame matrix [n, {FEATURE_DIM}], got {values.shape}")
    left, right = _presence_arrays(values, left_hand_present, right_hand_present)
    times = _time_axis(timestamps, len(values))
    active = left | right
    if not np.any(active):
        return values[:0], left[:0], right[:0], times[:0]
    start = int(np.flatnonzero(active)[0])
    stop = int(np.flatnonzero(active)[-1]) + 1
    return values[start:stop], left[start:stop], right[start:stop], times[start:stop]


def bridge_missing_hand_gaps(
    sequence: np.ndarray,
    left_hand_present: np.ndarray | list[bool] | None,
    right_hand_present: np.ndarray | list[bool] | None,
    *,
    timestamps: np.ndarray | list[float] | None = None,
    max_gap_seconds: float = MAX_BRIDGE_GAP_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bridge only short bounded missing runs for each hand independently.

    A gap is eligible only when the same hand is observed immediately before and after it. Long or
    one-sided gaps remain zero/missing. Timestamp duration keeps this stable across file and camera
    frame rates.
    """

    sequence = _validate_sequence(sequence)
    if not np.isfinite(max_gap_seconds) or max_gap_seconds < 0.0:
        raise ValueError("max_gap_seconds must be finite and >= 0")
    left, right = _presence_arrays(sequence, left_hand_present, right_hand_present)
    times = _time_axis(timestamps, len(sequence))
    output = sequence.copy()
    # Presence is authoritative.  A malformed upstream frame/cache must not smuggle coordinates
    # through a false mask and later look like a real hand observation.
    output[~left, N_POSE * 3 : (N_POSE + N_HAND) * 3] = 0.0
    output[~right, (N_POSE + N_HAND) * 3 : LANDMARK_FEATURE_DIM] = 0.0

    def bridge(mask: np.ndarray, start_column: int) -> None:
        i = 0
        while i < len(mask):
            if mask[i]:
                i += 1
                continue
            gap_start = i
            while i < len(mask) and not mask[i]:
                i += 1
            gap_end = i - 1
            before = gap_start - 1
            after = i
            if before < 0 or after >= len(mask):
                continue
            elapsed = float(times[after] - times[before])
            if elapsed > max_gap_seconds + 1e-9:
                continue
            denom = elapsed if elapsed > 0.0 else float(after - before)
            if denom <= 0.0:
                continue
            positions = (times[gap_start : gap_end + 1] - times[before]) / denom
            positions = np.clip(positions, 0.0, 1.0).astype(np.float32)[:, None]
            before_values = output[before, start_column : start_column + N_HAND * 3]
            after_values = output[after, start_column : start_column + N_HAND * 3]
            output[gap_start : gap_end + 1, start_column : start_column + N_HAND * 3] = (
                before_values[None, :] * (1.0 - positions) + after_values[None, :] * positions
            )
            mask[gap_start : gap_end + 1] = True

    bridge(left, N_POSE * 3)
    bridge(right, (N_POSE + N_HAND) * 3)
    output[:, LEFT_PRESENCE_INDEX] = left.astype(np.float32)
    output[:, RIGHT_PRESENCE_INDEX] = right.astype(np.float32)
    return output, left, right


def _sample_at(sequence: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Linear interpolation of every feature column at fractional frame positions."""

    last = len(sequence) - 1
    positions = np.clip(np.asarray(positions, dtype=np.float64), 0.0, last)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, last)
    frac = (positions - lower).astype(np.float32)[:, None]
    return (sequence[lower] * (1.0 - frac) + sequence[upper] * frac).astype(np.float32)


def _sample_presence_aware(
    sequence: np.ndarray,
    positions: np.ndarray,
    left_hand_present: np.ndarray,
    right_hand_present: np.ndarray,
) -> np.ndarray:
    """Resample pose normally but never interpolate a hand across a missing observation."""

    sequence = _validate_sequence(sequence)
    if len(sequence) == 0:
        raise ValueError("Cannot resample an empty sequence")
    positions = np.clip(np.asarray(positions, dtype=np.float64), 0.0, len(sequence) - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(sequence) - 1)
    frac = (positions - lower).astype(np.float32)[:, None]
    output = np.zeros((len(positions), FEATURE_DIM), dtype=np.float32)
    output[:, : N_POSE * 3] = (
        sequence[lower, : N_POSE * 3] * (1.0 - frac)
        + sequence[upper, : N_POSE * 3] * frac
    )

    def sample_hand(mask: np.ndarray, start_column: int, presence_column: int) -> None:
        # ``floor(1.0)`` is 1 while ``upper`` is 2, even though the requested sample is exactly
        # frame 1.  At an exact integer position only that observed frame is authoritative; require
        # both masks only for a genuinely fractional sample between two frames.
        exact = np.abs(frac[:, 0]) <= 1e-7
        valid = np.where(exact, mask[lower], mask[lower] & mask[upper])
        values = (
            sequence[lower, start_column : start_column + N_HAND * 3] * (1.0 - frac)
            + sequence[upper, start_column : start_column + N_HAND * 3] * frac
        )
        output[:, start_column : start_column + N_HAND * 3] = np.where(valid[:, None], values, 0.0)
        output[:, presence_column] = valid.astype(np.float32)

    sample_hand(left_hand_present, N_POSE * 3, LEFT_PRESENCE_INDEX)
    sample_hand(right_hand_present, (N_POSE + N_HAND) * 3, RIGHT_PRESENCE_INDEX)
    return output


def resample_sequence(
    sequence: np.ndarray,
    target_len: int = SEQUENCE_LENGTH,
    *,
    left_hand_present: np.ndarray | list[bool] | None = None,
    right_hand_present: np.ndarray | list[bool] | None = None,
) -> np.ndarray:
    sequence = _validate_sequence(sequence)
    if len(sequence) == 0:
        raise ValueError("Cannot resample an empty sequence")
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    left, right = _presence_arrays(sequence, left_hand_present, right_hand_present)
    if len(sequence) == 1:
        output = np.repeat(sequence, target_len, axis=0)
        output[:, LEFT_PRESENCE_INDEX] = left[0]
        output[:, RIGHT_PRESENCE_INDEX] = right[0]
        return output
    return _sample_presence_aware(
        sequence,
        np.linspace(0.0, len(sequence) - 1, target_len, dtype=np.float64),
        left,
        right,
    )


def preprocess_sequence(
    sequence: np.ndarray,
    *,
    left_hand_present: np.ndarray | list[bool] | None = None,
    right_hand_present: np.ndarray | list[bool] | None = None,
    timestamps: np.ndarray | list[float] | None = None,
    target_len: int = SEQUENCE_LENGTH,
    max_gap_seconds: float = MAX_BRIDGE_GAP_SECONDS,
) -> np.ndarray:
    """The single preprocessing path used by video extraction and realtime classification."""

    sequence = _validate_sequence(sequence)
    left, right = _presence_arrays(sequence, left_hand_present, right_hand_present)
    bridged, left, right = bridge_missing_hand_gaps(
        sequence,
        left,
        right,
        timestamps=timestamps,
        max_gap_seconds=max_gap_seconds,
    )
    return resample_sequence(bridged, target_len, left_hand_present=left, right_hand_present=right)


def _warp_positions(n_frames: int, rng: np.random.Generator, max_warp: float = 0.30) -> np.ndarray:
    """A monotone reparameterisation of [0, 1] sampled on `n_frames` points."""

    if not 0.0 <= max_warp < 1.0 / np.pi:
        raise ValueError(f"max_warp must be in [0, 1/pi); got {max_warp}")
    warp = float(rng.uniform(-max_warp, max_warp))
    normalized = np.linspace(0.0, 1.0, n_frames, dtype=np.float64)
    return normalized + warp * np.sin(np.pi * normalized)


def time_warp_sequence(sequence: np.ndarray, rng: np.random.Generator, max_warp: float = 0.30) -> np.ndarray:
    """Bend internal timing while preserving presence-aware missing groups."""

    sequence = _validate_sequence(sequence)
    if len(sequence) < 3:
        return sequence.copy()
    left, right = _presence_arrays(sequence, None, None)
    return _sample_presence_aware(
        sequence,
        _warp_positions(len(sequence), rng, max_warp) * (len(sequence) - 1),
        left,
        right,
    )


def augment_sequence(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild landmark augmentation that preserves missing-hand masks."""

    sequence = _validate_sequence(sequence)
    n_frames = len(sequence)
    left, right = _presence_arrays(sequence, None, None)
    max_crop = max(1, int(round(n_frames * 0.08)))
    crop_left = int(rng.integers(0, max_crop + 1))
    crop_right = int(rng.integers(0, max_crop + 1))
    end = n_frames - crop_right
    if end - crop_left >= 8:
        cropped = sequence[crop_left:end]
        cropped_left, cropped_right = left[crop_left:end], right[crop_left:end]
    else:
        cropped = sequence
        cropped_left, cropped_right = left, right
    if len(cropped) < 3:
        augmented = resample_sequence(
            cropped,
            n_frames,
            left_hand_present=cropped_left,
            right_hand_present=cropped_right,
        )
    else:
        augmented = _sample_presence_aware(
            cropped,
            _warp_positions(n_frames, rng) * (len(cropped) - 1),
            cropped_left,
            cropped_right,
        )

    points = augmented[:, :LANDMARK_FEATURE_DIM].reshape(n_frames, N_LANDMARKS, 3).copy()
    augmented_left = augmented[:, LEFT_PRESENCE_INDEX] >= 0.5
    augmented_right = augmented[:, RIGHT_PRESENCE_INDEX] >= 0.5
    angle = np.deg2rad(rng.uniform(-5.0, 5.0))
    scale = float(rng.uniform(0.94, 1.06))
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    x = points[:, :, 0].copy()
    y = points[:, :, 1].copy()
    points[:, :, 0] = scale * (x * cos_a - y * sin_a)
    points[:, :, 1] = scale * (x * sin_a + y * cos_a)
    points[:, :, 2] *= scale

    valid = np.any(np.abs(points) > 1e-8, axis=2)
    jitter = rng.normal(0.0, 0.012, size=points.shape).astype(np.float32)
    jitter[:, :, 2] *= 0.5
    points[valid] += jitter[valid]
    points[~valid] = 0.0

    if n_frames >= 12 and rng.random() < 0.45:
        drop_count = int(rng.integers(1, max(2, n_frames // 18)))
        candidates = np.arange(1, n_frames - 1)
        dropped = rng.choice(candidates, size=min(drop_count, len(candidates)), replace=False)
        for idx in dropped:
            points[idx, :N_POSE] = (points[idx - 1, :N_POSE] + points[idx + 1, :N_POSE]) / 2.0
            if augmented_left[idx - 1] and augmented_left[idx + 1]:
                points[idx, N_POSE : N_POSE + N_HAND] = (
                    points[idx - 1, N_POSE : N_POSE + N_HAND]
                    + points[idx + 1, N_POSE : N_POSE + N_HAND]
                ) / 2.0
            else:
                points[idx, N_POSE : N_POSE + N_HAND] = 0.0
                augmented_left[idx] = False
            if augmented_right[idx - 1] and augmented_right[idx + 1]:
                points[idx, N_POSE + N_HAND :] = (
                    points[idx - 1, N_POSE + N_HAND :]
                    + points[idx + 1, N_POSE + N_HAND :]
                ) / 2.0
            else:
                points[idx, N_POSE + N_HAND :] = 0.0
                augmented_right[idx] = False

    np.clip(points, -6.0, 6.0, out=points)
    output = np.zeros_like(augmented)
    output[:, :LANDMARK_FEATURE_DIM] = points.reshape(n_frames, LANDMARK_FEATURE_DIM)
    output[:, LEFT_PRESENCE_INDEX] = augmented_left.astype(np.float32)
    output[:, RIGHT_PRESENCE_INDEX] = augmented_right.astype(np.float32)
    return output.astype(np.float32)


class HolisticExtractor:
    def __init__(self, min_detection_confidence: float = 0.45, min_tracking_confidence: float = 0.45):
        self._config = dict(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._live = mp.solutions.holistic.Holistic(**self._config)

    def close(self) -> None:
        self._live.close()

    def __enter__(self) -> "HolisticExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _observe(self, frame_bgr: np.ndarray, holistic) -> FrameObservation:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        points, left_present, right_present = _to_landmark_array(results)
        return FrameObservation(
            normalize_landmarks(points, left_present, right_present), left_present, right_present
        )

    def process_frame(self, frame_bgr: np.ndarray) -> FrameObservation:
        return self._observe(frame_bgr, self._live)

    def extract_video(self, video_path: str | Path, target_len: int = SEQUENCE_LENGTH) -> tuple[np.ndarray, dict]:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ClipExtractionError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            fps = require_valid_video_fps(cap.get(cv2.CAP_PROP_FPS), video_path)
        except ClipExtractionError:
            cap.release()
            raise
        stride = max(1, total_frames // 240) if total_frames > 0 else 1
        frames: list[np.ndarray] = []
        left_flags: list[bool] = []
        right_flags: list[bool] = []
        frame_times: list[float] = []
        hand_frames = 0
        frame_index = 0

        # A fresh instance per video avoids a previous clip's temporal tracking state contaminating
        # this clip; the live camera path intentionally keeps its one instance across frames.
        with mp.solutions.holistic.Holistic(**self._config) as holistic:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    obs = self._observe(frame, holistic)
                    frames.append(obs.features)
                    left_flags.append(obs.left_hand_present)
                    right_flags.append(obs.right_hand_present)
                    frame_times.append(frame_index / fps)
                    hand_frames += int(obs.hands_present)
                frame_index += 1
        cap.release()

        if len(frames) < 8:
            raise ClipExtractionError(f"Video {video_path} has too few readable frames ({len(frames)})")
        hand_ratio = hand_frames / len(frames)
        if hand_ratio < 0.10:
            raise ClipExtractionError(
                f"MediaPipe detected hands in only {hand_ratio:.1%} of sampled frames for {video_path}. "
                "Re-record with both hands/body clearly visible."
            )

        trimmed_frames, trimmed_left, trimmed_right, trimmed_times = trim_active_frames(
            np.stack(frames).astype(np.float32), left_flags, right_flags, frame_times
        )
        if len(trimmed_frames) < 8:
            trimmed_frames = np.stack(frames).astype(np.float32)
            trimmed_left = np.asarray(left_flags, dtype=bool)
            trimmed_right = np.asarray(right_flags, dtype=bool)
            trimmed_times = np.asarray(frame_times, dtype=np.float64)

        sequence = preprocess_sequence(
            np.asarray(trimmed_frames, dtype=np.float32),
            left_hand_present=trimmed_left,
            right_hand_present=trimmed_right,
            timestamps=trimmed_times,
            target_len=target_len,
        )
        return sequence, {
            "video": str(video_path),
            "sampled_frames": len(frames),
            "trimmed_frames": len(trimmed_frames),
            "hand_frame_ratio": hand_ratio,
            "left_hand_frame_ratio": float(np.mean(left_flags)),
            "right_hand_frame_ratio": float(np.mean(right_flags)),
            "fps": fps,
        }
