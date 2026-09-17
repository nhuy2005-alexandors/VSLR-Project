from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .vsl3.console import configure_utf8_stdio
from .vsl3.features import (
    FEATURE_DIM,
    LANDMARK_FEATURE_DIM,
    LEFT_PRESENCE_INDEX,
    MAX_BRIDGE_GAP_SECONDS,
    N_HAND,
    N_POSE,
    RIGHT_PRESENCE_INDEX,
    HolisticExtractor,
    preprocess_sequence,
)
from .vsl3.model import load_checkpoint, predict_sequence
from .vsl3.reject import (
    RejectPolicy,
    load_reject_policy,
    reject_prediction,
    sha256_file,
)
from .tts import TTSManager, speak_text as tts_speak_text
try:
    from .recorder import GestureVideoRecorder, draw_unicode_text
except ImportError:
    from prototype_3_gestures.recorder import GestureVideoRecorder, draw_unicode_text
import mediapipe as mp

configure_utf8_stdio()


GESTURE_ACTIVITY_WRIST_ABOVE_HIP = 0.5
GESTURE_PRE_ROLL_SECONDS = 1.0
GESTURE_POST_ROLL_SECONDS = 0.5


def gesture_activity_from_features(
    features: np.ndarray,
    left_hand_present: bool,
    right_hand_present: bool,
    *,
    wrist_above_hip: float = GESTURE_ACTIVITY_WRIST_ABOVE_HIP,
) -> bool:
    """Distinguish a signing hand from a visible hand resting beside the body."""

    values = np.asarray(features, dtype=np.float32).reshape(-1)
    if values.size < LANDMARK_FEATURE_DIM:
        return bool(left_hand_present or right_hand_present)
    points = values[:LANDMARK_FEATURE_DIM].reshape(-1, 3)

    def valid(index: int) -> bool:
        return bool(np.any(np.abs(points[index]) > 1e-8))

    hip_y = [float(points[index, 1]) for index in (23, 24) if valid(index)]
    scores: list[float] = []
    if hip_y:
        hip_line = float(np.mean(hip_y))
        for present, hand_wrist_index in (
            (left_hand_present, N_POSE),
            (right_hand_present, N_POSE + N_HAND),
        ):
            if present and valid(hand_wrist_index):
                scores.append(hip_line - float(points[hand_wrist_index, 1]))

    if not scores:
        # A cropped camera may not contain the hips. Preserve the old presence behavior instead of
        # silently discarding every gesture when the body-relative gate cannot be evaluated.
        return bool(left_hand_present or right_hand_present)
    return max(scores) >= wrist_above_hip


def speak_text(text: str) -> None:
    tts_speak_text(text)


@dataclass(frozen=True)
class Segment:
    """One completed gesture and its independent per-frame presence/timestamp tracks."""

    features: list[np.ndarray]
    start_time: float
    active_end_time: float
    end_time: float
    forced: bool
    timestamps: list[float] = field(default_factory=list)
    left_hand_present: list[bool] = field(default_factory=list)
    right_hand_present: list[bool] = field(default_factory=list)
    gesture_active: list[bool] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Time from first to last hand detection; excludes the trailing no-hand word gap."""

        return max(0.0, self.active_end_time - self.start_time)

    def active_frames(self) -> tuple[list[np.ndarray], list[bool], list[bool], list[float]]:
        """Return only frames with at least one observed hand for model input."""

        n_frames = len(self.features)
        if not n_frames:
            return [], [], [], []
        if len(self.left_hand_present) == n_frames and len(self.right_hand_present) == n_frames:
            left = list(self.left_hand_present)
            right = list(self.right_hand_present)
        elif all(np.asarray(frame).reshape(-1).size >= FEATURE_DIM for frame in self.features):
            stacked = np.asarray(self.features, dtype=np.float32)
            left = (stacked[:, LEFT_PRESENCE_INDEX] >= 0.5).tolist()
            right = (stacked[:, RIGHT_PRESENCE_INDEX] >= 0.5).tolist()
        else:
            # Compatibility for callers that supplied old 3-value synthetic frames: these are
            # considered active because there is no independent mask to consult.
            left = [True] * n_frames
            right = [False] * n_frames
            times = list(self.timestamps) if len(self.timestamps) == n_frames else list(range(n_frames))
            return list(self.features), left, right, times
        has_timestamps = len(self.timestamps) == n_frames
        times = list(self.timestamps) if has_timestamps else list(range(n_frames))
        has_explicit_activity = len(self.gesture_active) == n_frames
        activity = (
            list(self.gesture_active)
            if has_explicit_activity
            else [left[index] or right[index] for index in range(n_frames)]
        )
        if not any(activity):
            return [], [], [], []
        # Trim resting frames using the activity gate while preserving the real hand-presence masks
        # inside the gesture. Short pauses and MediaPipe gaps remain meaningful model input.
        active_indices = [index for index, is_active in enumerate(activity) if is_active]
        start = active_indices[0]
        stop = active_indices[-1] + 1
        if has_timestamps and has_explicit_activity:
            context_start = times[start] - GESTURE_PRE_ROLL_SECONDS
            context_stop = times[stop - 1] + GESTURE_POST_ROLL_SECONDS
            epsilon = 1e-9
            while start > 0 and times[start - 1] + epsilon >= context_start:
                start -= 1
            while (
                stop < n_frames
                and times[stop] <= context_stop + epsilon
                and (left[stop] or right[stop])
            ):
                stop += 1
        return (
            list(self.features[start:stop]),
            left[start:stop],
            right[start:stop],
            times[start:stop],
        )


def classify_segment_frames(
    segment: Segment,
) -> tuple[list[np.ndarray], list[bool], list[bool], list[float]]:
    """Expose the exact active-frame selection used before model preprocessing."""

    return segment.active_frames()


def _classify_segment(
    model,
    labels,
    device,
    segment: Segment | list[np.ndarray],
    seq_len: int,
) -> tuple[str, float, np.ndarray]:
    if isinstance(segment, Segment):
        active, left, right, timestamps = segment.active_frames()
    else:
        active = list(segment)
        if not active:
            raise ValueError("cannot classify an empty segment")
        stacked = np.asarray(active, dtype=np.float32)
        if stacked.ndim != 2 or stacked.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected segment frames [n, {FEATURE_DIM}], got {stacked.shape}")
        left = (stacked[:, LEFT_PRESENCE_INDEX] >= 0.5).tolist()
        right = (stacked[:, RIGHT_PRESENCE_INDEX] >= 0.5).tolist()
        if not any(left) and not any(right):
            # A direct legacy call has no mask; preserving all frames is safer than making a
            # surprising empty prediction. SegmentTracker always supplies masks in the live path.
            left = [True] * len(active)
        timestamps = list(range(len(active)))
    if not active:
        raise ValueError("cannot classify a segment containing no hand observation")
    sequence = preprocess_sequence(
        np.stack(active).astype(np.float32),
        left_hand_present=left,
        right_hand_present=right,
        timestamps=timestamps,
        target_len=seq_len,
        max_gap_seconds=MAX_BRIDGE_GAP_SECONDS,
    )
    label, confidence, probabilities = predict_sequence(model, sequence, labels, device)
    return label, confidence, probabilities


def classify_segment(
    model,
    labels,
    device,
    segment: Segment | list[np.ndarray],
    seq_len: int,
) -> tuple[str, float]:
    label, confidence, _ = _classify_segment(model, labels, device, segment, seq_len)
    return label, confidence


def should_accept_prediction(confidence: float, threshold: float) -> bool:
    return confidence >= threshold


@dataclass(frozen=True)
class SegmentDecision:
    label: str
    confidence: float
    accepted: bool
    reason: str
    probabilities: np.ndarray | None = None


def decide_segment(
    model,
    labels,
    device,
    segment: Segment,
    seq_len: int,
    *,
    confidence_threshold: float,
    reject_policy: RejectPolicy,
    allow_uncalibrated: bool = False,
) -> SegmentDecision:
    label, confidence, probabilities = _classify_segment(model, labels, device, segment, seq_len)
    accepted, reason = reject_prediction(
        confidence,
        probabilities,
        reject_policy,
        manual_threshold=confidence_threshold,
        allow_uncalibrated=allow_uncalibrated,
    )
    return SegmentDecision(label, confidence, accepted, reason, probabilities=probabilities)


class SegmentTracker:
    """Testable boundary state machine for one continuous frame stream."""

    def __init__(
        self,
        word_gap: float,
        max_seconds: float,
        min_active_seconds: float = 0.0,
    ):
        if not math.isfinite(word_gap) or word_gap <= 0.0:
            raise ValueError("word_gap must be finite and > 0")
        if not math.isfinite(max_seconds) or max_seconds <= 0.0:
            raise ValueError("max_seconds must be finite and > 0")
        if not math.isfinite(min_active_seconds) or min_active_seconds < 0.0:
            raise ValueError("min_active_seconds must be finite and >= 0")
        self.word_gap = word_gap
        self.max_seconds = max_seconds
        self.min_active_seconds = min_active_seconds
        self.segment: list[np.ndarray] = []
        self.segment_times: list[float] = []
        self.segment_left: list[bool] = []
        self.segment_right: list[bool] = []
        self.segment_activity: list[bool] = []
        self.pending: list[tuple[np.ndarray, float, bool, bool]] = []
        self.context_buffer: list[tuple[np.ndarray, float, bool, bool]] = []
        self.pending_start_time = 0.0
        self.last_timestamp: float | None = None
        self.in_segment = False
        self.start_time = 0.0
        self.last_hand_time = 0.0
        self.awaiting_hand_drop = False

    def feed(
        self,
        hands_present: bool,
        features: np.ndarray,
        now: float,
        left_hand_present: bool | None = None,
        right_hand_present: bool | None = None,
        *,
        gesture_active: bool | None = None,
    ) -> Segment | None:
        if not math.isfinite(now):
            raise ValueError(f"frame timestamp must be finite, got {now}")
        if self.last_timestamp is not None and now < self.last_timestamp:
            raise ValueError(
                f"frame timestamps must be non-decreasing: previous={self.last_timestamp}, now={now}"
            )
        self.last_timestamp = now
        values = np.asarray(features, dtype=np.float32).copy()
        if values.ndim != 1:
            raise ValueError(f"frame features must be one-dimensional, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("frame features contain NaN or infinity")
        if values.size >= FEATURE_DIM:
            raw_presence = values[[LEFT_PRESENCE_INDEX, RIGHT_PRESENCE_INDEX]]
            if not np.all(np.isclose(raw_presence, 0.0) | np.isclose(raw_presence, 1.0)):
                raise ValueError("presence channels must be finite binary values")
        if left_hand_present is None or right_hand_present is None:
            if values.size >= FEATURE_DIM:
                left = bool(values[LEFT_PRESENCE_INDEX] >= 0.5)
                right = bool(values[RIGHT_PRESENCE_INDEX] >= 0.5)
            else:
                left, right = bool(hands_present), False
            if left_hand_present is None:
                left_hand_present = left
            if right_hand_present is None:
                right_hand_present = right
        left = bool(left_hand_present)
        right = bool(right_hand_present)
        if bool(hands_present) != (left or right):
            raise ValueError(
                "hands_present does not match authoritative left/right presence masks"
            )
        if values.size >= FEATURE_DIM:
            values[LEFT_PRESENCE_INDEX] = float(left)
            values[RIGHT_PRESENCE_INDEX] = float(right)
        detected = left or right
        active = detected if gesture_active is None else bool(gesture_active)
        if active and not detected:
            raise ValueError("gesture activity requires at least one observed hand")

        if active:
            self.last_hand_time = now
            if self.awaiting_hand_drop:
                return None
            promoted = False
            if not self.in_segment:
                if self.min_active_seconds > 0.0:
                    if not self.pending:
                        self.pending_start_time = now
                    self.pending.append((values, now, left, right))
                    if now - self.pending_start_time < self.min_active_seconds:
                        return None
                    pending = self.pending
                    self.pending = []
                    self._begin_segment(pending[0][1])
                    for pending_values, pending_time, pending_left, pending_right in pending:
                        completed = self._append(
                            pending_values,
                            pending_time,
                            hands_up=True,
                            left_hand_present=pending_left,
                            right_hand_present=pending_right,
                            gesture_active=True,
                        )
                        if completed is not None:
                            return completed
                    promoted = True
                else:
                    self._begin_segment(now)
            if self.awaiting_hand_drop:
                return None
            if promoted:
                return None
            return self._append(
                values,
                now,
                hands_up=True,
                left_hand_present=left,
                right_hand_present=right,
                gesture_active=True,
            )

        if self.pending:
            # Pending evidence was a blip; discard it without ever exposing a segment.
            self.pending = []
            self._remember_context(values, now, left, right)
            return None
        if self.awaiting_hand_drop:
            if now - self.last_hand_time >= self.word_gap:
                self.awaiting_hand_drop = False
                self._remember_context(values, now, left, right)
            return None

        if not self.in_segment:
            self._remember_context(values, now, left, right)
            return None
        if now - self.last_hand_time < self.word_gap:
            # Keep boundary timing for the tracker, but classify_segment() removes these frames
            # using the explicit masks. Thus a no-hand tail can never enter the model.
            return self._append(
                values,
                now,
                hands_up=False,
                left_hand_present=left,
                right_hand_present=right,
                gesture_active=False,
            )
        return self._close(now, forced=False)

    def force_boundary(self, now: float) -> Segment | None:
        if not math.isfinite(now):
            raise ValueError(f"frame timestamp must be finite, got {now}")
        if self.last_timestamp is not None and now < self.last_timestamp:
            raise ValueError(
                f"frame timestamps must be non-decreasing: previous={self.last_timestamp}, now={now}"
            )
        self.last_timestamp = now
        self.pending = []
        return self._close(now, forced=False) if self.in_segment else None

    def reset(self) -> None:
        self.segment = []
        self.segment_times = []
        self.segment_left = []
        self.segment_right = []
        self.segment_activity = []
        self.pending = []
        self.context_buffer = []
        self.in_segment = False
        self.awaiting_hand_drop = False
        self.last_timestamp = None

    def _remember_context(
        self,
        features: np.ndarray,
        now: float,
        left_hand_present: bool,
        right_hand_present: bool,
    ) -> None:
        if not (left_hand_present or right_hand_present):
            self.context_buffer = []
            return
        self.context_buffer.append((features.copy(), now, left_hand_present, right_hand_present))
        cutoff = now - GESTURE_PRE_ROLL_SECONDS
        self.context_buffer = [item for item in self.context_buffer if item[1] >= cutoff]

    def _begin_segment(self, active_start_time: float) -> None:
        context = self.context_buffer
        self.context_buffer = []
        self.segment = [item[0] for item in context]
        self.segment_times = [item[1] for item in context]
        self.segment_left = [item[2] for item in context]
        self.segment_right = [item[3] for item in context]
        self.segment_activity = [False] * len(context)
        self.in_segment = True
        self.start_time = active_start_time

    def _append(
        self,
        features: np.ndarray,
        now: float,
        *,
        hands_up: bool,
        left_hand_present: bool,
        right_hand_present: bool,
        gesture_active: bool,
    ) -> Segment | None:
        self.segment.append(features)
        self.segment_times.append(now)
        self.segment_left.append(left_hand_present)
        self.segment_right.append(right_hand_present)
        self.segment_activity.append(gesture_active)
        if now - self.start_time >= self.max_seconds:
            # Once a segment is closed by the hard duration cap, the next active frame may still
            # belong to the same gesture even if the cap was crossed during a no-hand tail. Wait
            # for a full word_gap of genuine no-hand time in every max-boundary path.
            self.awaiting_hand_drop = True
            return self._close(now, forced=hands_up)
        return None

    def _close(self, now: float, forced: bool) -> Segment:
        done, self.segment, self.in_segment = self.segment, [], False
        times, self.segment_times = self.segment_times, []
        left, self.segment_left = self.segment_left, []
        right, self.segment_right = self.segment_right, []
        activity, self.segment_activity = self.segment_activity, []
        return Segment(
            features=done,
            start_time=self.start_time,
            active_end_time=self.last_hand_time,
            end_time=now,
            forced=forced,
            timestamps=times,
            left_hand_present=left,
            right_hand_present=right,
            gesture_active=activity,
        )


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not math.isfinite(args.confidence) or not 0.0 < args.confidence <= 1.0:
        parser.error(f"--confidence must be in (0, 1], got {args.confidence}")
    for name, value in (("--word-gap", args.word_gap), ("--sentence-gap", args.sentence_gap)):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and > 0, got {value}")
    if not math.isfinite(args.cooldown) or args.cooldown < 0.0:
        parser.error(f"--cooldown must be finite and >= 0, got {args.cooldown}")
    if not math.isfinite(args.min_seconds) or args.min_seconds <= 0.0:
        parser.error(f"--min-seconds must be finite and > 0, got {args.min_seconds}")
    if not math.isfinite(args.max_seconds) or args.max_seconds < args.min_seconds:
        parser.error(
            f"--max-seconds ({args.max_seconds}) must be finite and >= "
            f"--min-seconds ({args.min_seconds})"
        )


def list_available_cameras() -> list[str]:
    """List available DirectShow camera devices on Windows."""
    try:
        import pygrabber.dshow_graph

        graph = pygrabber.dshow_graph.FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []


def resolve_camera(camera_arg: str | int) -> tuple[int, str]:
    """Resolve camera index and human-readable name from integer or name string."""
    devices = list_available_cameras()
    arg_str = str(camera_arg).strip()

    try:
        idx = int(arg_str)
        if idx >= 0:
            name = devices[idx] if 0 <= idx < len(devices) else f"Camera #{idx}"
            return idx, name
    except ValueError:
        pass

    query = arg_str.lower()
    for idx, dev_name in enumerate(devices):
        if query in dev_name.lower():
            return idx, dev_name

    available = ", ".join(f"[{i}] {name}" for i, name in enumerate(devices)) if devices else "None detected"
    raise ValueError(f"Cannot find camera matching '{camera_arg}'. Available cameras: {available}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime VSL gesture -> sentence -> TTS demo")
    parser.add_argument("--model", default="models/gesture_lstm.pt")
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index (0, 1) hoặc tên camera (ví dụ: 'A16' hoặc 'Webcam')",
    )
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--reject-policy", help="Calibrated reject-policy JSON; absent means uncalibrated")
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Allow manual --confidence threshold before negative calibration (never claim OOD safety)",
    )
    parser.add_argument(
        "--word-gap",
        type=float,
        default=0.45,
        help="Inactive/resting-hand gap that ends one gesture",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="Duplicate phrase suppression cooldown in seconds (0 to disable)",
    )
    parser.add_argument("--sentence-gap", type=float, default=2.2, help="Additional idle time before speaking")
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.35,
        help="Minimum active evidence/duration for a gesture, in seconds",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=5.0,
        help="Hard cap on one gesture; the rest is discarded until hands come down",
    )
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument(
        "--tts-engine",
        default="vieneu",
        choices=["vieneu", "pyttsx3", "system", "none"],
        help="Bộ phát âm TTS tiếng Việt (mặc định: vieneu)",
    )
    parser.add_argument(
        "--tts-voice",
        default="Trúc Ly",
        help="Giọng đọc VieNeu-TTS (mặc định: 'Trúc Ly', ví dụ: 'Mai Anh', 'Minh Quân', 'Phạm Tuyên')",
    )
    parser.add_argument(
        "--record-dir",
        default=None,
        help="Thư mục lưu video cử chỉ (mặc định: videos/)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Tắt tính năng tự động quay video cử chỉ",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=30.0,
        help="Tốc độ khung hình (FPS) dự phòng cho video quay cử chỉ (mặc định: 30.0)",
    )
    parser.add_argument(
        "--record-mode",
        default="both",
        choices=["both", "skeleton", "raw"],
        help="Chế độ quay video cử chỉ: 'both' (cả bản khung xương & bản gốc), 'skeleton' (chỉ bản khung xương), 'raw' (chỉ bản gốc)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    if not model_path.is_file():
        repo_root = Path(__file__).resolve().parent.parent.parent
        if (repo_root / args.model).is_file():
            model_path = repo_root / args.model
        elif (Path.cwd() / "VSLR-Project-pipeline-signer-split" / args.model).is_file():
            model_path = Path.cwd() / "VSLR-Project-pipeline-signer-split" / args.model

    model, labels, config = load_checkpoint(model_path, device)
    try:
        checkpoint_sha256 = sha256_file(model_path)
        policy = load_reject_policy(
            args.reject_policy or config.get("reject_policy"),
            expected_feature_contract=config.get("feature_contract"),
            expected_training_signature=config.get("training_signature"),
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_rejection_contract=config.get("rejection_contract"),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: invalid reject policy: {exc}") from exc
    if not policy.calibrated and not args.allow_uncalibrated:
        print(
            "REJECT POLICY: uncalibrated — every segment will be rejected until negative/idle/OOV "
            "calibration is supplied. Use --allow-uncalibrated only for an explicitly provisional demo."
        )

    seq_len = int(config["sequence_length"])
    print(f"Loaded labels: {labels}")
    print("Controls: Q quit | C clear sentence | S speak now | SPACE force current gesture boundary")
    print("Lower hands between gestures; rejected/unknown segments are never sent to TTS.")

    cams = list_available_cameras()
    if cams:
        print("Cameras phát hiện được trên máy:")
        for i, cname in enumerate(cams):
            print(f"  [{i}] {cname}")
    try:
        cam_idx, cam_name = resolve_camera(args.camera)
        print(f"Khởi động camera: [{cam_idx}] {cam_name}")
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera [{cam_idx}] {cam_name}")

    tts_enabled = not args.no_tts and args.tts_engine != "none"
    tts_mgr = TTSManager(engine=args.tts_engine, voice=args.tts_voice, enabled=tts_enabled)

    recorder = GestureVideoRecorder(
        record_dir=args.record_dir,
        fps=args.record_fps,
        enabled=not args.no_record,
        record_mode=args.record_mode,
    )
    if recorder.enabled:
        mode_desc = {
            "both": "cả 2 bản: khung xương & gốc",
            "skeleton": "chỉ bản khung xương",
            "raw": "chỉ bản gốc",
        }.get(recorder.record_mode, recorder.record_mode)
        print(f"Quay video cử chỉ: BẬT ({mode_desc}) -> Thư mục lưu: {recorder.output_dir}")
    else:
        print("Quay video cử chỉ: TẮT (--no-record)")

    sentence: list[str] = []
    tracker = SegmentTracker(args.word_gap, args.max_seconds, min_active_seconds=args.min_seconds)
    last_sentence_activity = time.monotonic()
    last_accepted_label: str | None = None
    last_accepted_time: float = 0.0
    last_saved_info: str | None = None
    last_saved_time: float = 0.0

    def handle_segment(segment: Segment | None) -> None:
        nonlocal last_sentence_activity, last_accepted_label, last_accepted_time, last_saved_info, last_saved_time
        if segment is None:
            return
        if segment.duration < args.min_seconds:
            recorder.cancel_gesture()
            return
        decision = decide_segment(
            model,
            labels,
            device,
            segment,
            seq_len,
            confidence_threshold=args.confidence,
            reject_policy=policy,
            allow_uncalibrated=args.allow_uncalibrated,
        )
        if recorder.enabled:
            recorder.save_gesture(decision, labels=labels, duration=segment.duration)
            status_desc = "ACCEPTED" if decision.accepted else "REJECTED"
            last_saved_info = f"Da luu video: {decision.label} [{status_desc}]"
            last_saved_time = time.monotonic()

        now_seg = time.monotonic()
        if decision.accepted:
            if (
                args.cooldown > 0
                and decision.label == last_accepted_label
                and (now_seg - last_accepted_time) < args.cooldown
            ):
                print(
                    f"COOLDOWN> Suppressed duplicate {decision.label} within {args.cooldown:.1f}s"
                )
                return
            last_accepted_label = decision.label
            last_accepted_time = now_seg
            sentence.append(decision.label)
            print(
                f"WORD> {decision.label} ({decision.confidence:.1%}) | sentence: {' '.join(sentence)}"
            )
            last_sentence_activity = now_seg
        else:
            last_accepted_label = None
            print(f"REJECT> {decision.label} ({decision.confidence:.1%}) — {decision.reason}")

    def speak_sentence(now: float) -> None:
        nonlocal last_sentence_activity
        if not sentence:
            return
        text = " ".join(sentence)
        if args.no_tts or args.tts_engine == "none":
            print(f"SENTENCE> {text}")
        else:
            tts_mgr.speak(text)
        sentence.clear()
        last_sentence_activity = now

    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        with HolisticExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                obs = extractor.process_frame(frame)
                gesture_active = gesture_activity_from_features(
                    obs.features, obs.left_hand_present, obs.right_hand_present
                )
                done = tracker.feed(
                    obs.hands_present,
                    obs.features,
                    now,
                    obs.left_hand_present,
                    obs.right_hand_present,
                    gesture_active=gesture_active,
                )
                # Frame này thuộc cử chỉ nếu đang trong cử chỉ HOẶC cử chỉ vừa kết thúc tại frame này
                in_gesture = tracker.in_segment or (done is not None)
                recorder.feed_frame(frame, obs, in_segment=in_gesture, now=now)

                if done is not None:
                    if done.forced:
                        print(
                            f"Segment hit --max-seconds ({args.max_seconds}); classifying and waiting for resting hands."
                        )
                    handle_segment(done)

                if sentence and not tracker.in_segment and now - last_sentence_activity >= args.sentence_gap:
                    speak_sentence(now)

                # Vẽ landmarks lên màn hình trực tiếp nếu quan sát thấy
                if getattr(obs, "results", None):
                    res = obs.results
                    if hasattr(res, "pose_landmarks") and res.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(60, 180, 75), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 225, 25), thickness=1, circle_radius=1),
                        )
                    if hasattr(res, "left_hand_landmarks") and res.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(230, 25, 75), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 130, 48), thickness=1, circle_radius=1),
                        )
                    if hasattr(res, "right_hand_landmarks") and res.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 130, 200), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(70, 240, 240), thickness=1, circle_radius=1),
                        )

                # Trạng thái REC / IDLE
                if tracker.in_segment:
                    cv2.circle(frame, (25, 25), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "REC GESTURE", (42, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (25, 25), 8, (140, 140, 140), -1)
                    cv2.putText(frame, "IDLE", (42, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # Hiển thị câu đang ghép
                if sentence:
                    sentence_display = f"Cau: {' '.join(sentence)}"
                    frame = draw_unicode_text(
                        frame, sentence_display, (20, 52), font_size=18, color_bgr=(255, 255, 255), bg_color_bgr=(30, 30, 30)
                    )

                # Hiển thị thông báo lưu video gần nhất (trong 3.5s)
                if last_saved_info and (now - last_saved_time) < 3.5:
                    frame = draw_unicode_text(
                        frame, last_saved_info, (20, 85), font_size=16, color_bgr=(50, 255, 50), bg_color_bgr=(20, 20, 20)
                    )

                h_f = frame.shape[0]
                cv2.putText(
                    frame,
                    f"Words: {len(sentence)} | Q quit C clear S speak SPACE boundary",
                    (20, h_f - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                )
                cv2.imshow("VSL gesture prototype", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    tracker.reset()
                    recorder.cancel_gesture()
                    sentence.clear()
                    last_accepted_label = None
                    last_accepted_time = 0.0
                    print("Sentence cleared")
                elif key == ord("s"):
                    speak_sentence(now)
                elif key == 32:
                    handle_segment(tracker.force_boundary(now))
    finally:
        recorder.close()
        tts_mgr.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
