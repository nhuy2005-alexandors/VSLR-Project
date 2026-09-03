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
    LEFT_PRESENCE_INDEX,
    MAX_BRIDGE_GAP_SECONDS,
    RIGHT_PRESENCE_INDEX,
    HolisticExtractor,
    preprocess_sequence,
    trim_active_frames,
)
from .vsl3.model import load_checkpoint, predict_sequence
from .vsl3.reject import (
    RejectPolicy,
    load_reject_policy,
    reject_prediction,
    sha256_file,
)

configure_utf8_stdio()


def speak_text(text: str) -> None:
    print(f"TTS> {text}")
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass

    if os.name == "nt":
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Speak([Console]::In.ReadToEnd())"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                input=text,
                text=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    print("TTS backend unavailable; sentence was printed instead.", file=sys.stderr)


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
        times = list(self.timestamps) if len(self.timestamps) == n_frames else list(range(n_frames))
        if not any(left[index] or right[index] for index in range(n_frames)):
            return [], [], [], []
        # Only remove no-hand padding outside the gesture. Internal gaps are meaningful input to
        # presence-aware preprocessing: it may bridge a short bounded gap or preserve a long gap
        # as missing. The helper is also used by offline video extraction.
        trimmed, trimmed_left, trimmed_right, trimmed_times = trim_active_frames(
            np.asarray(self.features, dtype=np.float32), left, right, times
        )
        return (
            [frame for frame in trimmed],
            trimmed_left.tolist(),
            trimmed_right.tolist(),
            trimmed_times.tolist(),
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
    return SegmentDecision(label, confidence, accepted, reason)


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
        self.pending: list[tuple[np.ndarray, float, bool, bool]] = []
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
        active = left or right

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
                    self.in_segment = True
                    self.start_time = pending[0][1]
                    for pending_values, pending_time, pending_left, pending_right in pending:
                        completed = self._append(
                            pending_values,
                            pending_time,
                            hands_up=True,
                            left_hand_present=pending_left,
                            right_hand_present=pending_right,
                        )
                        if completed is not None:
                            return completed
                    promoted = True
                else:
                    self.segment = []
                    self.segment_times = []
                    self.segment_left = []
                    self.segment_right = []
                    self.in_segment = True
                    self.start_time = now
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
            )

        if self.pending:
            # Pending evidence was a blip; discard it without ever exposing a segment.
            self.pending = []
            return None
        if self.awaiting_hand_drop:
            if now - self.last_hand_time >= self.word_gap:
                self.awaiting_hand_drop = False
            return None

        if not self.in_segment:
            return None
        if now - self.last_hand_time < self.word_gap:
            # Keep boundary timing for the tracker, but classify_segment() removes these frames
            # using the explicit masks. Thus a no-hand tail can never enter the model.
            return self._append(
                values,
                now,
                hands_up=False,
                left_hand_present=False,
                right_hand_present=False,
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
        self.pending = []
        self.in_segment = False
        self.awaiting_hand_drop = False
        self.last_timestamp = None

    def _append(
        self,
        features: np.ndarray,
        now: float,
        *,
        hands_up: bool,
        left_hand_present: bool,
        right_hand_present: bool,
    ) -> Segment | None:
        self.segment.append(features)
        self.segment_times.append(now)
        self.segment_left.append(left_hand_present)
        self.segment_right.append(right_hand_present)
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
        return Segment(
            features=done,
            start_time=self.start_time,
            active_end_time=self.last_hand_time,
            end_time=now,
            forced=forced,
            timestamps=times,
            left_hand_present=left,
            right_hand_present=right,
        )


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not math.isfinite(args.confidence) or not 0.0 < args.confidence <= 1.0:
        parser.error(f"--confidence must be in (0, 1], got {args.confidence}")
    for name, value in (("--word-gap", args.word_gap), ("--sentence-gap", args.sentence_gap)):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{name} must be finite and > 0, got {value}")
    if not math.isfinite(args.min_seconds) or args.min_seconds <= 0.0:
        parser.error(f"--min-seconds must be finite and > 0, got {args.min_seconds}")
    if not math.isfinite(args.max_seconds) or args.max_seconds < args.min_seconds:
        parser.error(
            f"--max-seconds ({args.max_seconds}) must be finite and >= "
            f"--min-seconds ({args.min_seconds})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime VSL gesture -> sentence -> TTS demo")
    parser.add_argument("--model", default="models/gesture_lstm.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--reject-policy", help="Calibrated reject-policy JSON; absent means uncalibrated")
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Allow manual --confidence threshold before negative calibration (never claim OOD safety)",
    )
    parser.add_argument("--word-gap", type=float, default=0.45, help="No-hand gap that ends one gesture")
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(Path(args.model), device)
    try:
        checkpoint_sha256 = sha256_file(args.model)
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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    sentence: list[str] = []
    tracker = SegmentTracker(args.word_gap, args.max_seconds, min_active_seconds=args.min_seconds)
    last_sentence_activity = time.monotonic()

    def handle_segment(segment: Segment | None) -> None:
        nonlocal last_sentence_activity
        if segment is None or segment.duration < args.min_seconds:
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
        if decision.accepted:
            sentence.append(decision.label)
            print(
                f"WORD> {decision.label} ({decision.confidence:.1%}) | sentence: {' '.join(sentence)}"
            )
            last_sentence_activity = time.monotonic()
        else:
            print(f"REJECT> {decision.label} ({decision.confidence:.1%}) — {decision.reason}")

    def speak_sentence(now: float) -> None:
        nonlocal last_sentence_activity
        if not sentence:
            return
        text = " ".join(sentence)
        if args.no_tts:
            print(f"SENTENCE> {text}")
        else:
            speak_text(text)
        sentence.clear()
        last_sentence_activity = now

    try:
        with HolisticExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                obs = extractor.process_frame(frame)
                done = tracker.feed(
                    obs.hands_present,
                    obs.features,
                    now,
                    obs.left_hand_present,
                    obs.right_hand_present,
                )
                if done is not None and done.forced:
                    print(
                        f"Segment hit --max-seconds ({args.max_seconds}); classifying and waiting for hand drop."
                    )
                handle_segment(done)

                if sentence and not tracker.in_segment and now - last_sentence_activity >= args.sentence_gap:
                    speak_sentence(now)

                status = "GESTURE" if tracker.in_segment else "IDLE"
                cv2.putText(frame, f"State: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"Words: {len(sentence)} | Q quit C clear S speak SPACE boundary",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow("VSL gesture prototype", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    tracker.reset()
                    sentence.clear()
                    print("Sentence cleared")
                elif key == ord("s"):
                    speak_sentence(now)
                elif key == 32:
                    handle_segment(tracker.force_boundary(now))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
