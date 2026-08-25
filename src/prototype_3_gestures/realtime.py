from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .vsl3.console import configure_utf8_stdio
from .vsl3.features import HolisticExtractor, resample_sequence
from .vsl3.model import load_checkpoint, predict_sequence

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


def classify_segment(model, labels, device, segment: list[np.ndarray], seq_len: int) -> tuple[str, float]:
    sequence = resample_sequence(np.stack(segment).astype(np.float32), seq_len)
    label, confidence, _ = predict_sequence(model, sequence, labels, device)
    return label, confidence


def should_accept_prediction(confidence: float, threshold: float) -> bool:
    return confidence >= threshold


@dataclass(frozen=True)
class Segment:
    """One completed gesture and both its active and boundary timestamps."""

    features: list[np.ndarray]
    start_time: float
    active_end_time: float
    end_time: float
    forced: bool

    @property
    def duration(self) -> float:
        """Time from first to last hand detection; excludes the trailing no-hand word gap."""
        return self.active_end_time - self.start_time


class SegmentTracker:
    """Decides where one gesture ends inside a continuous frame stream.

    A pure state machine — no model, no camera — so the boundary rules are testable. Feed it one
    frame at a time; it returns a completed `Segment` or None.

    Thresholds are DURATIONS, not frame counts. The same gesture reaches this class at ~60 fps from
    a video file and at MediaPipe's throughput (~20 fps measured) from the live camera, so a
    frame-count cap silently encoded two different real-world limits — 300 frames was 5 s offline
    and 14 s on the webcam.
    """

    def __init__(self, word_gap: float, max_seconds: float):
        self.word_gap = word_gap
        self.max_seconds = max_seconds
        self.segment: list[np.ndarray] = []
        self.in_segment = False
        self.start_time = 0.0
        self.last_hand_time = 0.0
        # Set when max_seconds forces a cut while the hands are still raised. Without it the very
        # next frame reopened a segment mid-gesture, so one slow gesture came out as two words.
        self.awaiting_hand_drop = False

    def feed(self, hands_present: bool, features: np.ndarray, now: float) -> Segment | None:
        if hands_present:
            self.last_hand_time = now
            if self.awaiting_hand_drop:
                return None  # same gesture still in progress; wait for the hands to come down
            if not self.in_segment:
                self.segment = []
                self.in_segment = True
                self.start_time = now
            return self._append(features, now, hands_up=True)

        if self.awaiting_hand_drop:
            # Same hysteresis the close path below uses. Clearing on the first no-hand frame
            # treated a dropped detection — noise everywhere else in this machine — as the hands
            # coming down, and the next frame opened a second word mid-gesture.
            if now - self.last_hand_time >= self.word_gap:
                self.awaiting_hand_drop = False
            return None

        if not self.in_segment:
            return None
        if now - self.last_hand_time < self.word_gap:
            return self._append(features, now, hands_up=False)
        return self._close(now, forced=False)

    def force_boundary(self, now: float) -> Segment | None:
        return self._close(now, forced=False) if self.in_segment else None

    def reset(self) -> None:
        self.segment = []
        self.in_segment = False
        self.awaiting_hand_drop = False

    def _append(self, features: np.ndarray, now: float, hands_up: bool) -> Segment | None:
        """Single place that grows a segment, so max_seconds caps every path into it."""
        self.segment.append(features)
        if now - self.start_time >= self.max_seconds:
            # Only wait for a hand-drop if the hands were actually still up when the cap hit;
            # otherwise the next word would be swallowed.
            self.awaiting_hand_drop = hands_up
            return self._close(now, forced=hands_up)
        return None

    def _close(self, now: float, forced: bool) -> Segment:
        done, self.segment, self.in_segment = self.segment, [], False
        return Segment(
            features=done,
            start_time=self.start_time,
            active_end_time=self.last_hand_time,
            end_time=now,
            forced=forced,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime 3-gesture -> sentence -> TTS demo")
    parser.add_argument("--model", default="models/gesture_lstm.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--word-gap", type=float, default=0.45, help="No-hand gap that ends one gesture")
    parser.add_argument("--sentence-gap", type=float, default=2.2, help="Additional idle time before speaking")
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.35,
        help="Segments shorter than this are dropped without classifying. In seconds, not frames, "
        "so it means the same thing on a 60 fps file and on a ~20 fps webcam.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=5.0,
        help="Hard cap on one gesture. On hitting it the segment is classified and the rest of "
        "that gesture is discarded until the hands come down, so keep it above the slowest "
        "gesture you intend to sign.",
    )
    parser.add_argument(
        "--allow-incompatible-model",
        action="store_true",
        help="Explicitly allow a checkpoint trained with another FEATURES_VERSION. Predictions may "
        "be unreliable; intended only for a temporary legacy demo before retraining.",
    )
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()
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
            f"--min-seconds ({args.min_seconds}); "
            "otherwise every segment is dropped and the demo silently recognises nothing."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(
        Path(args.model),
        device,
        allow_incompatible_features=args.allow_incompatible_model,
    )
    seq_len = int(config["sequence_length"])
    print(f"Loaded labels: {labels}")
    print("Controls: Q quit | C clear sentence | S speak now | SPACE force current gesture boundary")
    print(
        "Prototype rule: lower hands / leave a short gap between gestures. "
        "After the final gesture, stay idle until TTS speaks the whole sentence."
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    sentence: list[str] = []
    tracker = SegmentTracker(args.word_gap, args.max_seconds)
    last_sentence_activity = time.monotonic()

    def handle_segment(segment: Segment | None) -> None:
        if segment is None:
            return
        nonlocal last_sentence_activity
        if segment.duration < args.min_seconds:
            return
        label, confidence = classify_segment(model, labels, device, segment.features, seq_len)
        if should_accept_prediction(confidence, args.confidence):
            sentence.append(label)
            print(f"WORD> {label} ({confidence:.1%}) | sentence: {' '.join(sentence)}")
            last_sentence_activity = time.monotonic()
        else:
            print(f"REJECT> {label} ({confidence:.1%}) below threshold {args.confidence:.0%}")

    def speak_sentence(now: float) -> None:
        nonlocal last_sentence_activity
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
                if obs.hands_present:
                    last_sentence_activity = now

                done = tracker.feed(obs.hands_present, obs.features, now)
                if done is not None and done.forced:
                    print(
                        f"Segment hit --max-seconds ({args.max_seconds}); classifying it and waiting "
                        "for the hands to come down before starting the next word."
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
                elif key == ord("s") and sentence:
                    speak_sentence(now)
                elif key == 32:
                    handle_segment(tracker.force_boundary(now))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
