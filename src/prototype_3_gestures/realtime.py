from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .vsl3.features import HolisticExtractor, resample_sequence
from .vsl3.model import load_checkpoint, predict_sequence

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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


class SegmentTracker:
    """Decides where one gesture ends inside a continuous frame stream.

    A pure state machine — no model, no camera — so the boundary rules are testable. Feed it one
    frame at a time; it returns a completed segment or None.
    """

    def __init__(self, word_gap: float, max_frames: int):
        self.word_gap = word_gap
        self.max_frames = max_frames
        self.segment: list[np.ndarray] = []
        self.in_segment = False
        self.last_hand_time = 0.0
        # Set when max_frames forces a cut while the hands are still raised. Without it the very
        # next frame reopened a segment mid-gesture, so one slow gesture came out as two words.
        self.awaiting_hand_drop = False

    def feed(self, hands_present: bool, features: np.ndarray, now: float) -> list[np.ndarray] | None:
        if hands_present:
            self.last_hand_time = now
            if self.awaiting_hand_drop:
                return None  # same gesture still in progress; wait for the hands to come down
            if not self.in_segment:
                self.segment = []
                self.in_segment = True
            self.segment.append(features)
            if len(self.segment) >= self.max_frames:
                self.awaiting_hand_drop = True
                return self._close()
            return None

        self.awaiting_hand_drop = False
        if not self.in_segment:
            return None
        if now - self.last_hand_time < self.word_gap:
            self.segment.append(features)
            return None
        return self._close()

    def force_boundary(self) -> list[np.ndarray] | None:
        return self._close() if self.in_segment else None

    def reset(self) -> None:
        self.segment = []
        self.in_segment = False
        self.awaiting_hand_drop = False

    def _close(self) -> list[np.ndarray]:
        done, self.segment, self.in_segment = self.segment, [], False
        return done


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime 3-gesture -> sentence -> TTS demo")
    parser.add_argument("--model", default="models/gesture_lstm.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--word-gap", type=float, default=0.45, help="No-hand gap that ends one gesture")
    parser.add_argument("--sentence-gap", type=float, default=2.2, help="Additional idle time before speaking")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Hard cap on frames in one gesture. On hitting it the segment is classified and the "
        "rest of that gesture is discarded until the hands come down, so keep it above the "
        "slowest gesture you intend to sign.",
    )
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(Path(args.model), device)
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
    tracker = SegmentTracker(args.word_gap, args.max_frames)
    last_sentence_activity = time.monotonic()

    def handle_segment(segment: list[np.ndarray] | None) -> None:
        if segment is None:
            return
        nonlocal last_sentence_activity
        if len(segment) < args.min_frames:
            return
        label, confidence = classify_segment(model, labels, device, segment, seq_len)
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
                if done is not None and tracker.awaiting_hand_drop:
                    print(
                        f"Segment hit --max-frames ({args.max_frames}); classifying it and waiting for "
                        "the hands to come down before starting the next word."
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
                    handle_segment(tracker.force_boundary())
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
