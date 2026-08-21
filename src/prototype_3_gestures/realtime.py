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


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime 3-gesture -> sentence -> TTS demo")
    parser.add_argument("--model", default="models/gesture_lstm.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--word-gap", type=float, default=0.45, help="No-hand gap that ends one gesture")
    parser.add_argument("--sentence-gap", type=float, default=2.2, help="Additional idle time before speaking")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=150)
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

    segment: list[np.ndarray] = []
    sentence: list[str] = []
    last_hand_time = time.monotonic()
    last_sentence_activity = time.monotonic()
    in_segment = False

    def commit_segment() -> None:
        nonlocal segment, in_segment, last_sentence_activity
        if len(segment) >= args.min_frames:
            label, confidence = classify_segment(model, labels, device, segment, seq_len)
            if should_accept_prediction(confidence, args.confidence):
                sentence.append(label)
                print(f"WORD> {label} ({confidence:.1%}) | sentence: {' '.join(sentence)}")
                last_sentence_activity = time.monotonic()
            else:
                print(f"REJECT> {label} ({confidence:.1%}) below threshold {args.confidence:.0%}")
        segment = []
        in_segment = False

    try:
        with HolisticExtractor() as extractor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                obs = extractor.process_frame(frame)

                if obs.hands_present:
                    if not in_segment:
                        segment = []
                        in_segment = True
                    segment.append(obs.features)
                    last_hand_time = now
                    last_sentence_activity = now
                elif in_segment:
                    if now - last_hand_time < args.word_gap:
                        segment.append(obs.features)
                    else:
                        commit_segment()

                if in_segment and len(segment) >= args.max_frames:
                    print("Segment reached max length; forcing a boundary.")
                    commit_segment()

                if sentence and not in_segment and now - last_sentence_activity >= args.sentence_gap:
                    text = " ".join(sentence)
                    if args.no_tts:
                        print(f"SENTENCE> {text}")
                    else:
                        speak_text(text)
                    sentence.clear()
                    last_sentence_activity = now

                status = "GESTURE" if in_segment else "IDLE"
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
                cv2.imshow("VSL 3-gesture prototype", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    segment = []
                    sentence.clear()
                    in_segment = False
                    print("Sentence cleared")
                elif key == ord("s") and sentence:
                    text = " ".join(sentence)
                    if args.no_tts:
                        print(f"SENTENCE> {text}")
                    else:
                        speak_text(text)
                    sentence.clear()
                    last_sentence_activity = now
                elif key == 32 and in_segment:
                    commit_segment()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
