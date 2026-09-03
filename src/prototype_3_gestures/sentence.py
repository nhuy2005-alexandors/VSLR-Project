"""Offline sentence evaluation using the exact realtime segment/classification path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from .realtime import Segment, SegmentDecision, SegmentTracker, decide_segment
from .vsl3.console import configure_utf8_stdio
from .vsl3.features import ClipExtractionError, HolisticExtractor, require_valid_video_fps
from .vsl3.model import load_checkpoint
from .vsl3.reject import load_reject_policy, reject_policy_dict, sha256_file

configure_utf8_stdio()


@dataclass(frozen=True)
class SentenceManifestRow:
    person: str
    clip: str
    words: list[str]
    negative: bool = False


@dataclass(frozen=True)
class SentenceResult:
    path: Path
    predicted_words: list[str]
    segments: list[dict]
    expected_words: list[str] | None = None
    negative: bool = False

    @property
    def exact(self) -> bool | None:
        if self.expected_words is None:
            return None
        return self.predicted_words == self.expected_words


def _split_words(text: str) -> list[str]:
    text = text.strip()
    if not text or text.casefold() in {"negative", "none", "null", "-"}:
        return []
    parts = re.split(r"\s*[|;,]\s*", text)
    return [part.strip() for part in parts if part.strip()]


def _truthy(text: str) -> bool:
    return text.strip().casefold() in {"1", "true", "yes", "y", "negative", "neg"}


def parse_sentence_manifest(path: str | Path) -> list[SentenceManifestRow]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"person", "clip", "words"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("sentence manifest must have CSV columns person,clip,words")
        rows: list[SentenceManifestRow] = []
        seen_paths: set[tuple[str, str]] = set()
        for line_number, raw in enumerate(reader, start=2):
            person = (raw.get("person") or "").strip()
            clip = (raw.get("clip") or "").strip()
            words = _split_words(raw.get("words") or "")
            negative = _truthy(raw.get("negative") or "") or not words
            if not person or not clip:
                raise ValueError(f"sentence manifest row {line_number} needs person and clip")
            path_key = (person.casefold(), Path(clip).as_posix().casefold())
            if path_key in seen_paths:
                raise ValueError(f"duplicate sentence clip path at row {line_number}: {person}/{clip}")
            seen_paths.add(path_key)
            rows.append(SentenceManifestRow(person, clip, words, negative))
    if not rows:
        raise ValueError(f"sentence manifest has no rows: {path}")
    return rows


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, start=1):
        current = [index]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    return edit_distance(reference, hypothesis) / max(len(reference), 1)


def _validate_labels(words: list[str], labels: list[str], context: str) -> None:
    unknown = [word for word in words if word not in labels]
    if unknown:
        raise ValueError(f"{context} contains words not present in checkpoint labels: {unknown}")


def infer_sentence_clip(
    path: str | Path,
    model,
    labels: list[str],
    config: dict,
    device: torch.device,
    *,
    word_gap: float = 0.45,
    max_seconds: float = 5.0,
    min_seconds: float = 0.35,
    confidence: float = 0.72,
    reject_policy=None,
    allow_uncalibrated: bool = False,
) -> tuple[list[str], list[dict]]:
    """Run offline frames through Holistic → SegmentTracker → shared classifier/reject policy."""

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ClipExtractionError(f"Cannot open video: {path}")
    try:
        fps = require_valid_video_fps(cap.get(cv2.CAP_PROP_FPS), path)
    except ClipExtractionError:
        cap.release()
        raise
    tracker = SegmentTracker(word_gap, max_seconds, min_active_seconds=min_seconds)
    decisions: list[dict] = []
    predicted: list[str] = []
    last_frame_time = 0.0

    def handle(segment: Segment | None) -> None:
        if segment is None or segment.duration < min_seconds:
            return
        decision: SegmentDecision = decide_segment(
            model,
            labels,
            device,
            segment,
            int(config["sequence_length"]),
            confidence_threshold=confidence,
            reject_policy=reject_policy,
            allow_uncalibrated=allow_uncalibrated,
        )
        row = {
            "start_frame": int(round(segment.start_time * fps)),
            "end_frame": int(round(segment.end_time * fps)),
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "active_duration": segment.duration,
            "label": decision.label,
            "confidence": decision.confidence,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "forced": segment.forced,
        }
        decisions.append(row)
        if decision.accepted:
            predicted.append(decision.label)

    try:
        with HolisticExtractor() as extractor:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                last_frame_time = frame_index / fps
                obs = extractor.process_frame(frame)
                handle(
                    tracker.feed(
                        obs.hands_present,
                        obs.features,
                        last_frame_time,
                        obs.left_hand_present,
                        obs.right_hand_present,
                    )
                )
                frame_index += 1
    finally:
        cap.release()
    if frame_index < 1:
        raise ClipExtractionError(f"Video {path} has no readable frames")
    if tracker.in_segment:
        handle(tracker.force_boundary(last_frame_time))
    return predicted, decisions


def evaluate_sentence_words(
    expected: list[str], predicted: list[str], *, negative: bool = False
) -> dict:
    distance = edit_distance(expected, predicted)
    return {
        "exact": predicted == expected if not negative else not predicted,
        "edit_distance": distance,
        "wer": word_error_rate(expected, predicted) if not negative else 0.0,
        "count_mismatch": len(expected) != len(predicted),
        "false_accept": bool(predicted) if negative else False,
    }


def _resolve_manifest_clip(root: Path, row: SentenceManifestRow) -> Path:
    candidate = Path(row.clip)
    if candidate.is_absolute():
        raise ValueError("sentence manifest batch paths must be relative to --dir")
    if row.person in {".", ".."} or Path(row.person).name != row.person:
        raise ValueError(f"invalid sentence manifest person path: {row.person!r}")
    if candidate.name != row.clip or any(part in {".", ".."} for part in candidate.parts):
        raise ValueError(f"sentence manifest clip must be a filename under its person: {row.clip!r}")
    person_root = (root / row.person).resolve()
    resolved = (person_root / candidate).resolve()
    if resolved.parent != person_root:
        raise ValueError(f"sentence manifest clip escapes its person directory: {row.person}/{row.clip}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline sentence evaluation using the same SegmentTracker as vslr-camera"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="One sentence video")
    source.add_argument("--dir", help="Directory containing person/clip paths from --manifest")
    parser.add_argument("--manifest", help="CSV person,clip,words for batch mode")
    parser.add_argument("--expect", help="Expected labels for --clip, separated by | or comma")
    parser.add_argument("--model", default="models/gesture_lstm.pt")
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--reject-policy", help="Calibrated reject-policy JSON")
    parser.add_argument("--allow-uncalibrated", action="store_true")
    parser.add_argument("--word-gap", type=float, default=0.45)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--min-seconds", type=float, default=0.35)
    parser.add_argument("--json-out", help="Optional evaluation JSON path outside models/")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("confidence", "word_gap", "max_seconds", "min_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.confidence > 1.0:
        parser.error("--confidence must be <= 1")
    if args.max_seconds < args.min_seconds:
        parser.error("--max-seconds must be >= --min-seconds")
    if args.dir and not args.manifest:
        parser.error("--dir requires --manifest")
    if args.manifest and not args.dir:
        parser.error("--manifest is only valid with --dir")
    if args.expect and not args.clip:
        parser.error("--expect is only valid with --clip")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, config = load_checkpoint(args.model, device)
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
        print("REJECT POLICY: uncalibrated — accepted words are disabled until calibration data is supplied.")
    report_metadata = {
        "checkpoint_sha256": checkpoint_sha256,
        "training_signature": config.get("training_signature"),
        "reject_policy": reject_policy_dict(policy),
    }
    print(
        f"REPORT> checkpoint_sha256={report_metadata['checkpoint_sha256']} "
        f"training_signature={report_metadata['training_signature'] or '<missing>'} "
        f"reject_policy={'calibrated' if policy.calibrated else 'uncalibrated'}"
    )

    jobs: list[tuple[Path, list[str] | None, bool]] = []
    if args.clip:
        expected = _split_words(args.expect or "") if args.expect is not None else None
        if expected is not None:
            _validate_labels(expected, labels, "--expect")
        jobs.append((Path(args.clip), expected, False))
    else:
        manifest_path = Path(args.manifest)
        rows = parse_sentence_manifest(manifest_path)
        for row in rows:
            _validate_labels(row.words, labels, f"manifest row {row.person}/{row.clip}")
            try:
                resolved = _resolve_manifest_clip(Path(args.dir), row)
            except ValueError as exc:
                raise SystemExit(f"error: invalid sentence manifest path: {exc}") from exc
            jobs.append((resolved, row.words, row.negative))

    output_rows: list[dict] = []
    seen_job_paths: set[Path] = set()
    for path, expected, negative in jobs:
        if path.resolve() in seen_job_paths:
            raise SystemExit(f"error: duplicate sentence clip path: {path}")
        seen_job_paths.add(path.resolve())
        predicted, segments = infer_sentence_clip(
            path,
            model,
            labels,
            config,
            device,
            word_gap=args.word_gap,
            max_seconds=args.max_seconds,
            min_seconds=args.min_seconds,
            confidence=args.confidence,
            reject_policy=policy,
            allow_uncalibrated=args.allow_uncalibrated,
        )
        score = (
            evaluate_sentence_words(expected, predicted, negative=negative)
            if expected is not None
            else {"exact": None, "edit_distance": None, "wer": None, "count_mismatch": None, "false_accept": False}
        )
        row = {
            "video": str(path),
            "expected": expected,
            "predicted": predicted,
            "negative": negative,
            "segments": segments,
            **report_metadata,
            **score,
        }
        output_rows.append(row)
        print(f"\nVIDEO> {path}")
        for segment in segments:
            state = "ACCEPT" if segment["accepted"] else "REJECT"
            print(
                f"  {segment['start_frame']}-{segment['end_frame']} | {segment['label']} "
                f"{segment['confidence']:.1%} | {state} ({segment['reason']})"
            )
        print(f"  SENTENCE> {' | '.join(predicted) if predicted else '<empty>'}")
        if expected is not None:
            print(
                f"  EXPECTED> {' | '.join(expected) if expected else '<negative>'} | "
                f"exact={score['exact']} WER={score['wer'] if score['wer'] is not None else '-'} "
                f"edit={score['edit_distance']} count_mismatch={score['count_mismatch']}"
            )

    checked = [row for row in output_rows if row["expected"] is not None and not row["negative"]]
    negatives = [row for row in output_rows if row["negative"]]
    if len(jobs) > 1 or args.dir:
        exact = sum(bool(row["exact"]) for row in checked)
        total_distance = sum(int(row["edit_distance"]) for row in checked)
        total_words = sum(len(row["expected"]) for row in checked)
        count_mismatch = sum(bool(row["count_mismatch"]) for row in checked)
        false_accepts = sum(bool(row["false_accept"]) for row in negatives)
        print(
            f"\nSUMMARY> exact={exact}/{len(checked)} WER={total_distance / max(total_words, 1):.3f} "
            f"edit_distance={total_distance} count_mismatch={count_mismatch}/{len(checked)} "
            f"false_accept={false_accepts}/{len(negatives)}"
        )

    if args.json_out:
        destination = Path(args.json_out)
        if "models" in {part.casefold() for part in destination.parts}:
            raise SystemExit("error: vslr-sentence cannot write evaluation output inside models/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            staging.write_text(
                json.dumps({**report_metadata, "results": output_rows}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            staging.replace(destination)
        finally:
            staging.unlink(missing_ok=True)

    failures = [row for row in output_rows if row["expected"] is not None and not row["exact"]]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
