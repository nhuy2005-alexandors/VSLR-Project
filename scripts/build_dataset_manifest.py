"""Verify dataset locking and generate dataset_files_sha256.csv for Phase 2."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
import unicodedata
from pathlib import Path

import cv2

from prototype_3_gestures.prepare_train import file_sha256, load_directory_contract, validate_recording_tree
from prototype_3_gestures.vsl3.labels import normalise_label


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path("dataset/recordings_v1_p123")
    plan_path = Path("dataset/recording_plan_p123.json")
    labels_path = Path("dataset/labels.txt")
    out_csv = Path("dataset_files_sha256.csv")

    print("=== PHASE 2: VERIFYING AND LOCKING DATASET ===")

    # 1. Verify contract
    plan, labels, manifest_path = load_directory_contract(root, plan_path, labels_path)
    assert len(labels) == 25, f"Expected exactly 25 labels, got {len(labels)}"
    assert list(plan.people) == ["P01", "P02", "P03"], f"Expected P01-P03, got {plan.people}"
    assert plan.clips_per_label == 6, f"Expected 6 clips per label, got {plan.clips_per_label}"

    # 2. Run structural validation gate
    result = validate_recording_tree(root, plan, labels, strict_counts=True)
    if not result.valid:
        print("ERROR: Structural validation failed!", file=sys.stderr)
        for err in result.errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(1)
    print("Structural validation: PASS (0 errors)")

    # 3. Find all video files
    video_paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".mov", ".mp4"})
    assert len(video_paths) == 450, f"Expected exactly 450 videos, got {len(video_paths)}"
    print(f"Found {len(video_paths)} videos in {root}")

    # Ensure no .bak files in dataset
    bak_files = list(root.rglob("*.bak"))
    assert len(bak_files) == 0, f"Found .bak files in dataset: {bak_files}"
    print("Checked no .bak files in dataset: PASS")

    # 4. Check each video metadata, hash, container
    seen_paths = set()
    seen_hashes = set()
    rows = []

    for p in video_paths:
        # Check person and label
        parts = p.relative_to(root).parts
        assert len(parts) == 3, f"Unexpected path structure: {p}"
        person, label_dir, filename = parts
        assert person in {"P01", "P02", "P03"}, f"Invalid person: {person}"
        assert person != "P04", f"P04 must not exist in dataset: {p}"
        
        # NFC label
        nfc_label = normalise_label(label_dir)
        assert nfc_label in labels, f"Label {nfc_label} not in manifest"

        # Unique relative path
        rel_posix = p.relative_to(root).as_posix()
        assert rel_posix not in seen_paths, f"Duplicate path: {rel_posix}"
        seen_paths.add(rel_posix)

        # File size and hash
        size_bytes = p.stat().st_size
        sha = file_sha256(p)
        assert sha not in seen_hashes, f"Duplicate hash: {sha} at {p}"
        seen_hashes.add(sha)

        # OpenCV inspection
        cap = cv2.VideoCapture(str(p.resolve()))
        opened = cap.isOpened()
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read() if opened else (False, None)
        cap.release()
        duration = frames / fps if fps > 0 else -1.0

        assert opened, f"Could not open video {p}"
        assert ok, f"Could not read first frame of {p}"
        assert width == 1920 and height == 1080, f"Unexpected resolution {width}x{height} for {p}"
        assert 58.0 <= fps <= 62.0, f"Unexpected FPS {fps} for {p}"
        assert 0.5 <= duration <= 10.0, f"Duration {duration:.3f}s out of bounds [0.5, 10.0] for {p}"

        rows.append({
            "relative_path": rel_posix,
            "person": person,
            "label": nfc_label,
            "filename": filename,
            "bytes": size_bytes,
            "sha256": sha,
            "width": width,
            "height": height,
            "fps": round(fps, 4),
            "duration": round(duration, 4),
        })

    assert len(seen_hashes) == 450, f"Expected 450 unique hashes, got {len(seen_hashes)}"
    print("450 unique paths and 450 unique SHA-256 hashes: PASS")
    print("100% videos are 1920x1080 ~60fps and duration in [0.5, 10.0]s: PASS")

    # Write CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["relative_path", "person", "label", "filename", "bytes", "sha256", "width", "height", "fps", "duration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated manifest: {out_csv} ({len(rows)} rows)")

    # Print SHA-256 of locked files
    csv_sha = get_sha256(out_csv)
    labels_sha = get_sha256(labels_path)
    plan_sha = get_sha256(plan_path)

    print(f"SHA-256 {labels_path}: {labels_sha}")
    print(f"SHA-256 {plan_path}: {plan_sha}")
    print(f"SHA-256 {out_csv}: {csv_sha}")

    # Verify trimmed clip specifically
    trimmed_path = root / "P02" / "Hôm nay bạn khỏe không" / "002.MOV"
    orig_bak_path = Path("recordings_v1/P02/Hôm nay bạn khỏe không/002.MOV.bak")
    
    cap_trim = cv2.VideoCapture(str(trimmed_path.resolve()))
    fps_trim = cap_trim.get(cv2.CAP_PROP_FPS)
    frames_trim = cap_trim.get(cv2.CAP_PROP_FRAME_COUNT)
    dur_trim = frames_trim / fps_trim
    cap_trim.release()

    cap_bak = cv2.VideoCapture(str(orig_bak_path.resolve()))
    fps_bak = cap_bak.get(cv2.CAP_PROP_FPS)
    frames_bak = cap_bak.get(cv2.CAP_PROP_FRAME_COUNT)
    dur_bak = frames_bak / fps_bak
    cap_bak.release()

    print(f"Trimmed clip {trimmed_path}: {dur_trim:.4f}s (Frames: {frames_trim}, FPS: {fps_trim:.2f})")
    print(f"Backup clip {orig_bak_path}: {dur_bak:.4f}s (Frames: {frames_bak}, FPS: {fps_bak:.2f})")
    assert 9.4 <= dur_trim <= 9.6, f"Expected trimmed clip ~9.51s, got {dur_trim}"
    assert 10.0 <= dur_bak <= 10.1, f"Expected backup clip ~10.055s, got {dur_bak}"
    print("Trimmed clip verification: PASS")

    print("=== PHASE 2 COMPLETE: DATASET FULLY LOCKED ===")


if __name__ == "__main__":
    main()
