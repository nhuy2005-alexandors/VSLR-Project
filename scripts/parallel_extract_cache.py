"""Parallel landmark cache extraction across CPU workers without importing PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prototype_3_gestures.vsl3.console import configure_utf8_stdio
from prototype_3_gestures.vsl3.features import FEATURE_DIM, HolisticExtractor, SEQUENCE_LENGTH
from prototype_3_gestures.vsl3.labels import read_expected_labels

configure_utf8_stdio()


@dataclass(frozen=True)
class SimpleClip:
    label: str
    person: str
    path: Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_cache_key(
    path: Path,
    mtime: float,
    features_version: int = 3,
    feature_dim: int = FEATURE_DIM,
    sequence_length: int = SEQUENCE_LENGTH,
) -> str:
    stat = path.stat()
    source_identity = f"{stat.st_mtime_ns}|{stat.st_size}|{file_sha256(path)}"
    payload = "|".join(
        (
            path.as_posix(),
            f"{float(mtime):.9f}",
            source_identity,
            str(features_version),
            str(feature_dim),
            str(sequence_length),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def write_cache_entry(cached: Path, sequence: np.ndarray, stats: dict) -> None:
    cached.parent.mkdir(parents=True, exist_ok=True)
    staging = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp")
    try:
        with open(staging, "wb") as handle:
            np.savez(
                handle,
                sequence=sequence,
                sampled_frames=stats["sampled_frames"],
                trimmed_frames=stats["trimmed_frames"],
                hand_frame_ratio=stats["hand_frame_ratio"],
                left_hand_frame_ratio=stats.get("left_hand_frame_ratio", stats["hand_frame_ratio"]),
                right_hand_frame_ratio=stats.get("right_hand_frame_ratio", stats["hand_frame_ratio"]),
                fps=stats.get("fps", 0.0),
            )
        staging.replace(cached)
    except Exception:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def is_valid_cache(cached: Path) -> bool:
    if not cached.exists():
        return False
    try:
        with np.load(cached, allow_pickle=False) as data:
            seq = np.asarray(data["sequence"], dtype=np.float32)
            if seq.shape != (SEQUENCE_LENGTH, FEATURE_DIM) or not np.isfinite(seq).all():
                return False
            sampled = int(np.asarray(data["sampled_frames"]).item())
            trimmed = int(np.asarray(data["trimmed_frames"]).item())
            hand_ratio = float(np.asarray(data["hand_frame_ratio"]).item())
            if sampled < 8 or not (8 <= trimmed <= sampled) or not (0.10 <= hand_ratio <= 1.0):
                return False
        return True
    except Exception:
        return False


_worker_extractor: HolisticExtractor | None = None


def _init_worker():
    global _worker_extractor
    _worker_extractor = HolisticExtractor()


def _process_clip(clip: SimpleClip, cache_dir_str: str) -> tuple[str, bool, str | None]:
    global _worker_extractor
    assert _worker_extractor is not None
    cache_dir = Path(cache_dir_str)
    try:
        key = compute_cache_key(clip.path, clip.path.stat().st_mtime)
        cached = cache_dir / f"{key}.npz"
        if is_valid_cache(cached):
            return str(clip.path), True, None

        sequence, stats = _worker_extractor.extract_video(clip.path, SEQUENCE_LENGTH)
        stats.update({"label": clip.label, "person": clip.person, "from_cache": False})
        write_cache_entry(cached, sequence, stats)
        return str(clip.path), True, None
    except Exception as exc:
        return str(clip.path), False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract landmark cache in parallel (lightweight)")
    parser.add_argument("--data-dir", default="dataset/recordings_v2_4x24", help="Path to recordings root")
    parser.add_argument("--recording-plan", default="dataset/recording_plan_v2_4x24.json", help="Path to plan JSON")
    parser.add_argument("--labels-file", default="dataset/labels_v2_24.txt", help="Path to labels text file")
    parser.add_argument("--cache-dir", default="dataset/processed/landmark_cache_v2_4x24", help="Landmark cache directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    root = Path(args.data_dir)
    plan_data = json.loads(Path(args.recording_plan).read_text(encoding="utf-8"))
    labels = read_expected_labels(Path(args.labels_file))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    allowed_people = plan_data["people"]
    clips: list[SimpleClip] = []
    for p in allowed_people:
        for lbl in labels:
            ldir = root / p / lbl
            if ldir.is_dir():
                for f in sorted(ldir.iterdir()):
                    if f.is_file() and f.suffix.lower() in {".mov", ".mp4"} and not f.name.endswith(".bak"):
                        clips.append(SimpleClip(label=lbl, person=p, path=f.resolve()))

    print(f"Total clips to check/extract: {len(clips)} across {len(allowed_people)} people")

    needed: list[SimpleClip] = []
    already_cached = 0
    for clip in clips:
        key = compute_cache_key(clip.path, clip.path.stat().st_mtime)
        cached = cache_dir / f"{key}.npz"
        if is_valid_cache(cached):
            already_cached += 1
        else:
            needed.append(clip)

    print(f"Already cached: {already_cached} / {len(clips)}")
    print(f"Needed extraction: {len(needed)} clips")

    if not needed:
        print("All clips are already cached!")
        return

    workers = min(args.workers, len(needed))
    print(f"Starting parallel extraction with {workers} workers (PyTorch NOT loaded in workers)...")
    start_time = time.time()
    completed = 0
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {pool.submit(_process_clip, clip, str(cache_dir)): clip for clip in needed}
        for future in as_completed(futures):
            clip = futures[future]
            path_str, success, err = future.result()
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(needed) - completed) / rate if rate > 0 else 0
            if success:
                print(
                    f"[{completed + already_cached}/{len(clips)}] OK {clip.person}/{clip.label}/{clip.path.name} "
                    f"({rate:.2f} clips/s, ETA: {eta/60:.1f}m)",
                    flush=True,
                )
            else:
                failures.append((path_str, str(err)))
                print(
                    f"[{completed + already_cached}/{len(clips)}] FAILED {clip.person}/{clip.label}/{clip.path.name}: {err}",
                    flush=True,
                )

    print(f"\nFinished in {(time.time() - start_time)/60:.2f} minutes.")
    if failures:
        print(f"ERROR: {len(failures)} clips failed extraction!", file=sys.stderr)
        for p, err in failures:
            print(f"  {p}: {err}", file=sys.stderr)
        sys.exit(1)

    print("All clips extracted and cached successfully!")


if __name__ == "__main__":
    main()
