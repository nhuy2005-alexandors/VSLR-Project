from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .vsl3.console import configure_utf8_stdio
from .vsl3.features import (
    AUGMENTATION_VERSION,
    FEATURE_DIM,
    FEATURE_CONTRACT,
    FEATURES_VERSION,
    SEQUENCE_LENGTH,
    ClipExtractionError,
    HolisticExtractor,
    augment_sequence,
)
from .vsl3.recording_plan import (
    DEFAULT_RECORDING_PLAN_FILE,
    RecordingPlan,
    load_recording_plan,
    resolve_labels_file,
)
from .vsl3.labels import (
    DEFAULT_LABELS_FILE,
    normalise_label,
    normalised_label_directories,
    read_expected_labels,
)
from .vsl3.model import (
    DEFAULT_BIDIRECTIONAL,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    MODEL_ARCHITECTURE_VERSION,
    POOLING_FWD_LAST_BWD_FIRST,
    GestureLSTM,
    save_checkpoint,
)

SINGLE_SIGNER = "unknown"
VIDEO_SUFFIXES = {".mov", ".mp4"}
MAX_FAILED_CLIP_RATIO = 0.05
TRAINING_RECIPE_VERSION = 1
REJECTION_CONTRACT = "closed-set-reject-v1-calibration-only"
SEGMENTATION_CONTRACT = "timestamp-segment-v2-min-active-no-tail"
TRAIN_LABEL_SMOOTHING = 0.03
OPTIMIZER_WEIGHT_DECAY = 1e-4
TRAINING_SIGNATURE_FIELDS = (
    "data_fingerprint",
    "labels",
    "epochs",
    "augmentations_per_clip",
    "batch_size",
    "learning_rate",
    "seed",
    "pooling",
    "features_version",
    "augmentation_version",
    "training_recipe_version",
    "sequence_length",
    "feature_dim",
    "feature_contract",
    "rejection_contract",
    "segmentation_contract",
    "recording_plan",
    "model",
    "optimizer",
    "loss",
    "device",
    "runtime",
)

configure_utf8_stdio()


@dataclass(frozen=True)
class Clip:
    label: str
    person: str
    path: Path


@dataclass(frozen=True)
class RecordingTreeValidation:
    """Structural result produced before MediaPipe is allowed to start."""

    errors: tuple[str, ...]
    content_hashes: dict[str, tuple[Path, ...]]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_recording_tree(
    data_dir: str | Path,
    plan: RecordingPlan,
    expected_labels: list[str],
    *,
    strict_counts: bool = True,
) -> RecordingTreeValidation:
    """Validate the signer/label tree and duplicate bytes without opening any video.

    ``strict_counts=False`` is used by the at-the-place recording checker: empty and short pairs
    are expected while recording, but unknown signers/labels, overfull pairs and exact duplicate
    content are still errors. Train and ship always use strict mode.
    """

    root = Path(data_dir)
    errors: list[str] = []
    hashes: dict[str, list[Path]] = {}
    expected = [normalise_label(label) for label in expected_labels]
    expected_set = set(expected)
    expected_people = set(plan.people)
    if not root.is_dir():
        return RecordingTreeValidation((f"--data-dir is not a directory: {root}",), {})

    actual_people = {path.name for path in root.iterdir() if path.is_dir()}
    unexpected_people = sorted(actual_people - expected_people)
    missing_people = sorted(expected_people - actual_people)
    if unexpected_people:
        errors.append(
            f"person directories outside recording plan: {unexpected_people}; "
            "do not mix legacy/unknown signers into V1"
        )
    if missing_people and strict_counts:
        errors.append(f"missing person directories from recording plan: {missing_people}")
    if strict_counts and actual_people != expected_people:
        errors.append(
            f"recording plan requires exactly {len(expected_people)} people {list(plan.people)}, "
            f"found {len(actual_people)} {sorted(actual_people)}"
        )

    for person in plan.people:
        person_dir = root / person
        if not person_dir.is_dir():
            continue
        try:
            label_dirs = normalised_label_directories(person_dir)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        actual_labels = {label for _, label in label_dirs}
        unknown = sorted(actual_labels - expected_set)
        if unknown:
            errors.append(f"{person_dir} has gesture directories outside labels manifest: {unknown}")
        by_label = {label: directory for directory, label in label_dirs}
        for label in expected:
            label_dir = by_label.get(label)
            paths = (
                sorted(
                    path
                    for path in label_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                )
                if label_dir is not None
                else []
            )
            count = len(paths)
            if strict_counts and count != plan.clips_per_label:
                state = "missing" if count < plan.clips_per_label else "overfull"
                errors.append(
                    f"{person}/{label}: {state} {count} video(s), expected exactly "
                    f"{plan.clips_per_label}"
                )
            elif not strict_counts and count > plan.clips_per_label:
                errors.append(
                    f"{person}/{label}: overfull {count} video(s), expected at most "
                    f"{plan.clips_per_label}"
                )
            for path in paths:
                digest = file_sha256(path)
                hashes.setdefault(digest, []).append(path.resolve())

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    for paths in sorted(duplicate_groups, key=lambda group: [path.as_posix() for path in group]):
        errors.append(
            "duplicate video content/hash detected; every path must be a distinct recording: "
            + " <-> ".join(str(path) for path in sorted(paths))
        )
    return RecordingTreeValidation(
        tuple(dict.fromkeys(errors)),
        {digest: tuple(sorted(paths)) for digest, paths in hashes.items()},
    )


def load_directory_contract(
    data_dir: str | Path,
    plan_path: str | Path,
    labels_file: str | Path | None,
) -> tuple[RecordingPlan, list[str], Path]:
    """Load plan and labels before discovery/extraction; shared by train and check callers."""

    plan = load_recording_plan(plan_path)
    manifest_path = (
        Path(labels_file).resolve()
        if labels_file is not None
        else resolve_labels_file(plan_path, plan)
    )
    labels = read_expected_labels(manifest_path)
    return plan, labels, manifest_path


def discover_clips(data_dir: str | Path, *, allow_empty: bool = False) -> list[Clip]:
    """Scan DIR/<person>/<gesture>/*.mov|*.mp4.

    The gesture directory name IS the label and goes straight into labels.json and the spoken
    sentence, so those directories must carry the human-readable Vietnamese name.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data-dir is not a directory: {data_dir}")

    clips: list[Clip] = []
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for label_dir, label in normalised_label_directories(person_dir):
            for path in sorted(label_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                    clips.append(
                        Clip(
                            label=label,
                            person=person_dir.name,
                            path=path.resolve(),
                        )
                    )

    if not clips and not allow_empty:
        raise ValueError(
            f"No .mov/.mp4 clips under {data_dir}. Expected layout DIR/<person>/<gesture>/*.mov"
        )
    return clips


def parse_video_specs(values: list[str]) -> list[Clip]:
    """Single-signer mode: every clip is tagged person=SINGLE_SIGNER.

    Filenames carry no signer identity, so this path can never support leave-one-signer-out.
    validate_clips(require_signer_split=True) is what refuses it.
    """
    clips: list[Clip] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --video '{value}'. Expected LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = normalise_label(label.strip())
        path = Path(raw_path.strip()).expanduser().resolve()
        if not label:
            raise ValueError("Gesture label cannot be empty")
        if not path.exists():
            raise FileNotFoundError(path)
        clips.append(Clip(label=label, person=SINGLE_SIGNER, path=path))

    paths = [clip.path for clip in clips]
    if len(set(paths)) != len(paths):
        raise ValueError("The same source video cannot be supplied more than once")
    return clips


def label_order(clips: list[Clip]) -> list[str]:
    return list(dict.fromkeys(clip.label for clip in clips))


def validate_clips(clips: list[Clip], *, require_signer_split: bool) -> None:
    if not clips:
        raise ValueError("No clips supplied")
    labels = label_order(clips)

    if not require_signer_split:
        counts = {label: sum(clip.label == label for clip in clips) for label in labels}
        thin = {label: count for label, count in counts.items() if count < 2}
        if thin:
            raise ValueError(f"Each gesture needs at least 2 clips, got {thin}")
        return

    people = sorted({clip.person for clip in clips})
    per_label_people = {label: {c.person for c in clips if c.label == label} for label in labels}

    if len(people) < 2:
        raise ValueError(
            f"Leave-one-signer-out needs clips from at least 2 different people, found {people}. "
            f"Gestures present: {labels}. Note --video mode tags every clip as person "
            f"'{SINGLE_SIGNER}', so it can never satisfy this — record into "
            "--data-dir DIR/<person>/<gesture>/ instead."
        )

    missing = [(label, person) for label in labels for person in people if person not in per_label_people[label]]
    if missing:
        raise ValueError(
            "Every gesture must be recorded by every person for leave-one-signer-out. "
            f"Missing {len(missing)} pair(s): {sorted(missing)}"
        )


def validate_expected_labels(clips: list[Clip], expected_labels: list[str]) -> None:
    """Require the data tree to match the authoritative manifest exactly.

    Looking only at the tree cannot detect a gesture that nobody recorded: an absent directory
    leaves no evidence. This gate runs before MediaPipe so a 26-class tree cannot silently train
    under a 27-class project configuration.
    """
    expected = [normalise_label(label) for label in expected_labels]
    actual = label_order(clips)
    actual_set = set(actual)
    expected_set = set(expected)
    missing = [label for label in expected if label not in actual_set]
    unexpected = [label for label in actual if label not in expected_set]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"labels with no clips anywhere: {missing}")
        if unexpected:
            details.append(f"unexpected gesture directories: {unexpected}")
        raise ValueError(
            "Dataset labels do not match the authoritative labels file (" + "; ".join(details) + "). "
            "Directory names must match the Vietnamese labels exactly."
        )


def validate_plan_clips(clips: list[Clip], plan: RecordingPlan, expected_labels: list[str]) -> None:
    """Validate exact surviving coverage after extraction failures, without touching MediaPipe."""

    expected = [normalise_label(label) for label in expected_labels]
    expected_people = set(plan.people)
    actual_people = {clip.person for clip in clips}
    errors: list[str] = []
    if actual_people != expected_people:
        errors.append(
            f"after extraction, people do not match plan: expected {list(plan.people)}, "
            f"found {sorted(actual_people)}"
        )
    for person in plan.people:
        for label in expected:
            count = sum(clip.person == person and normalise_label(clip.label) == label for clip in clips)
            if count != plan.clips_per_label:
                state = "missing" if count < plan.clips_per_label else "overfull"
                errors.append(
                    f"after extraction {person}/{label}: {state} {count} clip(s), expected exactly "
                    f"{plan.clips_per_label}"
                )
    if errors:
        raise ValueError("Recording plan coverage failed: " + "; ".join(errors))


def duplicate_content_errors(clips: list[Clip]) -> list[str]:
    """Return exact byte-duplicate groups, including every path in each group."""

    by_hash: dict[str, list[Path]] = {}
    for clip in clips:
        by_hash.setdefault(file_sha256(clip.path), []).append(clip.path.resolve())
    return [
        "duplicate video content/hash detected: " + " <-> ".join(str(path) for path in sorted(paths))
        for paths in sorted(by_hash.values(), key=lambda group: [path.as_posix() for path in group])
        if len(paths) > 1
    ]


def missing_labels_after_drop(expected_labels: list[str], surviving_clips: list[Clip]) -> list[str]:
    """Labels that lost every single clip.

    validate_clips only ever sees survivors, so a label with zero survivors is invisible to it —
    training would quietly solve a smaller problem under the original label count.
    """
    survivors = set(label_order(surviving_clips))
    return [label for label in expected_labels if label not in survivors]


def signer_split(clips: list[Clip], val_person: str) -> tuple[list[int], list[int]]:
    people = {clip.person for clip in clips}
    if val_person not in people:
        raise ValueError(f"Unknown person '{val_person}'. Known people: {sorted(people)}")
    train_idx = [i for i, clip in enumerate(clips) if clip.person != val_person]
    val_idx = [i for i, clip in enumerate(clips) if clip.person == val_person]
    if not train_idx:
        raise ValueError(f"Holding out '{val_person}' leaves no training clips")
    return train_idx, val_idx


def cache_key(
    path: str | Path,
    mtime: float,
    features_version: int = FEATURES_VERSION,
    feature_dim: int = FEATURE_DIM,
    sequence_length: int = SEQUENCE_LENGTH,
) -> str:
    source_path = Path(path)
    try:
        stat = source_path.stat()
    except FileNotFoundError:
        # Keeps the pure key helper usable for a not-yet-created path in callers/tests. Real
        # extraction stats the clip first and therefore never takes this fallback.
        source_identity = "missing"
    else:
        source_identity = f"{stat.st_mtime_ns}|{stat.st_size}|{file_sha256(source_path)}"
    payload = "|".join(
        (
            source_path.as_posix(),
            f"{float(mtime):.9f}",
            source_identity,
            str(features_version),
            str(feature_dim),
            str(sequence_length),
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def data_fingerprint(clips: list[Clip]) -> str:
    """Identify the exact clip set a run consumed.

    Lets a loso_report.json and a metrics.json be *shown* to describe the same data instead of
    merely claiming it — the repo's numbers rule needs that link to be checkable.
    """
    rows = []
    for clip in sorted(clips, key=lambda c: (c.person, c.label, c.path.as_posix())):
        stat = clip.path.stat()
        rows.append(
            f"{clip.person}|{clip.label}|{clip.path.as_posix()}|{stat.st_mtime_ns}|"
            f"{stat.st_size}|{file_sha256(clip.path)}"
        )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def training_signature(metadata: dict) -> str:
    """Hash every input that makes LOSO and ship training procedures comparable."""
    # Keep callers that build the pre-hardening metadata shape usable in tests/tools while making
    # every real run explicit about the new presence-aware feature contract.
    metadata = dict(metadata)
    metadata.setdefault("feature_contract", FEATURE_CONTRACT)
    metadata.setdefault("rejection_contract", REJECTION_CONTRACT)
    metadata.setdefault("segmentation_contract", SEGMENTATION_CONTRACT)
    metadata.setdefault("recording_plan", {})
    missing = [field for field in TRAINING_SIGNATURE_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"Cannot build training signature; missing fields: {missing}")
    payload = {field: metadata[field] for field in TRAINING_SIGNATURE_FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_pair_matches(report_signature: object, metrics: dict, checkpoint_path: str | Path) -> bool:
    """Verify the current weights file immediately before making a pairing claim."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_signature = checkpoint.get("config", {}).get("training_signature")
    checkpoint_labels = list(checkpoint.get("labels", []))
    return (
        isinstance(report_signature, str)
        and report_signature == metrics.get("training_signature") == checkpoint_signature
        and checkpoint_labels == list(metrics.get("labels", []))
        and metrics.get("checkpoint_sha256") == file_sha256(checkpoint_path)
    )


def run_metadata(args, clips: list[Clip], labels: list[str], people: list[str], device) -> dict:
    """Everything needed to reproduce a run, written into both report kinds."""
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": labels,
        "people": people,
        "clips": len(clips),
        "clips_per_label": {label: sum(clip.label == label for clip in clips) for label in labels},
        "data_fingerprint": data_fingerprint(clips),
        "epochs": args.epochs,
        "augmentations_per_clip": args.augment,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pooling": POOLING_FWD_LAST_BWD_FIRST,
        "features_version": FEATURES_VERSION,
        "feature_contract": FEATURE_CONTRACT,
        "rejection_contract": REJECTION_CONTRACT,
        "segmentation_contract": SEGMENTATION_CONTRACT,
        "reject_policy": {
            "schema_version": 1,
            "calibrated": False,
            "source": "not-calibrated",
        },
        "recording_plan": getattr(args, "recording_plan_metadata", {"mode": "legacy"}),
        "augmentation_version": AUGMENTATION_VERSION,
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "model": {
            "architecture_version": MODEL_ARCHITECTURE_VERSION,
            "hidden_size": DEFAULT_HIDDEN_SIZE,
            "num_layers": DEFAULT_NUM_LAYERS,
            "bidirectional": DEFAULT_BIDIRECTIONAL,
        },
        "optimizer": {"name": "AdamW", "weight_decay": OPTIMIZER_WEIGHT_DECAY},
        "loss": {"name": "CrossEntropyLoss", "train_label_smoothing": TRAIN_LABEL_SMOOTHING},
        "device": str(device),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
    }
    metadata["training_signature"] = training_signature(metadata)
    return metadata


def _write_cache_entry(cached: Path, sequence: np.ndarray, stats: dict) -> None:
    """Best effort. A cache write failure must never fail the clip — the landmarks are already in hand.

    Staged under a pid-tagged name and renamed into place, so an interrupted run resumes and two
    concurrent runs cannot fight over one staging file.
    """
    staging = cached.with_name(f"{cached.stem}.{os.getpid()}.tmp")
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends ".npz" to a path that lacks it, so hand it an open handle instead.
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
    except OSError as exc:
        warnings.warn(f"Could not cache landmarks for {Path(stats['video']).name}: {exc}", stacklevel=2)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_write_text(path: str | Path, text: str) -> None:
    """Publish JSON/text artifacts only after the complete payload is on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        staging.write_text(text, encoding="utf-8")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _cache_scalar(data, name: str):
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"cache field {name!r} must be scalar, got shape {value.shape}")
    return value.item()


def _read_cache_entry(cached: Path) -> tuple[np.ndarray, dict]:
    """Read and validate a cache entry before it can bypass MediaPipe."""
    with np.load(cached, allow_pickle=False) as data:
        sequence = np.asarray(data["sequence"], dtype=np.float32)
        if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            raise ValueError(
                f"cached sequence has shape {sequence.shape}, expected {(SEQUENCE_LENGTH, FEATURE_DIM)}"
            )
        if not np.isfinite(sequence).all():
            raise ValueError("cached sequence contains NaN or infinity")

        sampled_raw = float(_cache_scalar(data, "sampled_frames"))
        trimmed_raw = float(_cache_scalar(data, "trimmed_frames"))
        hand_ratio = float(_cache_scalar(data, "hand_frame_ratio"))
        if not sampled_raw.is_integer() or not trimmed_raw.is_integer():
            raise ValueError("cached frame counts must be integers")
        sampled_frames = int(sampled_raw)
        trimmed_frames = int(trimmed_raw)
        if sampled_frames < 8 or not 8 <= trimmed_frames <= sampled_frames:
            raise ValueError(
                f"invalid cached frame counts sampled={sampled_frames}, trimmed={trimmed_frames}"
            )
        if not np.isfinite(hand_ratio) or not 0.10 <= hand_ratio <= 1.0:
            raise ValueError(f"invalid cached hand_frame_ratio={hand_ratio}")

        def optional_ratio(name: str) -> float:
            if name not in data:
                return hand_ratio
            ratio = float(_cache_scalar(data, name))
            if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(f"invalid cached {name}={ratio}")
            return ratio

        left_ratio = optional_ratio("left_hand_frame_ratio")
        right_ratio = optional_ratio("right_hand_frame_ratio")
        fps = float(_cache_scalar(data, "fps")) if "fps" in data else 0.0
        if not np.isfinite(fps) or fps < 0.0:
            raise ValueError(f"invalid cached fps={fps}")

    return sequence, {
        "sampled_frames": sampled_frames,
        "trimmed_frames": trimmed_frames,
        "hand_frame_ratio": hand_ratio,
        "left_hand_frame_ratio": left_ratio,
        "right_hand_frame_ratio": right_ratio,
        "fps": fps,
    }


def extract_with_cache(clip: Clip, extractor, cache_dir: Path) -> tuple[np.ndarray, dict]:
    key = cache_key(clip.path, clip.path.stat().st_mtime)
    cached = cache_dir / f"{key}.npz"

    if cached.exists():
        try:
            sequence, cached_stats = _read_cache_entry(cached)
            stats = {
                "video": str(clip.path),
                "label": clip.label,
                "person": clip.person,
                **cached_stats,
                "from_cache": True,
            }
            return sequence, stats
        except Exception as exc:  # truncated / corrupt / structurally invalid entry
            # Must not be reported as a bad recording, and must not poison this clip forever:
            # drop the entry and fall through to a real extraction.
            warnings.warn(
                f"Discarding invalid cache entry for {clip.path.name} ({exc}); re-extracting.",
                stacklevel=2,
            )
            try:
                cached.unlink(missing_ok=True)
            except OSError:
                pass

    sequence, stats = extractor.extract_video(clip.path, SEQUENCE_LENGTH)
    stats.update({"label": clip.label, "person": clip.person, "from_cache": False})
    _write_cache_entry(cached, sequence, stats)
    return sequence, stats


class GestureDataset(Dataset):
    """Augments in __getitem__ instead of materialising every variant up front.

    At 540 clips the old pre-materialised array was ~3.15 GB (peak ~9.4 GB through np.stack +
    astype). Sample count per clip is unchanged, so --epochs and batch size keep their meaning:
    index k == 0 of each clip's block is the untouched original, k > 0 are augmentations.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        targets: list[int],
        augmentations_per_clip: int,
        seed: int,
    ):
        if len(sequences) != len(targets):
            raise ValueError("sequences and targets must have the same length")
        if augmentations_per_clip < 0:
            raise ValueError(f"augmentations_per_clip must be >= 0, got {augmentations_per_clip}")
        self.sequences = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
        self.targets = [int(target) for target in targets]
        self.samples_per_clip = augmentations_per_clip + 1
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.sequences) * self.samples_per_clip

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        clip_idx, aug_idx = divmod(index, self.samples_per_clip)
        original = self.sequences[clip_idx]
        if aug_idx == 0:
            features = original.copy()
        else:
            # Seeded by position, not by epoch: the same index always yields the same sample, so
            # a run is reproducible and DataLoader workers cannot collide.
            features = augment_sequence(original, np.random.default_rng([self.seed, clip_idx, aug_idx]))
        return torch.from_numpy(np.ascontiguousarray(features)), self.targets[clip_idx]


def build_eval_loader(sequences: list[np.ndarray], targets: list[int], args) -> DataLoader:
    """Unaugmented and unshuffled, so row i of `evaluate` is clip i.

    That alignment is the whole basis for naming which clip a wrong prediction came from. Shuffling
    or augmenting here would keep every accuracy number correct while silently attaching each
    prediction to the wrong filename.
    """
    dataset = GestureDataset(sequences, targets, 0, args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, list[dict]]:
    """Returns (mean loss, accuracy, one row per sample in loader order).

    `confidence` is softmax-max so it is the same quantity `--confidence` compares against in
    realtime (`predict_sequence`); a raw-logit "confidence" would not be comparable.
    """
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            total_loss += float(criterion(logits, targets)) * len(targets)
            confidences, predicted = torch.softmax(logits, dim=1).max(dim=1)
            correct += int((predicted == targets).sum())
            total += len(targets)
            for target, prediction, confidence in zip(
                targets.tolist(), predicted.tolist(), confidences.tolist(), strict=True
            ):
                rows.append({"target": target, "predicted": prediction, "confidence": confidence})
    return total_loss / max(total, 1), correct / max(total, 1), rows


def worst_labels(predictions: list[dict], limit: int = 5) -> list[dict]:
    """Lowest-accuracy labels, each with the label it is most often mistaken for.

    A single overall number cannot separate "wrong everywhere" from "two labels dead, 25 perfect",
    and those need completely different actions.
    """
    totals: dict[str, list[int]] = {}
    confusions: dict[str, dict[str, int]] = {}
    for row in predictions:
        label = row["label"]
        seen, right = totals.setdefault(label, [0, 0])
        totals[label] = [seen + 1, right + int(row["correct"])]
        if not row["correct"]:
            confusions.setdefault(label, {})
            confusions[label][row["predicted"]] = confusions[label].get(row["predicted"], 0) + 1

    ranked = []
    for label, (seen, right) in totals.items():
        if right == seen:
            continue
        wrong_as = confusions.get(label, {})
        ranked.append(
            {
                "label": label,
                "clips": seen,
                "correct": right,
                "accuracy": right / seen,
                "confused_with": max(wrong_as, key=wrong_as.get) if wrong_as else None,
            }
        )
    ranked.sort(key=lambda row: (row["accuracy"], row["label"]))
    return ranked[:limit]


def suspect_clips(predictions: list[dict], extraction: list[dict], min_hand_ratio: float = 0.5) -> list[dict]:
    """Clips that are BOTH predicted wrong and poorly recorded — the re-record candidates.

    Deliberately an absolute threshold, not "below the median": half of any set is below its own
    median, so a median rule can never return an empty list however good the recording is.
    """
    ratios = {row["video"]: row["hand_frame_ratio"] for row in extraction}
    missing = sorted({row["video"] for row in predictions if row["video"] not in ratios})
    if missing:
        raise ValueError(f"Prediction/extraction join is incomplete; missing video stats for: {missing}")
    suspects = [
        {**row, "hand_frame_ratio": ratios[row["video"]]}
        for row in predictions
        if not row["correct"] and ratios[row["video"]] < min_hand_ratio
    ]
    suspects.sort(key=lambda row: row["hand_frame_ratio"])
    return suspects


def train_model(
    train_sequences: list[np.ndarray],
    train_targets: list[int],
    num_classes: int,
    args,
    device: torch.device,
    val_sequences: list[np.ndarray] | None = None,
    val_targets: list[int] | None = None,
    log_prefix: str = "",
) -> tuple[GestureLSTM, list[dict]]:
    """Train for exactly args.epochs. No early stopping, no best-epoch selection.

    Deliberate: in --loso the held-out signer IS the test set, so stopping or selecting on it
    would report a number tuned on its own test data. And with a saturating val signal, picking
    the argmin val_loss is picking noise — that is how the shipped checkpoint once ended up
    trained for a single epoch (docs/GOTCHAS.md).
    """
    torch.manual_seed(args.seed)
    train_dataset = GestureDataset(train_sequences, train_targets, args.augment, args.seed)
    loader_kwargs = {"num_workers": args.num_workers}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)

    val_loader = None
    if val_sequences:
        # Never augmented: the reported number is per real clip. Small enough that workers cost more
        # than they save.
        val_loader = build_eval_loader(val_sequences, val_targets or [], args)

    model = GestureLSTM(
        FEATURE_DIM,
        num_classes,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        bidirectional=DEFAULT_BIDIRECTIONAL,
        pooling=POOLING_FWD_LAST_BWD_FIRST,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=OPTIMIZER_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_LABEL_SMOOTHING)

    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
            total += len(targets)

        # Named "_smoothed" because training uses label_smoothing=0.03 while accuracy() uses plain
        # cross-entropy: the two losses are on different scales and must not be read as one curve.
        row = {
            "epoch": epoch,
            "train_loss_smoothed": running_loss / total,
            "train_accuracy": correct / total,
        }
        if val_loader is not None:
            val_loss, val_acc, _ = evaluate(model, val_loader, device)
            row["val_loss_plain_ce"] = val_loss
            row["val_accuracy"] = val_acc
        history.append(row)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            line = (
                f"{log_prefix}epoch {epoch:03d} | "
                f"train {row['train_accuracy']:.1%}/{row['train_loss_smoothed']:.4f}"
            )
            if "val_accuracy" in row:
                line += f" | held-out signer {row['val_accuracy']:.1%}/{row['val_loss_plain_ce']:.4f}"
            print(line)

    del train_loader
    if val_loader is not None:
        del val_loader
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, history


def extract_all(clips: list[Clip], cache_dir: Path) -> tuple[list[np.ndarray], list[Clip], list[dict], list[dict]]:
    sequences: list[np.ndarray] = []
    kept: list[Clip] = []
    stats: list[dict] = []
    failures: list[dict] = []

    with HolisticExtractor() as extractor:
        for position, clip in enumerate(clips, start=1):
            try:
                sequence, clip_stats = extract_with_cache(clip, extractor, cache_dir)
            except (ClipExtractionError, FileNotFoundError) as exc:
                # Only failures attributable to this source clip may be tolerated. Permission,
                # MediaPipe/runtime, cache-programming, and other infrastructure errors must abort;
                # dropping up to 5% of those would train on a silently damaged dataset.
                failures.append(
                    {
                        "video": str(clip.path),
                        "label": clip.label,
                        "person": clip.person,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"  [{position}/{len(clips)}] SKIP {clip.path.name}: {exc}")
                continue
            sequences.append(sequence)
            kept.append(clip)
            stats.append(clip_stats)
            source = "cache" if clip_stats["from_cache"] else "mediapipe"
            print(
                f"  [{position}/{len(clips)}] {clip.person}/{clip.label}: "
                f"hands {clip_stats['hand_frame_ratio']:.1%} ({source})"
            )

    return sequences, kept, stats, failures


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def unit_interval(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {value}")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MediaPipe + BiLSTM gesture recogniser")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-dir",
        help="Root of DIR/<person>/<gesture>/*.mov. Gesture directories must match --labels-file; "
        "names are NFC-normalised and used as the model/TTS labels.",
    )
    source.add_argument(
        "--video",
        action="append",
        help="Single-signer mode: repeat as LABEL=PATH. Cannot be used with --loso (no signer identity).",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Measure only: leave-one-signer-out over every person. Writes loso_report.json, no checkpoint.",
    )
    parser.add_argument(
        "--labels-file",
        default=DEFAULT_LABELS_FILE,
        help="Authoritative label manifest for --data-dir. Every listed label must exist in the tree, "
        "and unexpected gesture directories are refused before extraction.",
    )
    parser.add_argument(
        "--recording-plan",
        default=DEFAULT_RECORDING_PLAN_FILE,
        help="Versioned directory-mode recording contract (people and clips per label)",
    )
    parser.add_argument(
        "--clips-per-label",
        type=positive_int,
        default=None,
        help="Optional assertion; directory mode must match recording plan exactly",
    )
    parser.add_argument("--model-dir", default="models", help="Directory for model and training metadata")
    parser.add_argument(
        "--cache-dir",
        default="dataset/processed/landmark_cache",
        help="Landmark cache; keyed on source SHA-256 plus path/mtime and the feature contract",
    )
    parser.add_argument(
        "--augment", type=non_negative_int, default=120, help="Augmented samples per clip, generated on the fly"
    )
    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=40,
        help="Fixed number of epochs. Not tuned on the LOSO folds by design; tuning it there would "
        "select on the test signer.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-3)
    parser.add_argument("--seed", type=non_negative_int, default=42)
    parser.add_argument(
        "--min-hand-ratio",
        type=unit_interval,
        default=0.5,
        help="In --loso reporting, mark a wrong clip as a re-record candidate below this hand ratio.",
    )
    parser.add_argument(
        "--num-workers",
        type=non_negative_int,
        default=0,
        help="DataLoader workers for on-the-fly augmentation. Augmentation costs ~0.62 ms/sample "
        "single-threaded (~40 s/epoch at 540 clips x 120), so raise this on a real run. Sampling is "
        "seeded by position, so worker count never changes the samples produced.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    plan: RecordingPlan | None = None
    try:
        if args.data_dir:
            # The plan and labels are loaded before discovery/extraction.  A default labels path
            # means "use the plan's relative labels_file"; an explicit custom path is supported
            # for isolated synthetic/test datasets but still shares the same people/count plan.
            labels_override = None if args.labels_file == DEFAULT_LABELS_FILE else args.labels_file
            plan, expected_labels, _ = load_directory_contract(
                args.data_dir, args.recording_plan, labels_override
            )
            args.recording_plan_metadata = {
                "schema_version": plan.schema_version,
                "dataset_version": plan.dataset_version,
                "people": list(plan.people),
                "clips_per_label": plan.clips_per_label,
            }
            if args.clips_per_label is not None and args.clips_per_label != plan.clips_per_label:
                raise ValueError(
                    f"--clips-per-label={args.clips_per_label} conflicts with recording plan "
                    f"({plan.clips_per_label})"
                )
            tree = validate_recording_tree(args.data_dir, plan, expected_labels, strict_counts=True)
            if not tree.valid:
                raise ValueError("recording plan gate failed before MediaPipe: " + " | ".join(tree.errors))
            clips = discover_clips(args.data_dir)
            validate_expected_labels(clips, expected_labels)
        else:
            clips = parse_video_specs(args.video)
            expected_labels = label_order(clips)
            duplicate_errors = duplicate_content_errors(clips)
            if duplicate_errors:
                raise ValueError("legacy --video leakage gate failed: " + " | ".join(duplicate_errors))
        validate_clips(clips, require_signer_split=args.loso)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print(f"Extracting landmarks for {len(clips)} clip(s) (cache: {cache_dir})...")
    sequences, clips, extraction_stats, failures = extract_all(clips, cache_dir)

    if not sequences:
        raise SystemExit("Every clip failed extraction; nothing to train on.")
    failed_ratio = len(failures) / (len(sequences) + len(failures))
    if failed_ratio > MAX_FAILED_CLIP_RATIO:
        raise SystemExit(
            f"{len(failures)} of {len(sequences) + len(failures)} clips failed extraction "
            f"({failed_ratio:.1%} > {MAX_FAILED_CLIP_RATIO:.0%}). Check the recording, and the video "
            "files themselves, before training.\n"
            + "\n".join(f"  {f['person']}/{f['label']} {Path(f['video']).name}: {f['error']}" for f in failures)
        )
    if failures:
        print(f"Dropped {len(failures)} unusable clip(s); re-checking invariants on what is left.")
        # Invariants were checked on the full list; dropped clips can break them.
        try:
            if plan is not None:
                validate_plan_clips(clips, plan, expected_labels)
            validate_clips(clips, require_signer_split=args.loso)
        except ValueError as exc:
            raise SystemExit(f"error after dropping {len(failures)} unusable clip(s): {exc}") from exc
        lost = missing_labels_after_drop(expected_labels, clips)
        if lost:
            raise SystemExit(
                f"error: every clip of {len(lost)} gesture(s) failed extraction, so they are gone from "
                f"the class set: {lost}. Training would silently solve an easier "
                f"{len(expected_labels) - len(lost)}-class problem under a {len(expected_labels)}-label "
                "name. Fix or re-record those clips."
            )

    # The manifest order is the class-index order for directory mode. Single-signer --video mode
    # intentionally remains self-contained for the legacy three-label demo.
    labels = expected_labels if args.data_dir else label_order(clips)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    targets = [label_to_idx[clip.label] for clip in clips]
    people = sorted({clip.person for clip in clips})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(clips)} clips | {len(labels)} labels | {len(people)} people | device {device}")

    if args.loso:
        folds = []
        for val_person in people:
            train_idx, val_idx = signer_split(clips, val_person)
            print(f"\nFold '{val_person}': train {len(train_idx)} clips / test {len(val_idx)} clips")
            fold_model, history = train_model(
                [sequences[i] for i in train_idx],
                [targets[i] for i in train_idx],
                len(labels),
                args,
                device,
                val_sequences=[sequences[i] for i in val_idx],
                val_targets=[targets[i] for i in val_idx],
                log_prefix=f"  [{val_person}] ",
            )
            final = history[-1]
            # A second pass over the same unshuffled, unaugmented loader: cheap, and it is the only
            # way to attach a prediction to the clip it came from without threading paths through
            # train_model or bloating history with 108 rows per epoch.
            fold_loader = build_eval_loader(
                [sequences[i] for i in val_idx], [targets[i] for i in val_idx], args
            )
            _, _, rows = evaluate(fold_model, fold_loader, device)
            predictions = [
                {
                    "video": str(clips[clip_index].path),
                    "person": clips[clip_index].person,
                    "label": labels[row["target"]],
                    "predicted": labels[row["predicted"]],
                    "confidence": row["confidence"],
                    "correct": row["target"] == row["predicted"],
                }
                for clip_index, row in zip(val_idx, rows, strict=True)
            ]
            folds.append(
                {
                    "held_out_person": val_person,
                    "train_clips": len(train_idx),
                    "test_clips": len(val_idx),
                    "test_accuracy": final["val_accuracy"],
                    "test_loss": final["val_loss_plain_ce"],
                    "predictions": predictions,
                    "history": history,
                }
            )
            print(f"  fold '{val_person}' accuracy {final['val_accuracy']:.1%} on {len(val_idx)} real clips")
            del fold_model
            del fold_loader
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        accuracies = [fold["test_accuracy"] for fold in folds]
        worst = min(folds, key=lambda fold: fold["test_accuracy"])
        # accuracy is correct/n exactly, so round() recovers the integer count.
        total_correct = sum(round(fold["test_accuracy"] * fold["test_clips"]) for fold in folds)
        total_tested = sum(fold["test_clips"] for fold in folds)
        all_predictions = [row for fold in folds for row in fold["predictions"]]
        ranked = worst_labels(all_predictions, limit=5)
        # Validate the prediction/extraction join before publishing a report. A broken join means
        # the diagnostics do not describe the evaluated clips and must leave no success artifact.
        suspects = suspect_clips(all_predictions, extraction_stats, args.min_hand_ratio)
        report = {
            "mode": "loso",
            **run_metadata(args, clips, labels, people, device),
            "macro_mean_accuracy": sum(accuracies) / len(accuracies),
            "pooled_accuracy": total_correct / total_tested,
            "worst_fold": {"person": worst["held_out_person"], "accuracy": worst["test_accuracy"]},
            "folds": folds,
            "extraction": extraction_stats,
            "failed_clips": failures,
            "note": (
                f"Leave-one-signer-out over {len(people)} signers; each fold trains on {len(people) - 1} "
                "signers and tests on the held-out one. Test clips are never augmented, so accuracy is "
                "per real clip. macro_mean_accuracy is the unweighted mean over folds; pooled_accuracy "
                f"is {total_correct}/{total_tested} clips overall (they differ only if folds hold "
                "different clip counts). Epochs were fixed in advance, NOT selected on these folds — "
                "selecting there would tune a hyperparameter on the test signer. Report the mean "
                "together with the worst fold, and state the signer count. train_loss_smoothed uses "
                "label_smoothing=0.03 while val_loss_plain_ce does not; do not read them as one curve. "
                "folds[].predictions holds one row per test clip; per-label accuracy, the worst labels "
                "and the re-record candidates are pure functions of it plus extraction[], so they are "
                "printed to the console rather than stored — a stored copy could drift from its source. "
                "Pair this report with a ship checkpoint only when training_signature matches exactly."
            ),
        }
        report_path = model_dir / "loso_report.json"
        _atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            f"\nLOSO over {len(people)} signers: macro mean {report['macro_mean_accuracy']:.1%}, "
            f"pooled {report['pooled_accuracy']:.1%} ({total_correct}/{total_tested} clips), "
            f"worst fold '{worst['held_out_person']}' {worst['test_accuracy']:.1%}"
        )
        print(f"training_signature {report['training_signature'][:12]}")
        if ranked:
            print(f"\n{len(ranked)} worst label(s) of {len(labels)}:")
            for row in ranked:
                confusion = f", most often read as '{row['confused_with']}'" if row["confused_with"] else ""
                print(f"  {row['label']:24.24s} {row['correct']}/{row['clips']} = {row['accuracy']:.0%}{confusion}")
        else:
            print(f"\nEvery one of the {len(labels)} labels was predicted correctly on every test clip.")

        print(
            f"\n{len(suspects)} clip(s) both predicted wrong AND recorded below "
            f"hand_frame_ratio {args.min_hand_ratio:.2f} — re-record candidates:"
        )
        for row in suspects[:10]:
            print(
                f"  {row['hand_frame_ratio']:.0%} hands | {row['person']}/{row['label']} "
                f"-> '{row['predicted']}' ({row['confidence']:.0%}) | {Path(row['video']).name}"
            )
        if len(suspects) > 10:
            print(f"  ... and {len(suspects) - 10} more (see folds[].predictions)")

        print(f"\nWrote {report_path}. No checkpoint written — this mode measures only.")
        return

    print(f"\nTraining on all {len(clips)} clips for {args.epochs} epochs (no validation, no early stopping)...")
    model, history = train_model(sequences, targets, len(labels), args, device)

    ship_metadata = run_metadata(args, clips, labels, people, device)
    checkpoint_config = {
        "input_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        # Read off the built model rather than restated, so the config can never drift from it.
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "bidirectional": model.bidirectional,
        "pooling": model.pooling,
        "features_version": FEATURES_VERSION,
        "feature_contract": FEATURE_CONTRACT,
        "rejection_contract": REJECTION_CONTRACT,
        "segmentation_contract": SEGMENTATION_CONTRACT,
        "reject_policy": {
            "schema_version": 1,
            "calibrated": False,
            "source": "not-calibrated",
        },
        "recording_plan": getattr(args, "recording_plan_metadata", {"mode": "legacy"}),
        "augmentation_version": AUGMENTATION_VERSION,
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        # This lives inside the same file as the weights. Comparing JSON files alone cannot detect
        # a stale/replaced checkpoint or a run interrupted between artifact writes.
        "training_signature": ship_metadata["training_signature"],
    }
    checkpoint_path = model_dir / "gesture_lstm.pt"
    save_checkpoint(checkpoint_path, model, labels, checkpoint_config)
    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_signature = saved_checkpoint.get("config", {}).get("training_signature")
    if saved_signature != ship_metadata["training_signature"] or list(saved_checkpoint.get("labels", [])) != labels:
        raise RuntimeError(f"Saved checkpoint verification failed for {checkpoint_path}")
    metrics = {
        "mode": "ship",
        "signer_split": False,
        **ship_metadata,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "samples": len(clips) * (args.augment + 1),
        "extraction": extraction_stats,
        "failed_clips": failures,
        "warning": (
            "This checkpoint is trained on EVERY clip listed in extraction[]: it has no holdout and "
            "carries no accuracy number. Any accuracy you report must come from "
            "`vslr-train --data-dir ... --loso` (models/loso_report.json), and that report is only "
            "about this checkpoint only when training_signature matches exactly in loso_report.json, "
            "metrics.json, and gesture_lstm.pt. The signature covers data, labels, hyperparameters, "
            "augmentation/model/training recipe versions, device, and runtime. checkpoint_sha256 binds "
            "metrics.json to the exact weights file. Cross-check test clips against extraction[].video."
        ),
        "history": history,
    }
    _atomic_write_text(model_dir / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(model_dir / "labels.json", json.dumps(labels, ensure_ascii=False, indent=2) + "\n")

    print(f"Saved {checkpoint_path} ({len(labels)} labels, {args.epochs} epochs, all clips)")
    print(
        f"training_signature {metrics['training_signature'][:12]} — must match "
        "loso_report.json to pair its accuracy with this checkpoint."
    )
    report_path = model_dir / "loso_report.json"
    if report_path.exists():
        try:
            loso_signature = json.loads(report_path.read_text(encoding="utf-8")).get("training_signature")
            pair_ok = artifact_pair_matches(loso_signature, metrics, checkpoint_path)
        except Exception as exc:
            # Verification is fail-closed: a concurrent replacement, corrupt checkpoint/report,
            # or read/hash error can suppress PAIR OK but can never manufacture it.
            print(f"WARNING: could not verify {report_path}: {exc}")
        else:
            if pair_ok:
                print(
                    f"PAIR OK: {report_path}, metrics.json, and {checkpoint_path.name} "
                    "carry the same training signature and the checkpoint SHA-256 still matches."
                )
            else:
                print(
                    f"PAIR MISMATCH: {report_path}, metrics.json, and {checkpoint_path.name} "
                    "do not carry one training signature."
                )
    print("This checkpoint has NO accuracy number. Run --loso to get one.")
    print("Next: vslr-camera --no-tts")


if __name__ == "__main__":
    main()

