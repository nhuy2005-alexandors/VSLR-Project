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

from .vsl3.features import (
    FEATURE_DIM,
    FEATURES_VERSION,
    SEQUENCE_LENGTH,
    HolisticExtractor,
    augment_sequence,
)
from .vsl3.model import POOLING_FWD_LAST_BWD_FIRST, GestureLSTM, save_checkpoint

SINGLE_SIGNER = "unknown"
VIDEO_SUFFIXES = {".mov", ".mp4"}
MAX_FAILED_CLIP_RATIO = 0.05


@dataclass(frozen=True)
class Clip:
    label: str
    person: str
    path: Path


def discover_clips(data_dir: str | Path) -> list[Clip]:
    """Scan DIR/<person>/<gesture>/*.mov|*.mp4.

    The gesture directory name IS the label and goes straight into labels.json and the spoken
    sentence, so those directories must carry the human-readable Vietnamese name.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"--data-dir is not a directory: {data_dir}")

    clips: list[Clip] = []
    for person_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in person_dir.iterdir() if p.is_dir()):
            for path in sorted(label_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                    clips.append(Clip(label=label_dir.name, person=person_dir.name, path=path.resolve()))

    if not clips:
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
        label = label.strip()
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
    payload = "|".join(
        (
            Path(path).as_posix(),
            f"{float(mtime):.6f}",
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
    payload = "\n".join(
        f"{clip.person}|{clip.label}|{clip.path.as_posix()}|{clip.path.stat().st_mtime:.6f}"
        for clip in sorted(clips, key=lambda c: (c.person, c.label, c.path.as_posix()))
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def run_metadata(args, clips: list[Clip], labels: list[str], people: list[str], device) -> dict:
    """Everything needed to reproduce a run, written into both report kinds."""
    return {
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
        "device": str(device),
    }


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
            )
        staging.replace(cached)
    except OSError as exc:
        warnings.warn(f"Could not cache landmarks for {Path(stats['video']).name}: {exc}", stacklevel=2)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def extract_with_cache(clip: Clip, extractor, cache_dir: Path) -> tuple[np.ndarray, dict]:
    key = cache_key(clip.path, clip.path.stat().st_mtime)
    cached = cache_dir / f"{key}.npz"

    if cached.exists():
        try:
            with np.load(cached, allow_pickle=False) as data:
                sequence = data["sequence"].astype(np.float32)
                stats = {
                    "video": str(clip.path),
                    "label": clip.label,
                    "person": clip.person,
                    "sampled_frames": int(data["sampled_frames"]),
                    "trimmed_frames": int(data["trimmed_frames"]),
                    "hand_frame_ratio": float(data["hand_frame_ratio"]),
                    "from_cache": True,
                }
            return sequence, stats
        except Exception as exc:  # truncated / corrupt / unreadable entry
            # Must not be reported as a bad recording, and must not poison this clip forever:
            # drop the entry and fall through to a real extraction.
            warnings.warn(
                f"Discarding unreadable cache entry for {clip.path.name} ({exc}); re-extracting.",
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


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            total_loss += float(criterion(logits, targets)) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
            total += len(targets)
    return total_loss / max(total, 1), correct / max(total, 1)


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
        val_dataset = GestureDataset(val_sequences, val_targets or [], 0, args.seed)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = GestureLSTM(FEATURE_DIM, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

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
            val_loss, val_acc = accuracy(model, val_loader, device)
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
            except Exception as exc:  # one unusable clip must not abort a 540-clip run
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MediaPipe + BiLSTM gesture recogniser")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-dir",
        help="Root of DIR/<person>/<gesture>/*.mov. Gesture directory names are used verbatim as labels.",
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
    parser.add_argument("--model-dir", default="models", help="Directory for model and training metadata")
    parser.add_argument(
        "--cache-dir",
        default="dataset/processed/landmark_cache",
        help="Landmark cache; keyed on (path, mtime, FEATURES_VERSION)",
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=non_negative_int, default=42)
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

    try:
        clips = discover_clips(args.data_dir) if args.data_dir else parse_video_specs(args.video)
        validate_clips(clips, require_signer_split=args.loso)
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    # Snapshot before extraction can drop anything. A label whose every clip fails would otherwise
    # disappear from the class set entirely and go unnoticed by the post-drop re-validation, which
    # only sees survivors — turning a 27-class problem into an easier 26-class one under the same name.
    expected_labels = label_order(clips)

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

    labels = label_order(clips)
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
            _, history = train_model(
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
            folds.append(
                {
                    "held_out_person": val_person,
                    "train_clips": len(train_idx),
                    "test_clips": len(val_idx),
                    "test_accuracy": final["val_accuracy"],
                    "test_loss": final["val_loss_plain_ce"],
                    "history": history,
                }
            )
            print(f"  fold '{val_person}' accuracy {final['val_accuracy']:.1%} on {len(val_idx)} real clips")

        accuracies = [fold["test_accuracy"] for fold in folds]
        worst = min(folds, key=lambda fold: fold["test_accuracy"])
        # accuracy is correct/n exactly, so round() recovers the integer count.
        total_correct = sum(round(fold["test_accuracy"] * fold["test_clips"]) for fold in folds)
        total_tested = sum(fold["test_clips"] for fold in folds)
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
                "label_smoothing=0.03 while val_loss_plain_ce does not; do not read them as one curve."
            ),
        }
        report_path = model_dir / "loso_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"\nLOSO over {len(people)} signers: macro mean {report['macro_mean_accuracy']:.1%}, "
            f"pooled {report['pooled_accuracy']:.1%} ({total_correct}/{total_tested} clips), "
            f"worst fold '{worst['held_out_person']}' {worst['test_accuracy']:.1%}"
        )
        print(f"Wrote {report_path}. No checkpoint written — this mode measures only.")
        return

    print(f"\nTraining on all {len(clips)} clips for {args.epochs} epochs (no validation, no early stopping)...")
    model, history = train_model(sequences, targets, len(labels), args, device)

    checkpoint_config = {
        "input_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        # Read off the built model rather than restated, so the config can never drift from it.
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "bidirectional": model.bidirectional,
        "pooling": model.pooling,
        "features_version": FEATURES_VERSION,
    }
    save_checkpoint(model_dir / "gesture_lstm.pt", model, labels, checkpoint_config)
    metrics = {
        "mode": "ship",
        "signer_split": False,
        **run_metadata(args, clips, labels, people, device),
        "samples": len(clips) * (args.augment + 1),
        "extraction": extraction_stats,
        "failed_clips": failures,
        "warning": (
            "This checkpoint is trained on EVERY clip listed in extraction[]: it has no holdout and "
            "carries no accuracy number. Any accuracy you report must come from "
            "`vslr-train --data-dir ... --loso` (models/loso_report.json), and that report is only "
            "about this checkpoint if its data_fingerprint, epochs, seed and features_version match "
            "the ones here. Cross-check the test clips against extraction[].video either way."
        ),
        "history": history,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {model_dir / 'gesture_lstm.pt'} ({len(labels)} labels, {args.epochs} epochs, all clips)")
    print(f"data_fingerprint {metrics['data_fingerprint'][:12]} — must match loso_report.json to pair them.")
    print("This checkpoint has NO accuracy number. Run --loso to get one.")
    print("Next: vslr-camera --no-tts")


if __name__ == "__main__":
    main()

