from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .vsl3.features import FEATURE_DIM, SEQUENCE_LENGTH, HolisticExtractor, augment_sequence
from .vsl3.model import GestureLSTM, save_checkpoint


def parse_video_specs(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
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
        parsed.append((label, path))

    labels = [label for label, _ in parsed]
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) != 3:
        raise ValueError(f"This prototype expects exactly 3 gesture labels, got {len(unique_labels)}")
    counts = {label: labels.count(label) for label in unique_labels}
    if any(count < 2 for count in counts.values()):
        raise ValueError(f"Each gesture needs at least 2 source videos for source-level validation, got {counts}")
    resolved_paths = [path for _, path in parsed]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("The same source video cannot be supplied more than once")
    return parsed


def build_dataset(
    base_sequences: list[np.ndarray],
    base_targets: list[int],
    augmentations_per_video: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(base_sequences) != len(base_targets):
        raise ValueError("base_sequences and base_targets must have the same length")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    targets: list[int] = []
    synthetic_flags: list[bool] = []
    source_ids: list[int] = []
    for source_idx, (original, class_idx) in enumerate(zip(base_sequences, base_targets, strict=True)):
        samples.append(original)
        targets.append(class_idx)
        synthetic_flags.append(False)
        source_ids.append(source_idx)
        for _ in range(augmentations_per_video):
            samples.append(augment_sequence(original, rng))
            targets.append(class_idx)
            synthetic_flags.append(True)
            source_ids.append(source_idx)
    return (
        np.stack(samples).astype(np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(synthetic_flags, dtype=bool),
        np.asarray(source_ids, dtype=np.int64),
    )


def source_holdout_split(y: np.ndarray, source_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) != len(source_ids):
        raise ValueError("y and source_ids must have the same length")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    for class_idx in np.unique(y):
        class_indices = np.flatnonzero(y == class_idx)
        class_sources = np.unique(source_ids[class_indices])
        if len(class_sources) < 2:
            raise ValueError(f"Class {class_idx} needs at least 2 source videos for holdout validation")
        val_source = int(rng.choice(class_sources))
        val_mask = source_ids[class_indices] == val_source
        val.extend(class_indices[val_mask].tolist())
        train.extend(class_indices[~val_mask].tolist())
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 3-gesture MediaPipe + LSTM proof of concept")
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        help="Repeat for every source video using LABEL=PATH. Exactly 3 labels are required; each label needs >=2 videos.",
    )
    parser.add_argument("--model-dir", default="models", help="Directory for model and training metadata")
    parser.add_argument(
        "--dataset-output",
        default="dataset/processed/dataset_3gestures.npz",
        help="Processed landmark dataset path",
    )
    parser.add_argument("--augment", type=int, default=120, help="Synthetic samples generated per video")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = parse_video_specs(args.video)
    labels = list(dict.fromkeys(label for label, _ in specs))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset_output = Path(args.dataset_output)
    dataset_output.parent.mkdir(parents=True, exist_ok=True)

    print("Extracting MediaPipe pose + hand landmarks...")
    base_sequences: list[np.ndarray] = []
    base_targets: list[int] = []
    extraction_stats: list[dict] = []
    with HolisticExtractor() as extractor:
        for label, path in specs:
            sequence, stats = extractor.extract_video(path, SEQUENCE_LENGTH)
            base_sequences.append(sequence)
            base_targets.append(label_to_idx[label])
            stats["label"] = label
            extraction_stats.append(stats)
            print(
                f"  {label}: {stats['sampled_frames']} sampled -> {stats['trimmed_frames']} trimmed frames, "
                f"hands detected {stats['hand_frame_ratio']:.1%}"
            )

    X, y, synthetic, source_ids = build_dataset(base_sequences, base_targets, args.augment, args.seed)
    train_idx, val_idx = source_holdout_split(y, source_ids, args.seed)
    np.savez_compressed(
        dataset_output,
        X=X,
        y=y,
        synthetic=synthetic,
        source_ids=source_ids,
        labels=np.asarray(labels),
        train_idx=train_idx,
        val_idx=val_idx,
    )

    train_dataset = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    val_dataset = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureLSTM(FEATURE_DIM, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    print(f"Training on {device}: {len(train_dataset)} train / {len(val_dataset)} source-holdout samples")
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

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = accuracy(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "holdout_val_loss": val_loss,
                "holdout_val_accuracy": val_acc,
            }
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | train {train_acc:.1%}/{train_loss:.4f} | "
                f"source-holdout {val_acc:.1%}/{val_loss:.4f}"
            )
        if stale_epochs >= 15:
            print(f"Early stop at epoch {epoch}; best source-holdout loss was epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_acc = accuracy(model, val_loader, device)

    final_fit_epochs = max(1, best_epoch)
    torch.manual_seed(args.seed)
    final_model = GestureLSTM(FEATURE_DIM, len(labels)).to(device)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    full_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Final fit on all {len(specs)} source videos for {final_fit_epochs} epochs...")
    for _ in range(final_fit_epochs):
        final_model.train()
        for features, targets in full_loader:
            features = features.to(device)
            targets = targets.to(device)
            final_optimizer.zero_grad(set_to_none=True)
            logits = final_model(features)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=2.0)
            final_optimizer.step()

    checkpoint_config = {
        "input_dim": FEATURE_DIM,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": 96,
        "num_layers": 1,
        "bidirectional": True,
    }
    save_checkpoint(model_dir / "gesture_lstm.pt", final_model, labels, checkpoint_config)
    heldout_sources = sorted(set(int(source_id) for source_id in source_ids[val_idx]))
    metrics = {
        "labels": labels,
        "device": str(device),
        "source_videos": len(specs),
        "augmentations_per_video": args.augment,
        "samples": int(len(X)),
        "train_samples": int(len(train_idx)),
        "holdout_val_samples": int(len(val_idx)),
        "heldout_source_ids": heldout_sources,
        "best_epoch": best_epoch,
        "best_holdout_val_loss": best_val_loss,
        "holdout_val_loss": final_val_loss,
        "holdout_val_accuracy": final_val_acc,
        "final_fit_epochs": final_fit_epochs,
        "extraction": extraction_stats,
        "warning": (
            "Validation holds out one complete source video per class, so augmentations from a held-out clip never enter training. "
            "With only three source clips per class (and likely the same signer/setup), this is still a small prototype estimate, "
            "not a real-world accuracy claim. The saved final checkpoint is then refit on all source videos for camera testing."
        ),
        "history": history,
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {model_dir / 'gesture_lstm.pt'}")
    print(f"Saved dataset: {dataset_output}")
    print(f"Source-holdout accuracy before final refit: {final_val_acc:.1%}")
    print("Saved checkpoint was refit on every supplied source video for the realtime camera demo.")
    print("Next: python -m prototype_3_gestures.realtime")


if __name__ == "__main__":
    main()
