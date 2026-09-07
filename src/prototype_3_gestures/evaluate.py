from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prototype_3_gestures.prepare_train import file_sha256
from prototype_3_gestures.vsl3.console import configure_utf8_stdio
from prototype_3_gestures.vsl3.features import (
    SEQUENCE_LENGTH,
    HolisticExtractor,
)
from prototype_3_gestures.vsl3.labels import normalise_label
from prototype_3_gestures.vsl3.model import load_checkpoint

configure_utf8_stdio()


@dataclass(frozen=True)
class EvalClip:
    video_path: Path
    signer: str
    ground_truth: str


def load_training_hashes(manifest_path: str | Path) -> set[str]:
    """Extract known SHA-256 hashes from a training manifest CSV."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    hashes = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sha256" not in reader.fieldnames:
            raise ValueError(
                f"Training manifest '{manifest_path}' is missing required 'sha256' column in header"
            )
        for row in reader:
            sha = row.get("sha256")
            if sha and sha.strip():
                hashes.add(sha.strip().lower())
    if not hashes:
        raise ValueError(
            f"No valid SHA-256 hashes found in training manifest '{manifest_path}'"
        )
    return hashes


def extract_sequence_from_video(
    video_path: str | Path,
    target_len: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Extract Holistic landmark sequence using the canonical pipeline."""
    with HolisticExtractor() as extractor:
        sequence, _ = extractor.extract_video(video_path, target_len=target_len)
    return sequence


def parse_video_spec(value: str) -> tuple[str, Path]:
    """Parse LABEL=PATH specification for single-clip testing."""
    if "=" not in value:
        raise ValueError(f"Invalid --video format: '{value}'. Expected LABEL=PATH")
    raw_label, raw_path = value.split("=", 1)
    label = normalise_label(raw_label.strip())
    path = Path(raw_path.strip()).expanduser().resolve()
    if not label:
        raise ValueError("Gesture label cannot be empty")
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a regular file: {path}")
    return label, path


def collect_clips(data_dir: str | Path) -> list[EvalClip]:
    """Collect clips from DIR/<signer>/<gesture>/*.mov layout."""
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"--data-dir is not a directory: {root}")

    clips: list[EvalClip] = []
    video_extensions = {".mov", ".mp4"}

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        signer = person_dir.name
        for label_dir in sorted(person_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            gt_label = normalise_label(label_dir.name)
            for file_path in sorted(label_dir.iterdir()):
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    clips.append(
                        EvalClip(
                            video_path=file_path.resolve(),
                            signer=signer,
                            ground_truth=gt_label,
                        )
                    )
    if not clips:
        raise ValueError(
            f"No .mov/.mp4 clips found under {data_dir}. Expected layout: DIR/<signer>/<gesture>/*.mov"
        )
    return clips


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall and per-label metrics with clean rejection accounting."""
    total_clips = len(predictions)
    if total_clips == 0:
        return {
            "total_clips": 0,
            "top1_correct": 0,
            "top1_accuracy": 0.0,
            "accepted_clips": 0,
            "coverage": 0.0,
            "rejection_rate": 0.0,
            "accepted_correct": 0,
            "accepted_accuracy": 0.0,
            "per_label": {},
        }

    top1_correct = sum(1 for p in predictions if p["correct"])
    accepted_clips = sum(1 for p in predictions if p["accepted"])
    accepted_correct = sum(1 for p in predictions if p["accepted"] and p["correct"])

    top1_accuracy = top1_correct / total_clips
    coverage = accepted_clips / total_clips
    rejection_rate = 1.0 - coverage
    accepted_accuracy = (accepted_correct / accepted_clips) if accepted_clips > 0 else 0.0

    # Per-label metrics
    per_label: dict[str, dict[str, Any]] = {}
    for p in predictions:
        gt = p["ground_truth"]
        lbl_stats = per_label.setdefault(
            gt,
            {"total": 0, "correct": 0, "accepted": 0, "accepted_correct": 0},
        )
        lbl_stats["total"] += 1
        if p["correct"]:
            lbl_stats["correct"] += 1
        if p["accepted"]:
            lbl_stats["accepted"] += 1
            if p["correct"]:
                lbl_stats["accepted_correct"] += 1

    for gt, stats in per_label.items():
        stats["raw_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["coverage"] = stats["accepted"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["accepted_accuracy"] = (
            stats["accepted_correct"] / stats["accepted"] if stats["accepted"] > 0 else 0.0
        )

    return {
        "total_clips": total_clips,
        "top1_correct": top1_correct,
        "top1_accuracy": top1_accuracy,
        "accepted_clips": accepted_clips,
        "coverage": coverage,
        "rejection_rate": rejection_rate,
        "accepted_correct": accepted_correct,
        "accepted_accuracy": accepted_accuracy,
        "per_label": per_label,
    }


def generate_report_markdown(
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    model_path: Path,
    model_sha256: str,
    confidence_threshold: float,
    training_manifest: str | None,
) -> str:
    """Build technical evaluation report markdown."""
    lines = [
        "# BÁO CÁO ĐÁNH GIÁ TẬP KIỂM THỬ ĐỘC LẬP (EXTERNAL EVALUATION)",
        "",
        f"- **Mô hình**: `{model_path.name}`",
        f"- **Checkpoint SHA-256**: `{model_sha256}`",
        f"- **Tập đối chiếu rò rỉ (Training Manifest)**: `{training_manifest or 'Không cung cấp'}`",
        f"- **Ngưỡng tin cậy chấp nhận (Confidence Threshold)**: `{confidence_threshold:.2f}` *(Ngưỡng chẩn đoán / thử nghiệm tạm thời, không phải ngưỡng production)*",
        "",
        "---",
        "",
        "## 1. Kết quả Tổng thể",
        "",
        f"- **Tổng số clips đánh giá**: **{metrics['total_clips']}**",
        f"- **Top-1 Raw Accuracy**: **{metrics['top1_accuracy']:.2%}** ({metrics['top1_correct']} / {metrics['total_clips']} clips đúng)",
        f"- **Tỉ lệ Chấp nhận (Coverage)**: **{metrics['coverage']:.2%}** ({metrics['accepted_clips']} / {metrics['total_clips']} clips)",
        f"- **Tỉ lệ Từ chối (Rejection Rate)**: **{metrics['rejection_rate']:.2%}** ({metrics['total_clips'] - metrics['accepted_clips']} / {metrics['total_clips']} clips bị reject)",
        f"- **Accepted Accuracy**: **{metrics['accepted_accuracy']:.2%}** ({metrics['accepted_correct']} / {metrics['accepted_clips']} clips được chấp nhận đúng)",
        "",
        "---",
        "",
        "## 2. Kết quả Chi tiết theo Từng Cử chỉ",
        "",
        "| Cử chỉ (Ground Truth) | Tổng clips | Raw Correct | Top-1 Accuracy | Accepted | Accepted Accuracy | Coverage |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for label, stats in sorted(metrics.get("per_label", {}).items()):
        lines.append(
            f"| `{label}` | {stats['total']} | {stats['correct']} | "
            f"{stats['raw_accuracy']:.1%} | {stats['accepted']} | "
            f"{stats['accepted_accuracy']:.1%} | {stats['coverage']:.1%} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Danh sách Clips Bị Từ chối hoặc Dự đoán Sai",
        "",
        "| Video | Signer | Ground Truth | Top-1 Pred | Conf | Top-2 | Top-3 | Status |",
        "|:---|:---:|:---|:---|:---:|:---|:---|:---:|",
    ])

    for p in predictions:
        if not p["correct"] or not p["accepted"]:
            status = "REJECTED" if not p["accepted"] else "WRONG"
            t2 = f"{p['top3'][1]['label']} ({p['top3'][1]['confidence']:.2f})" if len(p['top3']) > 1 else "-"
            t3 = f"{p['top3'][2]['label']} ({p['top3'][2]['confidence']:.2f})" if len(p['top3']) > 2 else "-"
            rel_vid = Path(p["video_path"]).name
            lines.append(
                f"| `{rel_vid}` | {p['signer']} | {p['ground_truth']} | "
                f"`{p['predicted']}` | {p['confidence']:.2f} | {t2} | {t3} | **{status}** |"
            )

    return "\n".join(lines) + "\n"


def plot_confusion_matrix(
    predictions: list[dict[str, Any]],
    labels: list[str],
    out_path: Path,
) -> None:
    """Render 24x24 confusion matrix plot to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((n, n), dtype=np.int32)

    for p in predictions:
        gt_idx = label_to_idx.get(p["ground_truth"])
        pred_idx = label_to_idx.get(p["predicted"])
        if gt_idx is not None and pred_idx is not None:
            cm[gt_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=labels,
        yticklabels=labels,
        title="External Evaluation Confusion Matrix",
        ylabel="Ground Truth",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, format(val, "d"),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=9,
                )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_dataset(
    data_dir: str | Path,
    model_path: str | Path,
    training_manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    confidence_threshold: float = 0.50,
    allow_uncalibrated: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Inference-only evaluation across an external test tree."""
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    initial_model_sha256 = file_sha256(model_path)

    # 1. Load model and verify class structure
    model, checkpoint_labels, config = load_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    # Reject policy gate
    reject_policy = config.get("reject_policy", {})
    if not reject_policy.get("calibrated", False) and not allow_uncalibrated:
        raise ValueError(
            "Reject policy is not calibrated. Realtime inference requires calibrated policy. "
            "Use --allow-uncalibrated only for temporary diagnostic/eval purposes."
        )

    # 2. Collect test clips
    clips = collect_clips(data_dir)

    # 3. Label compatibility gate
    checkpoint_labels_norm = [normalise_label(lbl) for lbl in checkpoint_labels]
    checkpoint_labels_set = set(checkpoint_labels_norm)
    for clip in clips:
        if clip.ground_truth not in checkpoint_labels_set:
            raise ValueError(
                f"Clip {clip.video_path} has label '{clip.ground_truth}' not in checkpoint labels: {checkpoint_labels_norm}"
            )

    # 4. Leakage gate
    if training_manifest_path:
        training_hashes = load_training_hashes(training_manifest_path)
        for clip in clips:
            clip_hash = file_sha256(clip.video_path).lower()
            if clip_hash in training_hashes:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: Test clip '{clip.video_path}' (SHA-256: {clip_hash}) "
                    f"overlaps with a clip in training manifest '{training_manifest_path}'!"
                )

    # 5. Run inference (Strictly no_grad)
    predictions = []
    with torch.no_grad():
        for position, clip in enumerate(clips, start=1):
            sequence = extract_sequence_from_video(clip.video_path, target_len=SEQUENCE_LENGTH)
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            k = min(3, len(checkpoint_labels))
            topk = torch.topk(probs, k=k)

            top1_idx = int(topk.indices[0])
            top1_label = checkpoint_labels[top1_idx]
            top1_conf = float(topk.values[0])

            top3 = [
                {"label": checkpoint_labels[int(idx)], "confidence": float(conf)}
                for idx, conf in zip(topk.indices, topk.values)
            ]

            accepted = top1_conf >= confidence_threshold
            correct = top1_label == clip.ground_truth

            predictions.append(
                {
                    "video_path": str(clip.video_path),
                    "signer": clip.signer,
                    "ground_truth": clip.ground_truth,
                    "predicted": top1_label,
                    "confidence": top1_conf,
                    "top3": top3,
                    "accepted": accepted,
                    "correct": correct,
                }
            )

    # Invariant verification: model weights file must not be modified
    after_model_sha256 = file_sha256(model_path)
    if after_model_sha256 != initial_model_sha256:
        raise RuntimeError(
            f"FATAL: Model weights file was altered during evaluation! "
            f"Before: {initial_model_sha256}, After: {after_model_sha256}"
        )

    metrics = compute_metrics(predictions)

    # 6. Save artifacts if output_dir specified
    if output_dir:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write predictions.csv
        csv_path = out_dir / "predictions.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_path", "signer", "ground_truth", "predicted_top1", "confidence",
                "top2_label", "top2_conf", "top3_label", "top3_conf", "accepted", "correct"
            ])
            for p in predictions:
                t2_lbl = p["top3"][1]["label"] if len(p["top3"]) > 1 else ""
                t2_cnf = f"{p['top3'][1]['confidence']:.4f}" if len(p["top3"]) > 1 else ""
                t3_lbl = p["top3"][2]["label"] if len(p["top3"]) > 2 else ""
                t3_cnf = f"{p['top3'][2]['confidence']:.4f}" if len(p["top3"]) > 2 else ""
                writer.writerow([
                    p["video_path"], p["signer"], p["ground_truth"], p["predicted"],
                    f"{p['confidence']:.4f}", t2_lbl, t2_cnf, t3_lbl, t3_cnf,
                    p["accepted"], p["correct"]
                ])

        # Write metrics.json
        full_metrics = {
            "model_path": str(model_path),
            "model_sha256": initial_model_sha256,
            "training_signature": config.get("training_signature"),
            "training_manifest": str(training_manifest_path) if training_manifest_path else None,
            "confidence_threshold": confidence_threshold,
            "summary": metrics,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(full_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Write REPORT.md
        report_md = generate_report_markdown(
            metrics,
            predictions,
            model_path,
            initial_model_sha256,
            confidence_threshold,
            str(training_manifest_path) if training_manifest_path else None,
        )
        (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

        # Write confusion_matrix.png
        plot_confusion_matrix(predictions, checkpoint_labels, out_dir / "confusion_matrix.png")

    return {"metrics": metrics, "predictions": predictions, "model_sha256": initial_model_sha256}


def evaluate_single_clip(
    video_spec: str,
    model_path: str | Path,
    training_manifest_path: str | Path | None = None,
    confidence_threshold: float = 0.50,
    allow_uncalibrated: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Single clip evaluation for ad-hoc / webcam saved videos."""
    ground_truth, video_path = parse_video_spec(video_spec)
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    initial_model_sha256 = file_sha256(model_path)

    # Check leakage
    if training_manifest_path:
        training_hashes = load_training_hashes(training_manifest_path)
        clip_hash = file_sha256(video_path).lower()
        if clip_hash in training_hashes:
            raise ValueError(
                f"DATA LEAKAGE DETECTED: Clip '{video_path}' (SHA-256: {clip_hash}) "
                f"overlaps with training manifest '{training_manifest_path}'!"
            )

    model, checkpoint_labels, config = load_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    if not config.get("reject_policy", {}).get("calibrated", False) and not allow_uncalibrated:
        raise ValueError(
            "Reject policy is not calibrated. Use --allow-uncalibrated only for temporary diagnostic purposes."
        )

    gt_norm = normalise_label(ground_truth)
    labels_norm = [normalise_label(lbl) for lbl in checkpoint_labels]
    if gt_norm not in labels_norm:
        raise ValueError(
            f"Ground truth '{ground_truth}' is not in checkpoint labels: {labels_norm}"
        )

    with torch.no_grad():
        sequence = extract_sequence_from_video(video_path, target_len=SEQUENCE_LENGTH)
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        k = min(3, len(checkpoint_labels))
        topk = torch.topk(probs, k=k)

        top1_idx = int(topk.indices[0])
        top1_label = checkpoint_labels[top1_idx]
        top1_conf = float(topk.values[0])

        top3 = [
            {"label": checkpoint_labels[int(idx)], "confidence": float(conf)}
            for idx, conf in zip(topk.indices, topk.values)
        ]

        accepted = top1_conf >= confidence_threshold
        correct = top1_label == gt_norm

    after_model_sha256 = file_sha256(model_path)
    if after_model_sha256 != initial_model_sha256:
        raise RuntimeError("FATAL: Model weights file was altered during evaluation!")

    return {
        "video_path": str(video_path),
        "ground_truth": gt_norm,
        "predicted": top1_label,
        "confidence": top1_conf,
        "top3": top3,
        "accepted": accepted,
        "correct": correct,
    }


def valid_confidence(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not (0.0 <= number <= 1.0):
        raise argparse.ArgumentTypeError(f"Confidence threshold must be in [0.0, 1.0], got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VSLR External Evaluation Tool (Inference Only, Zero Training, Zero Leakage)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-dir",
        help="Root directory of external evaluation dataset: DIR/<signer>/<gesture>/*.mov",
    )
    group.add_argument(
        "--video",
        help="Single video mode: 'LABEL=PATH' (e.g. 'Xin chào=path/to/clip.mov')",
    )
    parser.add_argument(
        "--model",
        default="models/gesture_lstm.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--training-manifest",
        default=None,
        help="Path to training manifest CSV to prevent data leakage",
    )
    parser.add_argument(
        "--confidence",
        type=valid_confidence,
        default=0.50,
        help="Confidence threshold for acceptance (default 0.50 for temporary diagnostics)",
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="Allow running with uncalibrated reject policy (for diagnostic/demo only)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write predictions.csv, metrics.json, confusion_matrix.png, REPORT.md",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.video:
        res = evaluate_single_clip(
            video_spec=args.video,
            model_path=args.model,
            training_manifest_path=args.training_manifest,
            confidence_threshold=args.confidence,
            allow_uncalibrated=args.allow_uncalibrated,
            device=device,
        )
        status = "CORRECT" if res["correct"] else "WRONG"
        accept_str = "ACCEPTED" if res["accepted"] else "REJECTED"
        print(f"\nEvaluation Result for: {res['video_path']}")
        print(f"  Ground Truth: {res['ground_truth']}")
        print(f"  Predicted:    {res['predicted']} ({res['confidence']:.2%}) [{status}, {accept_str}]")
        print("  Top-3 Predictions:")
        for idx, item in enumerate(res["top3"], start=1):
            print(f"    {idx}. {item['label']}: {item['confidence']:.2%}")
        return

    res = evaluate_dataset(
        data_dir=args.data_dir,
        model_path=args.model,
        training_manifest_path=args.training_manifest,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        allow_uncalibrated=args.allow_uncalibrated,
        device=device,
    )
    m = res["metrics"]
    print("\n==================================================")
    print("VSLR EXTERNAL EVALUATION RESULTS")
    print("==================================================")
    print(f"Total clips evaluated: {m['total_clips']}")
    print(f"Top-1 Raw Accuracy:    {m['top1_accuracy']:.2%} ({m['top1_correct']}/{m['total_clips']})")
    print(f"Coverage:              {m['coverage']:.2%} ({m['accepted_clips']}/{m['total_clips']} accepted)")
    print(f"Rejection Rate:        {m['rejection_rate']:.2%}")
    print(f"Accepted Accuracy:     {m['accepted_accuracy']:.2%} ({m['accepted_correct']}/{m['accepted_clips']})")
    if args.output_dir:
        print(f"\nArtifacts saved to: {args.output_dir}")
        print("  - predictions.csv")
        print("  - metrics.json")
        print("  - REPORT.md")
        print("  - confusion_matrix.png")


if __name__ == "__main__":
    main()
