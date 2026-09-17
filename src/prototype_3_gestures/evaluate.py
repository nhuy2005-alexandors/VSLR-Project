from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
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

EVALUATION_ARTIFACT_FILENAMES = (
    "predictions.csv",
    "metrics.json",
    "REPORT.md",
    "confusion_matrix.png",
)
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationGateError(ValueError):
    """Raised when an evaluation gate or integrity constraint is violated."""


@dataclass(frozen=True)
class EvalClip:
    video_path: Path
    signer: str
    ground_truth: str
    video_sha256: str


def is_safe_relative_to(path: Path, base: Path) -> bool:
    """Safely check if path is equal to or located under base across platforms."""
    try:
        path_resolved = path.resolve()
        base_resolved = base.resolve()
        return path_resolved == base_resolved or base_resolved in path_resolved.parents
    except (ValueError, RuntimeError):
        return False


def validate_run_manifest(
    run_manifest_path: str | Path | None,
    training_manifest_path: str | Path,
    model_path: str | Path,
    checkpoint_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate authoritative 3-way binding against RUN_MANIFEST.json before MediaPipe or inference.

    Checks:
      1. SHA-256 of --training-manifest == RUN_MANIFEST.json.dataset_manifest_sha256
      2. SHA-256 of model checkpoint == RUN_MANIFEST.json.checkpoint_sha256
      3. checkpoint['training_signature'] == RUN_MANIFEST.json.training_signature

    Fails closed if any check fails, returning (run_manifest_data, run_manifest_sha256).
    """
    if not run_manifest_path:
        raise EvaluationGateError(
            "Run manifest is required (--run-manifest). "
            "Cannot guarantee model-data binding without authoritative RUN_MANIFEST.json."
        )

    manifest_file = Path(run_manifest_path).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Run manifest file not found: {manifest_file}")
    if not manifest_file.is_file():
        raise EvaluationGateError(f"Run manifest path is not a regular file: {manifest_file}")

    run_manifest_sha = file_sha256(manifest_file)

    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvaluationGateError(f"Failed to parse run manifest JSON at '{manifest_file}': {exc}")

    for req_key in ("dataset_manifest_sha256", "checkpoint_sha256", "training_signature"):
        if req_key not in data or not str(data[req_key]).strip():
            raise EvaluationGateError(
                f"Run manifest '{manifest_file}' is missing or has empty required key '{req_key}'"
            )

    # 1. Compare training manifest SHA-256
    train_file = Path(training_manifest_path).resolve()
    if not train_file.exists():
        raise FileNotFoundError(f"Training manifest not found: {train_file}")
    actual_train_sha = file_sha256(train_file).lower()
    expected_train_sha = str(data["dataset_manifest_sha256"]).strip().lower()
    if actual_train_sha != expected_train_sha:
        raise EvaluationGateError(
            f"Training manifest SHA-256 mismatch with run manifest dataset_manifest_sha256! "
            f"Expected {expected_train_sha}, got {actual_train_sha}"
        )

    # 2. Compare model checkpoint SHA-256
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_file}")
    actual_model_sha = file_sha256(model_file).lower()
    expected_model_sha = str(data["checkpoint_sha256"]).strip().lower()
    if actual_model_sha != expected_model_sha:
        raise EvaluationGateError(
            f"Model checkpoint SHA-256 mismatch with run manifest checkpoint_sha256! "
            f"Expected {expected_model_sha}, got {actual_model_sha}"
        )

    # 3. Compare training_signature
    actual_signature = checkpoint_config.get("training_signature")
    expected_signature = str(data["training_signature"]).strip()
    if not actual_signature or str(actual_signature).strip() != expected_signature:
        raise EvaluationGateError(
            f"Training signature mismatch with run manifest training_signature! "
            f"Expected '{expected_signature}', got '{actual_signature}'"
        )

    return data, run_manifest_sha


def validate_output_dir(
    output_dir: str | Path,
    model_path: Path,
    data_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Enforce strict output directory safety: no models/ pollution, no data collision, no silent clobber."""
    out_path = Path(output_dir).resolve()

    # 1. Block models/ and its subdirectories
    workspace_models = Path("models").resolve()
    if is_safe_relative_to(out_path, workspace_models):
        raise ValueError(
            f"Safety violation: --output-dir cannot be 'models/' or inside it: {out_path}"
        )

    # 2. Block model's own parent directory and its subdirectories
    model_parent = model_path.resolve().parent
    if is_safe_relative_to(out_path, model_parent):
        raise ValueError(
            f"Safety violation: --output-dir cannot be checkpoint parent directory or inside it: {out_path}"
        )

    # 3. Block data directory and its subdirectories
    if data_dir is not None:
        data_resolved = Path(data_dir).resolve()
        if is_safe_relative_to(out_path, data_resolved):
            raise ValueError(
                f"Safety violation: --output-dir cannot be data directory or inside it: {out_path}"
            )

    # 4. Check for existing artifacts (fail-closed unless overwrite=True)
    if out_path.exists() and not overwrite:
        existing = [name for name in EVALUATION_ARTIFACT_FILENAMES if (out_path / name).exists()]
        if existing:
            raise FileExistsError(
                f"--output-dir '{out_path}' already contains evaluation artifacts ({existing}). "
                "Refusing to overwrite without explicit overwrite flag."
            )

    return out_path


def load_and_validate_training_manifest(manifest_path: str | Path | None) -> tuple[set[str], str]:
    """Extract and validate 64-hex SHA-256 hashes from training manifest.

    Fail-closed: manifest is required, non-empty, every row valid hex, zero internal duplicates.
    Returns (set_of_hashes, manifest_file_sha256).
    """
    if not manifest_path:
        raise ValueError(
            "Training manifest is required (--training-manifest). "
            "Cannot guarantee zero data leakage without an authoritative training manifest."
        )

    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.exists():
        raise FileNotFoundError(f"Training manifest not found: {manifest_file}")

    manifest_sha256 = file_sha256(manifest_file)
    hashes: set[str] = set()

    with open(manifest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "sha256" not in reader.fieldnames:
            raise ValueError(
                f"Training manifest '{manifest_file}' is missing required 'sha256' column in header"
            )
        for row_idx, row in enumerate(reader, start=2):
            raw_sha = row.get("sha256")
            if not raw_sha or not raw_sha.strip():
                raise ValueError(
                    f"Training manifest '{manifest_file}' row {row_idx}: empty sha256 field"
                )
            clean_sha = raw_sha.strip().lower()
            if not SHA256_HEX_PATTERN.match(clean_sha):
                raise ValueError(
                    f"Training manifest '{manifest_file}' row {row_idx}: "
                    f"invalid sha256 '{raw_sha}' (must be exactly 64 hexadecimal characters)"
                )
            if clean_sha in hashes:
                raise ValueError(
                    f"Training manifest '{manifest_file}' row {row_idx}: "
                    f"duplicate SHA-256 detected: {clean_sha}"
                )
            hashes.add(clean_sha)

    if not hashes:
        raise ValueError(
            f"No valid SHA-256 hashes found in training manifest '{manifest_file}'"
        )

    return hashes, manifest_sha256


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


def collect_and_validate_clips(
    data_dir: str | Path,
    canonical_labels: list[str],
    expected_signers: list[str] | None = None,
    clips_per_label: int | None = 2,
) -> list[EvalClip]:
    """Collect clips from DIR/<signer>/<gesture>/*.mov layout with strict completeness gates."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"--data-dir is not a directory: {root}")

    # Canonical labels normalized to NFC
    canonical_nfc = [unicodedata.normalize("NFC", lbl) for lbl in canonical_labels]
    canonical_set = set(canonical_nfc)
    if len(canonical_set) != len(canonical_nfc):
        raise ValueError(f"Checkpoint labels contain duplicates after NFC normalization: {canonical_labels}")

    signer_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not signer_dirs:
        raise ValueError(f"No signer directories found under {data_dir}")

    actual_signers = [p.name for p in signer_dirs]

    # Check expected signers
    if expected_signers:
        expected_set = set(expected_signers)
        actual_set = set(actual_signers)
        if actual_set != expected_set:
            unexpected = sorted(actual_set - expected_set)
            missing = sorted(expected_set - actual_set)
            errors = []
            if unexpected:
                errors.append(f"unexpected signers: {unexpected}")
            if missing:
                errors.append(f"missing expected signers: {missing}")
            raise ValueError(f"Signer validation failed: {'; '.join(errors)}")

    video_extensions = {".mov", ".mp4"}
    all_clips: list[EvalClip] = []

    for person_dir in signer_dirs:
        signer = person_dir.name

        # Map directory names normalized to NFC
        label_dirs: dict[str, Path] = {}
        for child in person_dir.iterdir():
            if child.is_dir():
                nfc_name = unicodedata.normalize("NFC", child.name)
                if nfc_name in label_dirs:
                    raise ValueError(
                        f"Signer {signer} has duplicate directory names after NFC normalization: "
                        f"'{child.name}' collides with '{label_dirs[nfc_name].name}'"
                    )
                label_dirs[nfc_name] = child

        # Check label completeness
        missing_labels = [lbl for lbl in canonical_nfc if lbl not in label_dirs]
        if missing_labels:
            raise ValueError(
                f"Signer {signer} missing labels: {missing_labels}. Expected exactly {len(canonical_nfc)} labels."
            )

        unexpected_labels = [lbl for lbl in label_dirs if lbl not in canonical_set]
        if unexpected_labels:
            raise ValueError(
                f"Signer {signer} has unexpected label directories not in checkpoint labels: {unexpected_labels}"
            )

        # Check clip counts per label
        for lbl in canonical_nfc:
            folder = label_dirs[lbl]
            files = [
                f for f in sorted(folder.iterdir())
                if f.is_file() and f.suffix.lower() in video_extensions
            ]
            if clips_per_label is not None and len(files) != clips_per_label:
                raise ValueError(
                    f"Signer {signer} / '{lbl}': expected exactly {clips_per_label} clips, found {len(files)}"
                )
            for f in files:
                sha = file_sha256(f).lower()
                all_clips.append(
                    EvalClip(
                        video_path=f.resolve(),
                        signer=signer,
                        ground_truth=lbl,
                        video_sha256=sha,
                    )
                )

    if not all_clips:
        raise ValueError(f"No .mov/.mp4 clips found under {data_dir}")

    # Test-set internal duplicate gate: no two clips may share identical bytes
    seen_hashes: dict[str, Path] = {}
    for clip in all_clips:
        if clip.video_sha256 in seen_hashes:
            dup_path = seen_hashes[clip.video_sha256]
            raise ValueError(
                f"DUPLICATE TEST CLIPS DETECTED: Multiple test clips have identical SHA-256 byte content: "
                f"'{clip.video_path}' <-> '{dup_path}' ({clip.video_sha256}). "
                "Cannot evaluate duplicate test clips."
            )
        seen_hashes[clip.video_sha256] = clip.video_path

    return all_clips


def compute_test_set_fingerprint(clips: list[EvalClip]) -> str:
    """Deterministic SHA-256 fingerprint identifying the exact test clip set."""
    sorted_clips = sorted(clips, key=lambda c: (c.signer, c.ground_truth, c.video_path.as_posix()))
    rows = [f"{c.signer}|{c.ground_truth}|{c.video_path.name}|{c.video_sha256}" for c in sorted_clips]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


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
    training_manifest_path: str,
    training_manifest_sha256: str,
    run_manifest_path: str,
    run_manifest_sha256: str,
    test_set_fingerprint: str,
    signers: list[str],
) -> str:
    """Build technical evaluation report markdown."""
    lines = [
        "# BÁO CÁO ĐÁNH GIÁ TẬP KIỂM THỬ ĐỘC LẬP (EXTERNAL EVALUATION)",
        "",
        f"- **Mô hình**: `{model_path.name}`",
        f"- **Checkpoint SHA-256**: `{model_sha256}`",
        f"- **Tập đối chiếu rò rỉ (Training Manifest)**: `{training_manifest_path}`",
        f"- **Manifest SHA-256**: `{training_manifest_sha256}`",
        f"- **Hồ sơ huấn luyện (Run Manifest)**: `{run_manifest_path}`",
        f"- **Run Manifest SHA-256**: `{run_manifest_sha256}`",
        f"- **Test Set Fingerprint**: `{test_set_fingerprint}`",
        f"- **Người ký đánh giá ({len(signers)})**: {', '.join(f'`{s}`' for s in signers)}",
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
                f"`{p['predicted_top1']}` | {p['confidence']:.2f} | {t2} | {t3} | **{status}** |"
            )

    return "\n".join(lines) + "\n"


def plot_confusion_matrix(
    predictions: list[dict[str, Any]],
    labels: list[str],
    out_path: Path,
) -> None:
    """Render confusion matrix plot to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    label_to_idx = {unicodedata.normalize("NFC", lbl): i for i, lbl in enumerate(labels)}
    cm = np.zeros((n, n), dtype=np.int32)

    for p in predictions:
        gt_idx = label_to_idx.get(unicodedata.normalize("NFC", p["ground_truth"]))
        pred_idx = label_to_idx.get(unicodedata.normalize("NFC", p["predicted_top1"]))
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
    training_manifest_path: str | Path,
    run_manifest_path: str | Path,
    expected_signers: list[str] | None = None,
    clips_per_label: int = 2,
    output_dir: str | Path | None = None,
    confidence_threshold: float = 0.50,
    allow_uncalibrated: bool = False,
    overwrite: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Inference-only evaluation across an external test tree."""
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    initial_model_sha256 = file_sha256(model_path)

    # 1. Output safety gate (before any computation)
    resolved_out_dir = None
    if output_dir:
        resolved_out_dir = validate_output_dir(
            output_dir=output_dir,
            model_path=model_path,
            data_dir=Path(data_dir),
            overwrite=overwrite,
        )

    # 2. Directory mode constraints
    if not expected_signers:
        raise EvaluationGateError(
            "--expected-signer is mandatory in directory mode. Specify at least one signer (e.g. --expected-signer P05)."
        )
    if clips_per_label is None or clips_per_label <= 0:
        raise EvaluationGateError(
            f"--clips-per-label must be a positive integer, got {clips_per_label}"
        )

    # 3. Mandatory training manifest validation
    training_hashes, manifest_sha256 = load_and_validate_training_manifest(training_manifest_path)

    # 4. Load model and verify class structure
    model, checkpoint_labels, config = load_checkpoint(model_path)

    # 5. Run manifest 3-way binding gate (Before MediaPipe or inference)
    run_manifest_data, run_manifest_sha = validate_run_manifest(
        run_manifest_path=run_manifest_path,
        training_manifest_path=training_manifest_path,
        model_path=model_path,
        checkpoint_config=config,
    )

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Reject policy gate
    reject_policy = config.get("reject_policy", {})
    is_calibrated = bool(reject_policy.get("calibrated", False))
    if not is_calibrated and not allow_uncalibrated:
        raise EvaluationGateError(
            "Reject policy is not calibrated. Realtime inference requires calibrated policy. "
            "Use --allow-uncalibrated only for temporary diagnostic/eval purposes."
        )

    # 6. Collect and validate test clips (Completeness Gate + Test Duplicate Gate)
    canonical_labels = [unicodedata.normalize("NFC", lbl) for lbl in checkpoint_labels]
    clips = collect_and_validate_clips(
        data_dir=data_dir,
        canonical_labels=canonical_labels,
        expected_signers=expected_signers,
        clips_per_label=clips_per_label,
    )

    # 7. Leakage Gate: check against training hashes (All before inference)
    for clip in clips:
        if clip.video_sha256 in training_hashes:
            raise EvaluationGateError(
                f"DATA LEAKAGE DETECTED: Test clip '{clip.video_path}' (SHA-256: {clip.video_sha256}) "
                f"overlaps with a clip in training manifest '{training_manifest_path}'!"
            )

    test_set_fingerprint = compute_test_set_fingerprint(clips)
    signers_found = sorted(set(clip.signer for clip in clips))

    predictions = []
    try:
        # 6. Run inference (Strictly inference_mode, zero gradient)
        with torch.inference_mode():
            for position, clip in enumerate(clips, start=1):
                sequence = extract_sequence_from_video(clip.video_path, target_len=SEQUENCE_LENGTH)
                tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)

                k = min(3, len(checkpoint_labels))
                topk = torch.topk(probs, k=k)

                top1_idx = int(topk.indices[0])
                top1_label_nfc = canonical_labels[top1_idx]
                top1_conf = float(topk.values[0])

                top3 = [
                    {"label": canonical_labels[int(idx)], "confidence": float(conf)}
                    for idx, conf in zip(topk.indices, topk.values)
                ]

                accepted = top1_conf >= confidence_threshold
                correct = top1_label_nfc == clip.ground_truth

                predictions.append(
                    {
                        "video_path": str(clip.video_path),
                        "video_sha256": clip.video_sha256,
                        "model_sha256": initial_model_sha256,
                        "signer": clip.signer,
                        "ground_truth": clip.ground_truth,
                        "predicted_top1": top1_label_nfc,
                        "confidence": top1_conf,
                        "top3": top3,
                        "accepted": accepted,
                        "correct": correct,
                    }
                )
    finally:
        # Invariant verification: model weights file must NEVER be altered under any circumstances
        after_model_sha256 = file_sha256(model_path)
        if after_model_sha256 != initial_model_sha256:
            raise RuntimeError(
                f"FATAL: Model weights file was altered during evaluation! "
                f"Before: {initial_model_sha256}, After: {after_model_sha256}"
            )

    metrics = compute_metrics(predictions)

    # 7. Save artifacts if output_dir specified
    if resolved_out_dir:
        resolved_out_dir.mkdir(parents=True, exist_ok=True)

        # Write predictions.csv
        csv_path = resolved_out_dir / "predictions.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_path", "video_sha256", "model_sha256", "signer",
                "ground_truth", "predicted_top1", "confidence",
                "top2_label", "top2_conf", "top3_label", "top3_conf",
                "accepted", "correct"
            ])
            for p in predictions:
                t2_lbl = p["top3"][1]["label"] if len(p["top3"]) > 1 else ""
                t2_cnf = f"{p['top3'][1]['confidence']:.4f}" if len(p["top3"]) > 1 else ""
                t3_lbl = p["top3"][2]["label"] if len(p["top3"]) > 2 else ""
                t3_cnf = f"{p['top3'][2]['confidence']:.4f}" if len(p["top3"]) > 2 else ""
                writer.writerow([
                    p["video_path"], p["video_sha256"], p["model_sha256"], p["signer"],
                    p["ground_truth"], p["predicted_top1"], f"{p['confidence']:.4f}",
                    t2_lbl, t2_cnf, t3_lbl, t3_cnf, p["accepted"], p["correct"]
                ])

        # Write metrics.json
        full_metrics = {
            "num_signers": len(signers_found),
            "signers": signers_found,
            "label_count": len(canonical_labels),
            "clips_per_label": clips_per_label,
            "total_clips": len(clips),
            "test_set_fingerprint": test_set_fingerprint,
            "training_manifest_path": str(training_manifest_path),
            "training_manifest_sha256": manifest_sha256,
            "run_manifest_path": str(run_manifest_path),
            "run_manifest_sha256": run_manifest_sha,
            "model_path": str(model_path),
            "model_sha256": initial_model_sha256,
            "training_signature": config.get("training_signature"),
            "confidence_threshold": confidence_threshold,
            "reject_policy_calibrated": is_calibrated,
            "summary": metrics,
        }
        (resolved_out_dir / "metrics.json").write_text(
            json.dumps(full_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Write REPORT.md
        report_md = generate_report_markdown(
            metrics=metrics,
            predictions=predictions,
            model_path=model_path,
            model_sha256=initial_model_sha256,
            confidence_threshold=confidence_threshold,
            training_manifest_path=str(training_manifest_path),
            training_manifest_sha256=manifest_sha256,
            run_manifest_path=str(run_manifest_path),
            run_manifest_sha256=run_manifest_sha,
            test_set_fingerprint=test_set_fingerprint,
            signers=signers_found,
        )
        (resolved_out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

        # Write confusion_matrix.png
        plot_confusion_matrix(predictions, canonical_labels, resolved_out_dir / "confusion_matrix.png")

    return {
        "metrics": metrics,
        "predictions": predictions,
        "model_sha256": initial_model_sha256,
        "test_set_fingerprint": test_set_fingerprint,
    }


def evaluate_single_clip(
    video_spec: str,
    model_path: str | Path,
    training_manifest_path: str | Path,
    run_manifest_path: str | Path,
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
    clip_hash = file_sha256(video_path).lower()

    # Mandatory training manifest check
    training_hashes, manifest_sha256 = load_and_validate_training_manifest(training_manifest_path)

    model, checkpoint_labels, config = load_checkpoint(model_path)

    # Mandatory run manifest 3-way binding gate (Before MediaPipe or inference)
    run_manifest_data, run_manifest_sha = validate_run_manifest(
        run_manifest_path=run_manifest_path,
        training_manifest_path=training_manifest_path,
        model_path=model_path,
        checkpoint_config=config,
    )

    if clip_hash in training_hashes:
        raise EvaluationGateError(
            f"DATA LEAKAGE DETECTED: Clip '{video_path}' (SHA-256: {clip_hash}) "
            f"overlaps with training manifest '{training_manifest_path}'!"
        )

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    if not config.get("reject_policy", {}).get("calibrated", False) and not allow_uncalibrated:
        raise EvaluationGateError(
            "Reject policy is not calibrated. Use --allow-uncalibrated only for temporary diagnostic purposes."
        )

    gt_norm = unicodedata.normalize("NFC", ground_truth)
    canonical_labels = [unicodedata.normalize("NFC", lbl) for lbl in checkpoint_labels]
    if gt_norm not in set(canonical_labels):
        raise EvaluationGateError(
            f"Ground truth '{ground_truth}' is not in checkpoint labels: {canonical_labels}"
        )

    try:
        with torch.inference_mode():
            sequence = extract_sequence_from_video(video_path, target_len=SEQUENCE_LENGTH)
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

            k = min(3, len(canonical_labels))
            topk = torch.topk(probs, k=k)

            top1_idx = int(topk.indices[0])
            top1_label = canonical_labels[top1_idx]
            top1_conf = float(topk.values[0])

            top3 = [
                {"label": canonical_labels[int(idx)], "confidence": float(conf)}
                for idx, conf in zip(topk.indices, topk.values)
            ]

            accepted = top1_conf >= confidence_threshold
            correct = top1_label == gt_norm
    finally:
        after_model_sha256 = file_sha256(model_path)
        if after_model_sha256 != initial_model_sha256:
            raise RuntimeError("FATAL: Model weights file was altered during evaluation!")

    return {
        "video_path": str(video_path),
        "video_sha256": clip_hash,
        "model_sha256": initial_model_sha256,
        "ground_truth": gt_norm,
        "predicted_top1": top1_label,
        "confidence": top1_conf,
        "top3": top3,
        "accepted": accepted,
        "correct": correct,
    }


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"Must be a positive integer (>= 1), got {number}")
    return number


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
        required=True,
        help="Path to training manifest CSV to prevent data leakage (mandatory)",
    )
    parser.add_argument(
        "--run-manifest",
        required=True,
        help="Path to RUN_MANIFEST.json binding training manifest, checkpoint, and signature (mandatory)",
    )
    parser.add_argument(
        "--expected-signer",
        action="append",
        dest="expected_signers",
        default=None,
        help="Expected signer directory name (mandatory in directory mode, e.g. --expected-signer P05)",
    )
    parser.add_argument(
        "--clips-per-label",
        type=positive_int,
        default=2,
        help="Exact expected clips per label directory in directory mode (positive integer, default: 2)",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing evaluation artifacts in --output-dir",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.data_dir and not args.expected_signers:
            raise EvaluationGateError(
                "--expected-signer is mandatory in directory mode. Specify at least one signer (e.g. --expected-signer P05)."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.video:
            if args.output_dir:
                print("Note: --output-dir is ignored in single --video mode.")
            res = evaluate_single_clip(
                video_spec=args.video,
                model_path=args.model,
                training_manifest_path=args.training_manifest,
                run_manifest_path=args.run_manifest,
                confidence_threshold=args.confidence,
                allow_uncalibrated=args.allow_uncalibrated,
                device=device,
            )
            status = "CORRECT" if res["correct"] else "WRONG"
            accept_str = "ACCEPTED" if res["accepted"] else "REJECTED"
            print(f"\nEvaluation Result for: {res['video_path']}")
            print(f"  SHA-256:      {res['video_sha256']}")
            print(f"  Ground Truth: {res['ground_truth']}")
            print(f"  Predicted:    {res['predicted_top1']} ({res['confidence']:.2%}) [{status}, {accept_str}]")
            print("  Top-3 Predictions:")
            for idx, item in enumerate(res["top3"], start=1):
                print(f"    {idx}. {item['label']}: {item['confidence']:.2%}")
            return

        res = evaluate_dataset(
            data_dir=args.data_dir,
            model_path=args.model,
            training_manifest_path=args.training_manifest,
            run_manifest_path=args.run_manifest,
            expected_signers=args.expected_signers,
            clips_per_label=args.clips_per_label,
            output_dir=args.output_dir,
            confidence_threshold=args.confidence,
            allow_uncalibrated=args.allow_uncalibrated,
            overwrite=args.overwrite,
            device=device,
        )
        m = res["metrics"]
        print("\n==================================================")
        print("VSLR EXTERNAL EVALUATION RESULTS")
        print("==================================================")
        print(f"Total clips evaluated: {m['total_clips']}")
        print(f"Test Set Fingerprint:  {res['test_set_fingerprint']}")
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

    except (EvaluationGateError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
