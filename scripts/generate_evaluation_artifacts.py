"""Generate evaluation charts and markdown summary from loso_report.json and metrics.json."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    loso_path = run_dir / "loso_report.json"
    metrics_path = run_dir / "metrics.json"

    if not loso_path.exists():
        print(f"Error: {loso_path} does not exist", file=sys.stderr)
        sys.exit(1)

    with open(loso_path, encoding="utf-8") as f:
        report = json.load(f)

    labels = report["labels"]
    folds = report["folds"]
    num_labels = len(labels)
    label_to_idx = {name: i for i, name in enumerate(labels)}

    # 1. Plot Loss & Accuracy Curves per Fold
    fig, axes = plt.subplots(len(folds), 2, figsize=(14, 4 * len(folds)), squeeze=False)
    for i, fold in enumerate(folds):
        person = fold["held_out_person"]
        hist = fold["history"]
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss_smoothed"] for h in hist]
        val_loss = [h["val_loss_plain_ce"] for h in hist]
        train_acc = [h["train_accuracy"] * 100 for h in hist]
        val_acc = [h["val_accuracy"] * 100 for h in hist]

        # Loss
        ax_loss = axes[i][0]
        ax_loss.plot(epochs, train_loss, label="Train Loss (smoothed)", color="#1f77b4", linewidth=2)
        ax_loss.plot(epochs, val_loss, label="Val Loss (plain CE)", color="#ff7f0e", linewidth=2, linestyle="--")
        ax_loss.set_title(f"Fold {person} (Held-out: {person}) — Loss", fontsize=12, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle=":", alpha=0.6)
        ax_loss.legend()

        # Accuracy
        ax_acc = axes[i][1]
        ax_acc.plot(epochs, train_acc, label="Train Acc", color="#2ca02c", linewidth=2)
        ax_acc.plot(epochs, val_acc, label="Val Acc (Held-out)", color="#d62728", linewidth=2, linestyle="--")
        final_val = fold["test_accuracy"] * 100
        ax_acc.set_title(f"Fold {person} — Accuracy (Final: {final_val:.1f}%)", fontsize=12, fontweight="bold")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(-5, 105)
        ax_acc.grid(True, linestyle=":", alpha=0.6)
        ax_acc.legend()

    plt.tight_layout()
    training_curves_path = run_dir / "training_curves.png"
    plt.savefig(training_curves_path, dpi=200)
    plt.close()
    print(f"Saved: {training_curves_path}")

    # 2. Confusion Matrix across all predictions
    all_predictions = [pred for fold in folds for pred in fold["predictions"]]
    cm = np.zeros((num_labels, num_labels), dtype=int)
    for p in all_predictions:
        target_idx = label_to_idx.get(p["label"], -1)
        pred_idx = label_to_idx.get(p["predicted"], -1)
        if target_idx >= 0 and pred_idx >= 0:
            cm[target_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(num_labels))
    ax.set_yticks(range(num_labels))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — LOSO {len(folds)}-Folds ({len(all_predictions)} clips)", fontsize=14, fontweight="bold")

    # Annotate numbers
    for r in range(num_labels):
        for c in range(num_labels):
            val = cm[r, c]
            if val > 0:
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    cm_path = run_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

    # 3. Per-label Accuracy Bar Chart
    per_label_counts: dict[str, dict[str, int]] = {}
    for p in all_predictions:
        lbl = p["label"]
        if lbl not in per_label_counts:
            per_label_counts[lbl] = {"total": 0, "correct": 0}
        per_label_counts[lbl]["total"] += 1
        if p["correct"]:
            per_label_counts[lbl]["correct"] += 1

    sorted_labels = sorted(
        per_label_counts.keys(),
        key=lambda l: (per_label_counts[l]["correct"] / per_label_counts[l]["total"], l),
    )
    accs = [per_label_counts[l]["correct"] / per_label_counts[l]["total"] * 100 for l in sorted_labels]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#d62728" if a < 50 else ("#ff7f0e" if a < 80 else "#2ca02c") for a in accs]
    bars = ax.barh(sorted_labels, accs, color=colors, height=0.65)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Gesture Accuracy (Ranked Worst to Best)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    for bar, a in zip(bars, accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{a:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    per_label_path = run_dir / "per_label_accuracy.png"
    plt.savefig(per_label_path, dpi=200)
    plt.close()
    print(f"Saved: {per_label_path}")

    # 4. Write Markdown Evaluation Report
    report_md_path = run_dir / "evaluation_report.md"
    macro_mean = report.get("macro_mean_accuracy", 0.0) * 100
    pooled = report.get("pooled_accuracy", 0.0) * 100
    worst = report.get("worst_fold", {})
    worst_person = worst.get("person", "-")
    worst_acc = worst.get("accuracy", 0.0) * 100
    total_correct = sum(1 for p in all_predictions if p["correct"])
    total_tested = len(all_predictions)

    from collections import Counter
    confusion_pairs = Counter()
    for p in all_predictions:
        if not p["correct"]:
            confusion_pairs[(p["label"], p["predicted"])] += 1

    lines = [
        f"# Tóm tắt Đánh giá Thử nghiệm LOSO {len(folds)}-Fold",
        f"",
        f"- **Thời gian đánh giá**: {report.get('created_at', report.get('timestamp', '-'))}",
        f"- **Tổng số clip test**: {total_tested} clips ({len(folds)} người ký x {num_labels} cử chỉ x 6 clips)",
        f"- **Số clip đoán đúng**: {total_correct} / {total_tested}",
        f"- **Pooled Accuracy**: {pooled:.2f}%",
        f"- **Macro Mean Accuracy**: {macro_mean:.2f}%",
        f"- **Fold thấp nhất**: `{worst_person}` ({worst_acc:.2f}%)",
        f"- **Trạng thái External Review**: PENDING",
        f"",
        f"## Kết quả từng Fold",
        f"",
        f"| Fold (Người ký kiểm thử) | Số clip test | Số clip đúng | Accuracy | Val Loss (CE) |",
        f"|---|:---:|:---:|:---:|:---:|",
    ]
    for f in folds:
        f_correct = sum(1 for p in f["predictions"] if p["correct"])
        lines.append(
            f"| Fold {f['held_out_person']} | {f['test_clips']} | {f_correct} | {f['test_accuracy'] * 100:.2f}% | {f['test_loss']:.4f} |"
        )

    lines.extend([
        f"",
        f"## Độ chính xác từng Cử chỉ (Xếp từ thấp đến cao)",
        f"",
        f"| Cử chỉ | Đúng / Tổng | Tỉ lệ (%) |",
        f"|---|:---:|:---:|",
    ])
    for l in sorted_labels:
        c = per_label_counts[l]
        lines.append(f"| {l} | {c['correct']} / {c['total']} | {c['correct']/c['total']*100:.1f}% |")

    lines.extend([
        f"",
        f"## Toàn bộ {len(confusion_pairs)} Cặp Nhầm Lẫn (Tổng cộng {sum(confusion_pairs.values())} clips sai)",
        f"",
        f"| STT | Cử chỉ thực tế (True Label) | Dự đoán nhầm sang (Predicted) | Số clips |",
        f"|:---:|---|---|:---:|",
    ])
    for idx, ((true_lbl, pred_lbl), cnt) in enumerate(confusion_pairs.most_common(), 1):
        lines.append(f"| {idx:2d} | {true_lbl} | {pred_lbl} | {cnt} |")
    lines.append(f"| **Tổng** | **{len(confusion_pairs)} hướng nhầm lẫn** | — | **{sum(confusion_pairs.values())}** |")

    lines.extend([
        f"",
        f"## Biểu đồ trực quan",
        f"",
        f"- Đường cong Huấn luyện: `training_curves.png`",
        f"- Ma trận nhầm lẫn: `confusion_matrix.png`",
        f"- Độ chính xác từng cử chỉ: `per_label_accuracy.png`",
    ])

    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_md_path}")


if __name__ == "__main__":
    main()
