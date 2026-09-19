"""Generate top-3 LOSO analysis directory and report from loso_report.json."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


from prototype_3_gestures.top3 import validate_top3_row


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    loso_path = run_dir / "loso_report.json"
    if not loso_path.exists():
        print(f"Error: {loso_path} does not exist", file=sys.stderr)
        sys.exit(1)

    with open(loso_path, encoding="utf-8") as f:
        report = json.load(f)

    # Durations from extraction stats
    durations: dict[str, float] = {}
    for ext in report.get("extraction", []):
        vid = ext.get("video", "")
        fps = ext.get("fps", 0.0)
        frames = ext.get("sampled_frames", 0)
        durations[vid] = (frames / fps) if fps > 0 else 0.0

    out_dir = run_dir / "top3_loso_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = [p for fold in report.get("folds", []) for p in fold.get("predictions", [])]
    if not predictions:
        raise ValueError(f"No predictions found in {loso_path}")

    top3_preds = []
    for p in predictions:
        validate_top3_row(p)
        vid = p["video"]
        top3_preds.append({
            "held_out_person": p["person"],
            "video": vid,
            "true_label": p["label"],
            "top1_label": p["predicted"],
            "top1_confidence": p["confidence"],
            "top2_label": p["top2_label"],
            "top2_confidence": p["top2_confidence"],
            "top3_label": p["top3_label"],
            "top3_confidence": p["top3_confidence"],
            "top1_top2_margin": p["top1_top2_margin"],
            "correct": p["correct"],
            "duration_seconds": durations.get(vid, 0.0),
        })

    # Save top3_predictions.json
    (out_dir / "top3_predictions.json").write_text(
        json.dumps(top3_preds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save top3_predictions.csv
    pred_fields = [
        "held_out_person", "video", "true_label", "top1_label", "top1_confidence",
        "top2_label", "top2_confidence", "top3_label", "top3_confidence",
        "top1_top2_margin", "correct", "duration_seconds"
    ]
    with open(out_dir / "top3_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(top3_preds)

    # Per gesture aggregation
    labels = report.get("labels", sorted({p["true_label"] for p in top3_preds}))
    by_gesture = []
    for lbl in labels:
        lbl_preds = [p for p in top3_preds if p["true_label"] == lbl]
        if not lbl_preds:
            continue
        clips = len(lbl_preds)
        correct = sum(1 for p in lbl_preds if p["correct"])
        acc = correct / clips
        mean_top1 = sum(p["top1_confidence"] for p in lbl_preds) / clips
        mean_margin = sum(p["top1_top2_margin"] for p in lbl_preds) / clips
        low_margin = sum(1 for p in lbl_preds if p["top1_top2_margin"] < 0.15)
        top2_counter = Counter(p["top2_label"] for p in lbl_preds)
        wrong_counter = Counter(p["top1_label"] for p in lbl_preds if not p["correct"])

        by_gesture.append({
            "label": lbl,
            "clips": clips,
            "correct": correct,
            "accuracy": acc,
            "mean_top1_confidence": mean_top1,
            "mean_top1_top2_margin": mean_margin,
            "low_margin_below_0_15": low_margin,
            "most_common_top2": top2_counter.most_common(3),
            "most_common_wrong_top1": wrong_counter.most_common(3),
        })

    # Sort by accuracy ascending, then label
    by_gesture.sort(key=lambda x: (x["accuracy"], x["label"]))

    # Save top3_by_gesture.json
    (out_dir / "top3_by_gesture.json").write_text(
        json.dumps(by_gesture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Save top3_by_gesture.csv
    gesture_fields = [
        "label", "clips", "correct", "accuracy", "mean_top1_confidence",
        "mean_top1_top2_margin", "low_margin_below_0_15", "most_common_top2", "most_common_wrong_top1"
    ]
    with open(out_dir / "top3_by_gesture.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gesture_fields)
        writer.writeheader()
        for g in by_gesture:
            row = dict(g)
            row["most_common_top2"] = json.dumps(g["most_common_top2"], ensure_ascii=False)
            row["most_common_wrong_top1"] = json.dumps(g["most_common_wrong_top1"], ensure_ascii=False)
            writer.writerow(row)

    # Generate REPORT.md
    md_lines = [
        "# LOSO Top-3 analysis",
        "",
        "Each probability comes from the held-out signer's fold model, never the all-signer ship model.",
        "",
        "| Gesture | Correct | Accuracy | Mean top-1 | Mean margin | Margin < 0.15 | Common top-2 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for g in by_gesture:
        common_t2_str = ", ".join(f"{lbl} ({cnt})" for lbl, cnt in g["most_common_top2"])
        md_lines.append(
            f"| {g['label']} | {g['correct']}/{g['clips']} | {g['accuracy'] * 100:.1f}% | "
            f"{g['mean_top1_confidence'] * 100:.1f}% | {g['mean_top1_top2_margin'] * 100:.1f}% | "
            f"{g['low_margin_below_0_15']} | {common_t2_str} |"
        )
    md_lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Top-3 analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
