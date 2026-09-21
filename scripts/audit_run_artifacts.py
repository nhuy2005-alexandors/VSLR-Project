"""Machine audit script for run artifacts verification and provenance auditing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

from prototype_3_gestures.vsl3.labels import read_expected_labels


FORBIDDEN_TERMS = [
    "0 blockers",
    "100% reproducible",
    "independent reviewer pass",
    "khách quan 100%",
    "gian lận",
    "mô hình chính thức",
    "đã tổng quát hóa",
    "đạt yêu cầu thực tế",
]

UNVERIFIED_CLAIMS = [
    "0.5 giây cuối là tĩnh",
    "0.5s tĩnh",
    "phần bị cắt là tĩnh",
    "google drive gốc có đúng 25 thư mục",
    "google drive gốc của nhóm thực tế chứa 25 thư mục",
]


def check_report_text_claims(text: str) -> None:
    """Verify that report text contains no forbidden phrases or unverified claims."""
    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in text_lower:
            raise ValueError(f"Forbidden term found in report: '{term}'")

    for claim in UNVERIFIED_CLAIMS:
        # Check if claim appears without disclaimer
        if claim in text_lower:
            # If it says "không chứng minh phần bị cắt là tĩnh", that is allowed
            if "không chứng minh" in text_lower and "phần bị cắt là tĩnh" in claim:
                continue
            raise ValueError(f"Unverified claim found in report: '{claim}'")


def get_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_run(
    run_dir: Path,
    data_dir: Path = Path("dataset/recordings_v2_4x24"),
    recording_plan_path: Path = Path("dataset/recording_plan_v2_4x24.json"),
    labels_file_path: Path = Path("dataset/labels_v2_24.txt"),
    cache_dir: Path | None = None,
    expected_clips: int | None = None,
) -> bool:
    print("==================================================")
    print(f"MACHINE AUDIT FOR: {run_dir.name}")
    print("==================================================")

    blockers: list[str] = []

    # 0. Load contracts
    plan_data = json.loads(recording_plan_path.read_text(encoding="utf-8"))
    allowed_people = set(plan_data["people"])
    expected_labels = read_expected_labels(labels_file_path)
    if expected_clips is None:
        expected_clips = len(allowed_people) * len(expected_labels) * plan_data.get("clips_per_label", 6)
    expected_folds = len(allowed_people)
    expected_preds_per_fold = expected_clips // expected_folds

    print(f"Audit configuration: {len(expected_labels)} labels, {expected_folds} people ({sorted(allowed_people)}), expected {expected_clips} clips ({expected_preds_per_fold} clips/fold)")

    # 1. Required artifact set
    required_files = [
        "base-commit.txt", "branch.txt", "worktree-status.txt", "pip-freeze.txt",
        "python-version.txt", "gpu-info.txt", "commands.txt", "test.log",
        "dataset_files_sha256.csv", "loso.log", "ship.log", "loso_report.json",
        "metrics.json", "gesture_lstm.pt", "labels.json", "training_curves.png",
        "confusion_matrix.png", "per_label_accuracy.png", "evaluation_report.md",
        "REPORT_FOR_AGENT.md", "RUN_MANIFEST.json", "SHA256SUMS.txt"
    ]
    for rf in required_files:
        if not (run_dir / rf).is_file():
            blockers.append(f"Missing required artifact: {rf}")

    if blockers:
        print("FAIL: Missing files:\n" + "\n".join(blockers), file=sys.stderr)
        return False

    # 2. Audit SHA256SUMS.txt
    sums_lines = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    recorded_files = set()
    for line in sums_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        expected_sha, fname = parts
        recorded_files.add(fname)
        target_f = run_dir / fname
        if not target_f.is_file():
            blockers.append(f"File in SHA256SUMS.txt does not exist on disk: {fname}")
            continue
        actual_sha = get_sha256(target_f)
        if expected_sha != actual_sha:
            blockers.append(f"Checksum mismatch for {fname}: recorded {expected_sha}, actual {actual_sha}")

    disk_files = {f.name for f in run_dir.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt"}
    unrecorded = disk_files - recorded_files
    if unrecorded:
        blockers.append(f"Files on disk not listed in SHA256SUMS.txt: {unrecorded}")

    # 3. Audit dataset_files_sha256.csv
    csv_path = run_dir / "dataset_files_sha256.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    if len(csv_rows) != expected_clips:
        blockers.append(f"dataset_files_sha256.csv row count {len(csv_rows)} != {expected_clips}")

    csv_paths = {r["relative_path"] for r in csv_rows}
    csv_hashes = {r["sha256"] for r in csv_rows}
    if len(csv_paths) != expected_clips:
        blockers.append(f"Unique paths in CSV {len(csv_paths)} != {expected_clips}")
    if len(csv_hashes) != expected_clips:
        blockers.append(f"Unique hashes in CSV {len(csv_hashes)} != {expected_clips}")

    # Verify no Hẹn gặp lại in CSV
    for r in csv_rows:
        if "Hẹn gặp lại" in r["relative_path"] or "Hẹn gặp lại" in r["label"]:
            blockers.append(f"Forbidden label 'Hẹn gặp lại' found in dataset_files_sha256.csv: {r['relative_path']}")

    # Sample verify files on disk against CSV
    for r in csv_rows[:10] + csv_rows[-10:]:
        p = data_dir / r["relative_path"]
        if not p.is_file():
            blockers.append(f"Video file missing: {p}")
        else:
            if p.stat().st_size != int(r["bytes"]):
                blockers.append(f"Size mismatch for {p}: disk {p.stat().st_size} vs CSV {r['bytes']}")
            if get_sha256(p) != r["sha256"]:
                blockers.append(f"Hash mismatch for {p} against CSV")

    # 4. Audit loso_report.json
    loso = json.loads((run_dir / "loso_report.json").read_text(encoding="utf-8"))
    extractions = loso["extraction"]
    failed_clips = loso["failed_clips"]
    if len(extractions) != expected_clips:
        blockers.append(f"Extraction count {len(extractions)} != {expected_clips}")
    if len(failed_clips) != 0:
        blockers.append(f"Failed clips count {len(failed_clips)} != 0")

    cache_misses = sum(1 for e in extractions if not e.get("from_cache", False))
    cache_hits = sum(1 for e in extractions if e.get("from_cache", False))
    if cache_misses + cache_hits != expected_clips:
        blockers.append(f"Extraction cache status invalid: misses={cache_misses} + hits={cache_hits} != {expected_clips}")

    for e in extractions:
        if "Hẹn gặp lại" in e["video"] or "Hẹn gặp lại" in e["label"]:
            blockers.append(f"Forbidden label 'Hẹn gặp lại' found in extraction: {e['video']}")
        if e["person"] not in allowed_people:
            blockers.append(f"Invalid person {e['person']} not in {allowed_people} in extraction")

    extraction_video_paths = {Path(e["video"]).resolve() for e in extractions}

    # Audit folds and predictions
    folds = loso["folds"]
    if len(folds) != expected_folds:
        blockers.append(f"Folds count {len(folds)} != {expected_folds}")

    all_preds = []
    fold_correct = {}
    fold_stats = {}
    for f in folds:
        p_id = f["held_out_person"]
        if p_id not in allowed_people:
            blockers.append(f"Held out person {p_id} not in allowed people {allowed_people}")
        preds = f["predictions"]
        if len(preds) != expected_preds_per_fold:
            blockers.append(f"Fold {p_id} predictions {len(preds)} != {expected_preds_per_fold}")
        for p in preds:
            if p["person"] != p_id:
                blockers.append(f"Prediction person {p['person']} != held_out {p_id}")
            if "Hẹn gặp lại" in p["video"] or "Hẹn gặp lại" in p["label"] or "Hẹn gặp lại" in p.get("predicted", ""):
                blockers.append(f"Forbidden label 'Hẹn gặp lại' found in prediction: {p}")
            all_preds.append(p)
        correct_cnt = sum(1 for p in preds if p["correct"])
        fold_correct[p_id] = correct_cnt
        fold_stats[p_id] = correct_cnt / len(preds)

    if len(all_preds) != expected_clips:
        blockers.append(f"Total predictions {len(all_preds)} != {expected_clips}")

    pred_video_paths = {Path(p["video"]).resolve() for p in all_preds}
    if pred_video_paths != extraction_video_paths:
        blockers.append("Prediction video paths do not match extraction video paths exactly")

    total_correct = sum(1 for p in all_preds if p["correct"])
    total_wrong = sum(1 for p in all_preds if not p["correct"])
    pooled_acc = total_correct / len(all_preds)
    macro_mean_acc = sum(fold_stats.values()) / len(fold_stats)

    wrong_preds = [p for p in all_preds if not p["correct"]]
    confusions = Counter((p["label"], p["predicted"]) for p in wrong_preds)
    total_confusions = sum(confusions.values())
    if total_confusions != total_wrong:
        blockers.append(f"Total confusion count {total_confusions} != total wrong {total_wrong}")

    # 5. Three-way binding and Checkpoint SHA-256
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ckpt = torch.load(run_dir / "gesture_lstm.pt", map_location="cpu")

    sig_loso = loso["training_signature"]
    sig_metrics = metrics["training_signature"]
    sig_ckpt = ckpt.get("config", {}).get("training_signature")
    if not (sig_loso == sig_metrics == sig_ckpt):
        blockers.append(f"Signature mismatch: loso={sig_loso}, metrics={sig_metrics}, ckpt={sig_ckpt}")

    actual_ckpt_sha = get_sha256(run_dir / "gesture_lstm.pt")
    if metrics["checkpoint_sha256"] != actual_ckpt_sha:
        blockers.append(f"Checkpoint SHA mismatch in metrics: {metrics['checkpoint_sha256']} != {actual_ckpt_sha}")

    if ckpt["labels"] != expected_labels:
        blockers.append(f"Checkpoint labels do not match {labels_file_path}")

    # 6. Audit RUN_MANIFEST.json
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if "external_review_status" not in manifest or manifest["external_review_status"] != "pending":
        blockers.append(f"Manifest external_review_status must be 'pending', got {manifest.get('external_review_status')}")

    if "<RunDir>" in manifest.get("command_loso", "") or "<RunDir>" in manifest.get("command_ship", ""):
        blockers.append("Manifest commands contain placeholder <RunDir>")

    env = manifest.get("environment", {})
    if not env.get("python_version") or not env.get("gpu_name") or not env.get("nvidia_driver"):
        blockers.append("Manifest environment missing python_version, gpu_name, or nvidia_driver")

    if not manifest.get("artifacts") or len(manifest["artifacts"]) < 15:
        blockers.append("Manifest artifacts list missing or incomplete")

    # 7. Audit Report Texts
    for r_name in ["REPORT_FOR_AGENT.md", "evaluation_report.md"]:
        r_path = run_dir / r_name
        if r_path.is_file():
            text = r_path.read_text(encoding="utf-8")
            try:
                check_report_text_claims(text)
            except ValueError as e:
                blockers.append(f"{r_name}: {e}")

            pct_str = f"{pooled_acc:.2%}"
            if pct_str not in text and f"{pooled_acc:.1%}" not in text:
                blockers.append(f"{r_name} missing {pct_str} accuracy statement")

    if blockers:
        print("\n=== MACHINE AUDIT VERDICT: FAILED ===", file=sys.stderr)
        for b in blockers:
            print(f"  [BLOCKER] {b}", file=sys.stderr)
        return False

    print("\n=== MACHINE AUDIT VERDICT: PASSED (ALL CRITERIA VERIFIED) ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine audit on training run artifacts")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--data-dir", type=str, default="dataset/recordings_v2_4x24", help="Path to dataset root")
    parser.add_argument("--recording-plan", type=str, default="dataset/recording_plan_v2_4x24.json", help="Recording plan JSON")
    parser.add_argument("--labels-file", type=str, default="dataset/labels_v2_24.txt", help="Labels text file")
    parser.add_argument("--cache-dir", type=str, default="dataset/processed/landmark_cache_v2_4x24", help="Landmark cache dir")
    parser.add_argument("--expected-clips", type=int, default=None, help="Expected clips count")
    args = parser.parse_args()
    ok = audit_run(
        run_dir=Path(args.run_dir),
        data_dir=Path(args.data_dir),
        recording_plan_path=Path(args.recording_plan),
        labels_file_path=Path(args.labels_file),
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        expected_clips=args.expected_clips,
    )
    sys.exit(0 if ok else 1)
