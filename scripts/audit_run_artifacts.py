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


def audit_run(run_dir: Path) -> bool:
    print("==================================================")
    print(f"MACHINE AUDIT FOR: {run_dir.name}")
    print("==================================================")

    blockers: list[str] = []

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
        expected_sha, fname = line.split(maxsplit=1)
        recorded_files.add(fname)
        target_f = run_dir / fname
        if not target_f.is_file():
            blockers.append(f"File in SHA256SUMS.txt does not exist on disk: {fname}")
            continue
        actual_sha = get_sha256(target_f)
        if expected_sha != actual_sha:
            blockers.append(f"Checksum mismatch for {fname}: recorded {expected_sha}, actual {actual_sha}")

    # Check for unrecorded files in run directory (except SHA256SUMS.txt itself)
    disk_files = {f.name for f in run_dir.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt"}
    unrecorded = disk_files - recorded_files
    if unrecorded:
        blockers.append(f"Files on disk not listed in SHA256SUMS.txt: {unrecorded}")

    # 3. Audit dataset_files_sha256.csv
    csv_path = run_dir / "dataset_files_sha256.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    if len(csv_rows) != 450:
        blockers.append(f"dataset_files_sha256.csv row count {len(csv_rows)} != 450")

    csv_paths = {r["relative_path"] for r in csv_rows}
    csv_hashes = {r["sha256"] for r in csv_rows}
    if len(csv_paths) != 450:
        blockers.append(f"Unique paths in CSV {len(csv_paths)} != 450")
    if len(csv_hashes) != 450:
        blockers.append(f"Unique hashes in CSV {len(csv_hashes)} != 450")

    # Sample verify files on disk against CSV
    data_dir = Path("dataset/recordings_v1_p123")
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
    if len(extractions) != 450:
        blockers.append(f"Extraction count {len(extractions)} != 450")
    if len(failed_clips) != 0:
        blockers.append(f"Failed clips count {len(failed_clips)} != 0")

    cache_misses = sum(1 for e in extractions if not e.get("from_cache", False))
    cache_hits = sum(1 for e in extractions if e.get("from_cache", False))
    if cache_misses != 450 or cache_hits != 0:
        blockers.append(f"Extraction cache status invalid: misses={cache_misses}, hits={cache_hits} (expected 450 misses, 0 hits)")

    for e in extractions:
        if "P04" in e["video"]:
            blockers.append(f"P04 found in extraction: {e['video']}")
        if e["person"] not in {"P01", "P02", "P03"}:
            blockers.append(f"Invalid person {e['person']} in extraction")

    extraction_video_paths = {Path(e["video"]).resolve() for e in extractions}

    # Audit folds and predictions
    folds = loso["folds"]
    if len(folds) != 3:
        blockers.append(f"Folds count {len(folds)} != 3")

    all_preds = []
    fold_correct = {}
    fold_stats = {}
    for f in folds:
        p_id = f["held_out_person"]
        preds = f["predictions"]
        if len(preds) != 150:
            blockers.append(f"Fold {p_id} predictions {len(preds)} != 150")
        for p in preds:
            if p["person"] != p_id:
                blockers.append(f"Prediction person {p['person']} != held_out {p_id}")
            if "P04" in p["video"]:
                blockers.append(f"P04 found in prediction: {p['video']}")
            all_preds.append(p)
        correct_cnt = sum(1 for p in preds if p["correct"])
        fold_correct[p_id] = correct_cnt
        fold_stats[p_id] = correct_cnt / len(preds)

    if len(all_preds) != 450:
        blockers.append(f"Total predictions {len(all_preds)} != 450")

    pred_video_paths = {Path(p["video"]).resolve() for p in all_preds}
    if pred_video_paths != extraction_video_paths:
        blockers.append("Prediction video paths do not match extraction video paths exactly")

    total_correct = sum(1 for p in all_preds if p["correct"])
    total_wrong = sum(1 for p in all_preds if not p["correct"])
    pooled_acc = total_correct / len(all_preds)
    macro_mean_acc = sum(fold_stats.values()) / 3

    if total_correct != 398 or total_wrong != 52:
        blockers.append(f"Unexpected prediction totals: correct={total_correct}, wrong={total_wrong} (expected 398/52)")

    # Audit all 23 confusion pairs totaling 52
    wrong_preds = [p for p in all_preds if not p["correct"]]
    confusions = Counter((p["label"], p["predicted"]) for p in wrong_preds)
    total_confusions = sum(confusions.values())
    if total_confusions != 52:
        blockers.append(f"Total confusion count {total_confusions} != 52")
    if len(confusions) != 23:
        blockers.append(f"Total distinct confusion pairs {len(confusions)} != 23")

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

    expected_labels = read_expected_labels(Path("dataset/labels.txt"))
    if ckpt["labels"] != expected_labels:
        blockers.append("Checkpoint labels do not match dataset/labels.txt")

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

            # Check that accuracy numbers are accurate in report text
            if "88.44%" not in text and "88.44" not in text:
                blockers.append(f"{r_name} missing 88.44% accuracy statement")

    if blockers:
        print("\n=== MACHINE AUDIT VERDICT: FAILED ===", file=sys.stderr)
        for b in blockers:
            print(f"  [BLOCKER] {b}", file=sys.stderr)
        return False

    print("\n=== MACHINE AUDIT VERDICT: PASSED (ALL CRITERIA VERIFIED) ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine audit on training run artifacts")
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    ok = audit_run(Path(args.run_dir))
    sys.exit(0 if ok else 1)
