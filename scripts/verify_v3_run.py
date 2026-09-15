"""Independent verifier for V3 LOSO run artifacts and integrity gates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from prototype_3_gestures.top3 import validate_top3_row

REQUIRED_RUN_FILES = [
    "base-commit.txt",
    "branch.txt",
    "worktree-status.txt",
    "pip-freeze.txt",
    "python-version.txt",
    "gpu-info.txt",
    "commands.txt",
    "dataset_files_sha256.csv",
    "loso.log",
    "loso_report.json",
    "evaluation_report.md",
    "confusion_matrix.png",
    "per_label_accuracy.png",
    "training_curves.png",
    "SHA256SUMS.txt",
    "top3_loso_analysis/REPORT.md",
    "top3_loso_analysis/top3_predictions.csv",
    "top3_loso_analysis/top3_predictions.json",
    "top3_loso_analysis/top3_by_gesture.csv",
    "top3_loso_analysis/top3_by_gesture.json",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def generate_sha256sums(run_dir: Path) -> Path:
    out_file = run_dir / "SHA256SUMS.txt"
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rel = p.relative_to(run_dir).as_posix()
            sha = sha256_file(p)
            lines.append(f"{sha}  {rel}")
    # Write UTF-8 without BOM
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Verify or generate V3 run checksums and integrity gates")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--generate-sums", action="store_true", help="Generate SHA256SUMS.txt before verification")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.generate_sums:
        generate_sha256sums(run_dir)
        print(f"Generated recursive SHA256SUMS.txt (UTF-8 no BOM) for {run_dir.name}")

    print(f"=== INDEPENDENT VERIFIER: {run_dir.name} ===")
    errors = []

    # 1. Check required artifacts exist and non-empty
    for rel_path in REQUIRED_RUN_FILES:
        target = run_dir / rel_path
        if not target.is_file():
            errors.append(f"Missing required artifact: {rel_path}")
        elif target.stat().st_size == 0:
            errors.append(f"Required artifact is empty (0 bytes): {rel_path}")

    if errors:
        print("FAIL: Missing or empty artifacts:\n" + "\n".join(f"  - {e}" for e in errors), file=sys.stderr)
        sys.exit(1)
    print(f"Artifact existence check: PASS ({len(REQUIRED_RUN_FILES)} required files verified)")

    # 2. Check SHA256SUMS.txt
    sums_file = run_dir / "SHA256SUMS.txt"
    sums_content = sums_file.read_bytes()
    if sums_content.startswith(b"\xef\xbb\xbf"):
        errors.append("SHA256SUMS.txt contains UTF-8 BOM; contract requires UTF-8 without BOM")

    sums_lines = sums_file.read_text(encoding="utf-8").splitlines()
    recorded = {}
    for idx, line in enumerate(sums_lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Invalid line format in SHA256SUMS.txt line {idx}: {line!r}")
            continue
        expected_sha, rel_name = parts
        rel_name = rel_name.strip()
        recorded[rel_name] = expected_sha.lower()

    for rel_name, expected_sha in recorded.items():
        actual_path = run_dir / rel_name
        if not actual_path.is_file():
            errors.append(f"File listed in SHA256SUMS.txt does not exist on disk: {rel_name}")
        else:
            actual_sha = sha256_file(actual_path)
            if actual_sha != expected_sha:
                errors.append(f"SHA-256 mismatch for {rel_name}: disk {actual_sha} vs recorded {expected_sha}")

    # Check all files on disk are in recorded
    for disk_file in run_dir.rglob("*"):
        if disk_file.is_file() and disk_file.name != "SHA256SUMS.txt":
            rel = disk_file.relative_to(run_dir).as_posix()
            if rel not in recorded:
                errors.append(f"Unrecorded file on disk not present in SHA256SUMS.txt: {rel}")

    if errors:
        print("FAIL: Checksum verification errors:\n" + "\n".join(f"  - {e}" for e in errors), file=sys.stderr)
        sys.exit(1)
    print(f"Recursive checksum verification: PASS ({len(recorded)} files checked, 0 mismatches)")

    # 3. Check loso_report.json data integrity
    with open(run_dir / "loso_report.json", encoding="utf-8") as f:
        report = json.load(f)

    folds = report.get("folds", [])
    if len(folds) != 4:
        errors.append(f"Expected exactly 4 folds, got {len(folds)}")

    expected_people = ["P01", "P02", "P03", "P04"]
    fold_people = [f["held_out_person"] for f in folds]
    if sorted(fold_people) != expected_people:
        errors.append(f"Unexpected fold signers {fold_people}, expected {expected_people}")

    total_preds = 0
    all_preds = []
    for f in folds:
        if f.get("train_clips") != 576:
            errors.append(f"Fold {f.get('held_out_person')}: train_clips={f.get('train_clips')} != 576")
        if f.get("test_clips") != 192:
            errors.append(f"Fold {f.get('held_out_person')}: test_clips={f.get('test_clips')} != 192")
        preds = f.get("predictions", [])
        total_preds += len(preds)
        all_preds.extend(preds)

    if total_preds != 768:
        errors.append(f"Expected 768 total predictions, got {total_preds}")

    # Zero leakage check
    for p in all_preds:
        if "P05" in p.get("video", "") or p.get("person") == "P05":
            errors.append(f"DATA LEAKAGE: P05 found in predictions: {p}")
        try:
            validate_top3_row(p)
        except ValueError as exc:
            errors.append(f"Top-3 fail-closed validation failed for {p.get('video')}: {exc}")

    for ext in report.get("extraction", []):
        if "P05" in ext.get("video", "") or ext.get("person") == "P05":
            errors.append(f"DATA LEAKAGE: P05 found in extraction stats: {ext}")

    # Accuracy math check
    fold_accs = [f["test_accuracy"] for f in folds]
    expected_macro = sum(fold_accs) / len(fold_accs)
    if abs(report.get("macro_mean_accuracy", 0.0) - expected_macro) > 1e-6:
        errors.append(f"macro_mean_accuracy mismatch: {report.get('macro_mean_accuracy')} vs expected {expected_macro}")

    total_correct = sum(1 for p in all_preds if p.get("correct"))
    expected_pooled = total_correct / total_preds if total_preds > 0 else 0.0
    if abs(report.get("pooled_accuracy", 0.0) - expected_pooled) > 1e-6:
        errors.append(f"pooled_accuracy mismatch: {report.get('pooled_accuracy')} vs expected {expected_pooled}")

    if errors:
        print("FAIL: Data integrity gate errors:\n" + "\n".join(f"  - {e}" for e in errors), file=sys.stderr)
        sys.exit(1)

    print("Data integrity & Zero-leakage check: PASS (768 predictions verified, 0 P05 leakage)")
    print("=== VERIFIER PASS: All integrity gates verified! ===")


if __name__ == "__main__":
    main()
