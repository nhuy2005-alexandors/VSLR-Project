"""Independent verifier for V3 Ship Candidate artifacts and integrity gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from prototype_3_gestures.prepare_train import (
    FEATURE_CONTRACT,
    FEATURE_DIM,
    FEATURES_VERSION,
    MODEL_ARCHITECTURE_VERSION,
    POOLING_FWD_LAST_BWD_FIRST,
    REJECTION_CONTRACT,
    SEGMENTATION_CONTRACT,
    SEQUENCE_LENGTH,
    TRAINING_RECIPE_VERSION,
    artifact_pair_matches,
    discover_clips,
    data_fingerprint,
    file_sha256,
)
from prototype_3_gestures.vsl3.model import GestureLSTM, load_checkpoint


def verify_ship_candidate(run_dir: Path, selected_loso_dir: Path) -> dict:
    print(f"=== VERIFYING SHIP CANDIDATE: {run_dir.name} ===")
    errors = []

    # 1. Required files
    required_files = [
        "base-commit.txt",
        "branch.txt",
        "worktree-status.txt",
        "commands.txt",
        "python-version.txt",
        "pip-freeze.txt",
        "gpu-info.txt",
        "source-hashes.txt",
        "script-hashes.txt",
        "dataset_files_sha256.csv",
        "dataset-fingerprint.txt",
        "selected-loso-run.txt",
        "old-model-hash.txt",
        "gesture_lstm.pt",
        "metrics.json",
        "labels.json",
        "ship_training.log",
    ]
    for rf in required_files:
        p = run_dir / rf
        if not p.is_file():
            errors.append(f"Missing required file: {rf}")
        elif p.stat().st_size == 0:
            errors.append(f"Required file is empty (0 bytes): {rf}")

    if errors:
        raise ValueError("Missing/empty files:\n" + "\n".join(f"  - {e}" for e in errors))

    # 2. Check invariant: models/gesture_lstm.pt hash NOT altered
    old_hash_recorded = (run_dir / "old-model-hash.txt").read_text(encoding="utf-8").strip().lower()
    current_prod_model = Path("models/gesture_lstm.pt")
    current_prod_hash = file_sha256(current_prod_model).lower()
    if current_prod_hash != old_hash_recorded:
        errors.append(f"CRITICAL INVARIANT VIOLATION: models/gesture_lstm.pt was modified! Recorded: {old_hash_recorded}, Current: {current_prod_hash}")
    else:
        print(f"Prod model invariant: PASS (models/gesture_lstm.pt unmodified: {current_prod_hash})")

    # 3. Load checkpoint
    ckpt_path = run_dir / "gesture_lstm.pt"
    ckpt_sha256 = file_sha256(ckpt_path).lower()
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    config = checkpoint.get("config", {})
    labels = checkpoint.get("labels", [])

    # Labels check
    manifest_labels = [line.strip() for line in Path("dataset/labels_v2_24.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if labels != manifest_labels:
        errors.append(f"Checkpoint labels mismatch: got {len(labels)} labels, expected 24 labels matching dataset/labels_v2_24.txt")
    if "Hẹn gặp lại" in labels:
        errors.append("Forbidden label 'Hẹn gặp lại' in checkpoint labels")

    # Input dim and sequence length
    if config.get("input_dim") != FEATURE_DIM:
        errors.append(f"Config input_dim={config.get('input_dim')} != {FEATURE_DIM}")
    if config.get("sequence_length") != SEQUENCE_LENGTH:
        errors.append(f"Config sequence_length={config.get('sequence_length')} != {SEQUENCE_LENGTH}")
    if config.get("features_version") != FEATURES_VERSION:
        errors.append(f"Config features_version={config.get('features_version')} != {FEATURES_VERSION}")
    if config.get("feature_contract") != FEATURE_CONTRACT:
        errors.append(f"Config feature_contract={config.get('feature_contract')} != {FEATURE_CONTRACT}")
    if config.get("pooling") != POOLING_FWD_LAST_BWD_FIRST:
        errors.append(f"Config pooling={config.get('pooling')} != {POOLING_FWD_LAST_BWD_FIRST}")
    if config.get("training_recipe_version") != TRAINING_RECIPE_VERSION:
        errors.append(f"Config training_recipe_version={config.get('training_recipe_version')} != {TRAINING_RECIPE_VERSION}")
    if config.get("model_architecture_version") != MODEL_ARCHITECTURE_VERSION:
        errors.append(f"Config model_architecture_version={config.get('model_architecture_version')} != {MODEL_ARCHITECTURE_VERSION}")

    # 4. Check metrics.json
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if metrics.get("mode") != "ship":
        errors.append(f"Metrics mode={metrics.get('mode')} != 'ship'")
    if metrics.get("checkpoint_sha256").lower() != ckpt_sha256:
        errors.append(f"Metrics checkpoint_sha256 mismatch: metrics {metrics.get('checkpoint_sha256')} vs actual {ckpt_sha256}")

    ckpt_sig = config.get("training_signature")
    metrics_sig = metrics.get("training_signature")
    if not ckpt_sig:
        errors.append("Missing training_signature in checkpoint config")
    if ckpt_sig != metrics_sig:
        errors.append(f"Signature mismatch: checkpoint {ckpt_sig} vs metrics {metrics_sig}")

    # 5. Check dataset manifest & zero leakage of P05
    manifest_csv = run_dir / "dataset_files_sha256.csv"
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    if len(manifest_rows) != 768:
        errors.append(f"Expected 768 clips in training manifest, got {len(manifest_rows)}")

    train_hashes = set()
    for row in manifest_rows:
        p = row["relative_path"]
        person = row["person"]
        sha = row["sha256"].lower()
        if "P05" in p or person == "P05":
            errors.append(f"P05 found in training manifest: {row}")
        if sha in train_hashes:
            errors.append(f"Duplicate sha256 in training manifest: {sha} ({p})")
        train_hashes.add(sha)

    # Cross check with P05 external test files
    p05_dir = Path("dataset/external_eval_v3/P05")
    if p05_dir.is_dir():
        for p05_file in p05_dir.rglob("*"):
            if p05_file.is_file() and p05_file.suffix.lower() in (".mov", ".mp4"):
                p05_sha = file_sha256(p05_file).lower()
                if p05_sha in train_hashes:
                    errors.append(f"DATA LEAKAGE: P05 clip {p05_file} SHA {p05_sha} found in training set!")

    # 6. Three-way pairing with selected LOSO run
    loso_report_path = selected_loso_dir / "loso_report.json"
    if loso_report_path.is_file():
        with open(loso_report_path, "r", encoding="utf-8") as f:
            loso_report = json.load(f)
        loso_sig = loso_report.get("training_signature")
        if loso_sig != metrics_sig:
            errors.append(f"Pairing signature mismatch: LOSO report {loso_sig} vs Ship metrics {metrics_sig}")
        else:
            pair_ok = artifact_pair_matches(loso_sig, metrics, ckpt_path)
            if not pair_ok:
                errors.append(f"artifact_pair_matches() returned False between {loso_report_path} and {ckpt_path}")
            else:
                print(f"Three-way artifact pairing: PASS (signature: {metrics_sig})")

    # 7. Model loading using official load_checkpoint and synthetic inference smoke (WITHOUT touching P05!)
    model, loaded_labels, loaded_config = load_checkpoint(ckpt_path, device="cpu")
    model.eval()

    # Synthetic input smoke test
    dummy_input = torch.randn(2, SEQUENCE_LENGTH, FEATURE_DIM)
    with torch.no_grad():
        out = model(dummy_input)
        probs = torch.softmax(out, dim=-1)

    if probs.shape != (2, len(labels)):
        errors.append(f"Synthetic inference shape mismatch: expected (2, {len(labels)}), got {probs.shape}")
    if not torch.all(torch.isfinite(probs)):
        errors.append("Synthetic inference produced non-finite probabilities")
    row_sums = probs.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
        errors.append("Synthetic probabilities do not sum to 1.0")

    print(f"Model architecture and synthetic inference smoke: PASS (shape {probs.shape}, finite, sums to 1.0)")

    if errors:
        print("FAIL: Verification errors encountered:\n" + "\n".join(f"  - {e}" for e in errors), file=sys.stderr)
        sys.exit(1)

    print(f"=== ALL SHIP CANDIDATE GATES VERIFIED SUCCESSFULLY ===")
    print(f"Candidate Checkpoint SHA-256: {ckpt_sha256}")
    print(f"Training Signature: {metrics_sig}")
    return {
        "checkpoint_sha256": ckpt_sha256,
        "training_signature": metrics_sig,
        "labels_count": len(labels),
        "clips_count": len(manifest_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ship candidate artifacts and integrity")
    parser.add_argument("run_dir", help="Path to ship run directory")
    parser.add_argument("--loso-run", default="runs/v3-4signers-24-8clips-20260915-215753", help="Path to corresponding LOSO run")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    loso_dir = Path(args.loso_run)

    verify_ship_candidate(run_dir, loso_dir)


if __name__ == "__main__":
    main()
