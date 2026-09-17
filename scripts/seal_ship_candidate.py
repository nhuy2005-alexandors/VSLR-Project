"""Seal ship candidate run with RUN_MANIFEST.json and SHA256SUMS.txt."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

from prototype_3_gestures.prepare_train import file_sha256

def seal_ship_run(run_dir: Path, loso_dir: Path) -> None:
    print(f"=== SEALING SHIP CANDIDATE: {run_dir.name} ===")

    # 1. Gather hashes
    ckpt_path = run_dir / "gesture_lstm.pt"
    ckpt_sha = file_sha256(ckpt_path).lower()

    dataset_manifest_path = run_dir / "dataset_files_sha256.csv"
    dataset_manifest_sha = file_sha256(dataset_manifest_path).lower()

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    training_sig = metrics["training_signature"]

    labels_path = Path("dataset/labels_v2_24.txt")
    labels_sha = file_sha256(labels_path).lower()

    plan_path = Path("dataset/recording_plan_v3_4x24_8.json")
    plan_sha = file_sha256(plan_path).lower()

    base_commit = (run_dir / "base-commit.txt").read_text(encoding="utf-8").strip()
    branch = (run_dir / "branch.txt").read_text(encoding="utf-8").strip()

    # 2. Build RUN_MANIFEST.json
    run_manifest = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_commit": base_commit,
        "branch": branch,
        "mode": "ship",
        "data_dir": "dataset/recordings_v3_4x24_8",
        "recording_plan": plan_path.as_posix(),
        "labels_sha256": labels_sha,
        "recording_plan_sha256": plan_sha,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "training_signature": training_sig,
        "checkpoint_sha256": ckpt_sha,
        "command_ship": (run_dir / "commands.txt").read_text(encoding="utf-8").strip(),
        "corresponding_loso_run": loso_dir.as_posix(),
        "hyperparameters": {
            "epochs": 13,
            "augment": 120,
            "batch_size": 32,
            "learning_rate": 0.001,
            "seed": 42,
            "num_workers": 4,
            "hidden_size": 96,
            "num_layers": 1,
            "bidirectional": True,
            "sequence_length": 60,
            "feature_dim": 203,
        },
        "pairing_status": "PAIR_OK",
        "evaluation_readiness": "READY_FOR_P05_FINAL_REVIEW",
    }

    run_manifest_path = run_dir / "RUN_MANIFEST.json"
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {run_manifest_path.name}")

    # 3. Generate recursive SHA256SUMS.txt (UTF-8 no BOM)
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            sha = file_sha256(p).lower()
            rel = p.relative_to(run_dir).as_posix()
            lines.append(f"{sha}  {rel}")

    sums_path = run_dir / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {sums_path.name} covering {len(lines)} files (UTF-8 no BOM)")

    print(f"Ship run sealed successfully!")

if __name__ == "__main__":
    run_dir = Path("runs/v3-ship-4signers-24-8clips-13ep-20260916-020145")
    loso_dir = Path("runs/v3-4signers-24-8clips-20260915-215753")
    seal_ship_run(run_dir, loso_dir)
