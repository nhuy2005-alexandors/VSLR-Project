"""Unit tests for generic audit parameterization (4 signers x 24 gestures, 576 clips)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from scripts.audit_run_artifacts import audit_run, get_sha256
from scripts.finalize_run_artifacts import build_run_manifest


@pytest.fixture
def mock_run_dir(tmp_path: Path) -> Path:
    """Create a fully valid mock 4-signer 24-gesture run directory."""
    run_dir = tmp_path / "v2-4signers-24-test"
    run_dir.mkdir(parents=True)

    people = ["P01", "P02", "P03", "P04"]
    labels = [f"Gesture_{i:02d}" for i in range(24)]
    clips_per_label = 6
    expected_clips = len(people) * len(labels) * clips_per_label  # 576

    # 1. Create dataset tree for sample verification
    data_dir = tmp_path / "mock_data"
    for p in people:
        for lbl in labels:
            ldir = data_dir / p / lbl
            ldir.mkdir(parents=True, exist_ok=True)
            for c in range(1, clips_per_label + 1):
                cfile = ldir / f"{c:03d}.mov"
                cfile.write_bytes(f"dummy_video_bytes_{p}_{lbl}_{c}".encode("utf-8"))

    # 2. Recording plan & labels
    plan_path = tmp_path / "recording_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_version": "recordings_v2_4x24",
                "labels_file": "labels.txt",
                "people": people,
                "clips_per_label": clips_per_label,
            }
        ),
        encoding="utf-8",
    )

    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    # 3. dataset_files_sha256.csv
    csv_path = run_dir / "dataset_files_sha256.csv"
    rows = []
    for p in people:
        for lbl in labels:
            for c in range(1, clips_per_label + 1):
                p_obj = data_dir / p / lbl / f"{c:03d}.mov"
                rel_p = f"{p}/{lbl}/{c:03d}.mov"
                rows.append(
                    {
                        "relative_path": rel_p,
                        "person": p,
                        "label": lbl,
                        "filename": f"{c:03d}.mov",
                        "bytes": str(p_obj.stat().st_size),
                        "sha256": get_sha256(p_obj),
                        "width": "1920",
                        "height": "1080",
                        "fps": "60.0",
                        "duration": "5.0",
                    }
                )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 4. Model and signatures
    sig = "mock_sig_4x24_576"
    ckpt_path = run_dir / "gesture_lstm.pt"
    ckpt_obj = {
        "model_state_dict": {},
        "labels": labels,
        "config": {"training_signature": sig, "features_version": 3, "input_dim": 203},
    }
    torch.save(ckpt_obj, ckpt_path)
    ckpt_sha = get_sha256(ckpt_path)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "training_signature": sig,
                "checkpoint_sha256": ckpt_sha,
                "features_version": 3,
                "feature_dim": 203,
                "sequence_length": 60,
                "model": {"hidden_size": 96, "num_layers": 1, "bidirectional": True},
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    # 5. loso_report.json
    extractions = []
    for r in rows:
        extractions.append(
            {
                "video": str((data_dir / r["relative_path"]).resolve()),
                "label": r["label"],
                "person": r["person"],
                "from_cache": True,
            }
        )

    folds = []
    for p in people:
        p_preds = []
        for r in rows:
            if r["person"] == p:
                p_preds.append(
                    {
                        "video": str((data_dir / r["relative_path"]).resolve()),
                        "person": p,
                        "label": r["label"],
                        "predicted": r["label"],
                        "confidence": 0.95,
                        "correct": True,
                    }
                )
        folds.append(
            {
                "held_out_person": p,
                "test_accuracy": 1.0,
                "predictions": p_preds,
                "history": [{"epoch": 40, "val_loss_plain_ce": 0.1, "val_accuracy": 1.0}],
            }
        )

    loso_path = run_dir / "loso_report.json"
    loso_path.write_text(
        json.dumps(
            {
                "training_signature": sig,
                "features_version": 3,
                "feature_dim": 203,
                "sequence_length": 60,
                "labels": labels,
                "extraction": extractions,
                "failed_clips": [],
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )

    # 6. Environmental and log files
    for name in [
        "base-commit.txt",
        "branch.txt",
        "worktree-status.txt",
        "pip-freeze.txt",
        "python-version.txt",
        "gpu-info.txt",
        "commands.txt",
        "test.log",
        "loso.log",
        "ship.log",
        "training_curves.png",
        "confusion_matrix.png",
        "per_label_accuracy.png",
    ]:
        (run_dir / name).write_text("dummy_content\n", encoding="utf-8")

    # Reports
    (run_dir / "REPORT_FOR_AGENT.md").write_text("Accuracy: 100.00%\n", encoding="utf-8")
    (run_dir / "evaluation_report.md").write_text("Pooled Accuracy: 100.00%\n", encoding="utf-8")

    # RUN_MANIFEST.json
    (run_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "external_review_status": "pending",
                "command_loso": "vslr-train --loso",
                "command_ship": "vslr-train",
                "environment": {
                    "python_version": "3.11",
                    "gpu_name": "RTX 4050",
                    "nvidia_driver": "560.94",
                },
                "artifacts": [f"file_{i}" for i in range(20)],
            }
        ),
        encoding="utf-8",
    )

    # SHA256SUMS.txt
    all_files = sorted(f for f in run_dir.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt")
    sums_lines = [f"{get_sha256(f)}  {f.name}" for f in all_files]
    (run_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    return run_dir


def test_generic_audit_passes_on_576_clips_4_folds(mock_run_dir: Path):
    tmp_path = mock_run_dir.parent
    data_dir = tmp_path / "mock_data"
    plan_path = tmp_path / "recording_plan.json"
    labels_path = tmp_path / "labels.txt"

    ok = audit_run(
        run_dir=mock_run_dir,
        data_dir=data_dir,
        recording_plan_path=plan_path,
        labels_file_path=labels_path,
        expected_clips=576,
    )
    assert ok is True


def test_generic_audit_refuses_hen_gap_lai(mock_run_dir: Path):
    tmp_path = mock_run_dir.parent
    data_dir = tmp_path / "mock_data"
    plan_path = tmp_path / "recording_plan.json"
    labels_path = tmp_path / "labels.txt"

    # Inject 'Hẹn gặp lại' into loso_report.json
    loso_path = mock_run_dir / "loso_report.json"
    loso = json.loads(loso_path.read_text(encoding="utf-8"))
    loso["extraction"][0]["label"] = "Hẹn gặp lại"
    loso_path.write_text(json.dumps(loso), encoding="utf-8")

    # Update SHA256SUMS
    all_files = sorted(f for f in mock_run_dir.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt")
    sums_lines = [f"{get_sha256(f)}  {f.name}" for f in all_files]
    (mock_run_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    ok = audit_run(
        run_dir=mock_run_dir,
        data_dir=data_dir,
        recording_plan_path=plan_path,
        labels_file_path=labels_path,
        expected_clips=576,
    )
    assert ok is False


def test_generic_audit_refuses_wrong_fold_count(mock_run_dir: Path):
    tmp_path = mock_run_dir.parent
    data_dir = tmp_path / "mock_data"
    plan_path = tmp_path / "recording_plan.json"
    labels_path = tmp_path / "labels.txt"

    # Remove one fold
    loso_path = mock_run_dir / "loso_report.json"
    loso = json.loads(loso_path.read_text(encoding="utf-8"))
    loso["folds"] = loso["folds"][:3]
    loso_path.write_text(json.dumps(loso), encoding="utf-8")

    all_files = sorted(f for f in mock_run_dir.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt")
    sums_lines = [f"{get_sha256(f)}  {f.name}" for f in all_files]
    (mock_run_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    ok = audit_run(
        run_dir=mock_run_dir,
        data_dir=data_dir,
        recording_plan_path=plan_path,
        labels_file_path=labels_path,
        expected_clips=576,
    )
    assert ok is False


def test_build_run_manifest_parameterized(tmp_path: Path):
    run_dir = tmp_path / "mock_run"
    run_dir.mkdir()
    manifest = build_run_manifest(
        run_dir=run_dir,
        base_commit="abc1234",
        branch_name="test-branch",
        worktree_status="",
        python_version="3.11.9",
        gpu_info_text="Driver Version: 560.94 CUDA Version: 12.6 | 0 NVIDIA GeForce RTX 4050",
        metrics_data={"runtime": {}, "model": {}},
        sig_loso="sig123",
        checkpoint_sha="ckpt123",
        labels_file=tmp_path / "custom_labels.txt",
        plan_file=tmp_path / "custom_plan.json",
        manifest_csv=tmp_path / "custom_manifest.csv",
        fold_stats={},
        all_preds=[],
        artifacts_info=[],
        data_dir="dataset/custom_data",
        cache_dir="dataset/processed/custom_cache",
    )
    assert manifest["data_dir"] == "dataset/custom_data"
    assert "dataset/custom_data" in manifest["command_loso"]
    assert "dataset/custom_data" in manifest["command_ship"]
    assert "custom_plan.json" in manifest["command_loso"]
    assert "custom_cache" in manifest["command_loso"]
    assert "recordings_v1_p123" not in manifest["command_loso"]
    assert "recordings_v1_p123" not in manifest["command_ship"]
