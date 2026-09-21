"""Tests for run artifact validation, machine audit, and provenance compliance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_audit_module_exists():
    """Script must be named audit_run_artifacts.py, NOT independent_reviewer_audit.py."""
    audit_script = Path("scripts/audit_run_artifacts.py")
    assert audit_script.is_file(), "scripts/audit_run_artifacts.py must exist"


def test_no_independent_reviewer_script_naming():
    """Old misleading name must be deleted or renamed."""
    old_script = Path("scripts/independent_reviewer_audit.py")
    assert not old_script.exists(), "scripts/independent_reviewer_audit.py must be removed/renamed"


def test_manifest_schema_and_provenance(tmp_path):
    """Manifest must require dynamic git status, environment details, and artifacts list."""
    from scripts.finalize_run_artifacts import build_run_manifest

    # Sample worktree status with untracked files
    worktree_text = "?? dataset_files_sha256.csv\n?? scripts/build_dataset_manifest.py\n"
    gpu_info_text = (
        "NVIDIA-SMI 572.16 Driver Version: 572.16 CUDA Version: 12.8\n"
        "NVIDIA GeForce RTX 4050 Laptop GPU"
    )

    manifest = build_run_manifest(
        run_dir=tmp_path,
        base_commit="2ec3dddc1286596c57eaa605101a6b06281e9fce",
        branch_name="pipeline-signer-split",
        worktree_status=worktree_text,
        python_version="Python 3.11.9",
        gpu_info_text=gpu_info_text,
        metrics_data={
            "runtime": {"torch": "2.12.1+cu126", "numpy": "1.26.4"},
            "device": "cuda",
            "feature_dim": 203,
            "sequence_length": 60,
        },
        sig_loso="564a3a0b32f06ac795ffa78b8d88083fa24c9a879ed55b6ac1ba2607bfbfcb2b",
        checkpoint_sha="277a8def21179bbcf96285ab9f7ef3ff22aafbfc8b61208e9b82ffcbb5236ac4",
        labels_file=Path("dataset/labels.txt"),
        plan_file=Path("dataset/recording_plan_p123.json"),
        manifest_csv=Path("dataset_files_sha256.csv"),
        fold_stats={"P01": {"accuracy": 0.84}, "P02": {"accuracy": 0.86}, "P03": {"accuracy": 0.953333}},
        all_preds=[{"correct": True}] * 398 + [{"correct": False}] * 52,
        artifacts_info=[{"filename": "test.txt", "bytes": 10, "sha256": "abc"}],
    )

    assert manifest["tracked_tree_clean"] is True
    assert manifest["worktree_clean"] is False
    assert len(manifest["untracked_at_start"]) == 2
    assert manifest["external_review_status"] == "pending"
    assert "provenance_limitations" in manifest
    assert "<RunDir>" not in manifest["command_loso"]
    assert "<RunDir>" not in manifest["command_ship"]
    assert "nvidia_driver" in manifest["environment"]
    assert "cuda_driver_version" in manifest["environment"]
    assert "gpu_name" in manifest["environment"]
    assert manifest["environment"]["gpu_name"] == "NVIDIA GeForce RTX 4050 Laptop GPU"
    assert len(manifest["artifacts"]) == 1


def test_confusion_accounting_matches_exactly_52():
    """All confusion directions must account for every single error, totaling exactly 52."""
    from scripts.finalize_run_artifacts import get_full_confusions_accounting

    loso_report_path = Path("runs/p123-clean-20260903-092539/loso_report.json")
    if not loso_report_path.is_file():
        pytest.skip("Run report not found")

    data = json.loads(loso_report_path.read_text(encoding="utf-8"))
    preds = [p for f in data["folds"] for p in f["predictions"]]
    wrong = [p for p in preds if not p["correct"]]
    assert len(wrong) == 52

    confusions = get_full_confusions_accounting(wrong)
    total_confused = sum(c["count"] for c in confusions)
    assert total_confused == 52
    assert len(confusions) == 23  # 23 distinct confusion directions


def test_forbidden_terms_rejected_by_audit():
    """Audit must reject forbidden or overstated claims."""
    from scripts.audit_run_artifacts import check_report_text_claims

    # Overstated claims must fail
    with pytest.raises(ValueError, match="Forbidden term"):
        check_report_text_claims("Báo cáo đạt kết quả 100% reproducible.")

    with pytest.raises(ValueError, match="Forbidden term"):
        check_report_text_claims("Independent reviewer PASS.")

    with pytest.raises(ValueError, match="Forbidden term"):
        check_report_text_claims("Chúng tôi có 0 blockers.")

    # Unverified claims must fail
    with pytest.raises(ValueError, match="Unverified claim"):
        check_report_text_claims("0.5 giây cuối là tĩnh.")

    with pytest.raises(ValueError, match="Unverified claim"):
        check_report_text_claims("Google Drive gốc có đúng 25 thư mục.")

    # Objective phrasing must pass
    ok_text = (
        "Metadata xác nhận clip giảm từ khoảng 10.055s xuống 9.5095s; artifact không chứng minh phần bị cắt là tĩnh.\n"
        "Manifest được nhóm chốt gồm 25 nhãn.\n"
        "External review status: PENDING.\n"
        "Machine audit passed."
    )
    check_report_text_claims(ok_text)
