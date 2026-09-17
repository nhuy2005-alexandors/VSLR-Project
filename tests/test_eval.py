import hashlib
import json
import shutil
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from prototype_3_gestures.prepare_train import file_sha256
from prototype_3_gestures.vsl3.features import FEATURE_DIM, SEQUENCE_LENGTH
from prototype_3_gestures.vsl3.model import GestureLSTM, save_checkpoint
from prototype_3_gestures.evaluate import (
    evaluate_dataset,
    evaluate_single_clip,
    compute_metrics,
    main as eval_main,
)

LABELS_FILE = Path("dataset/labels_v2_24.txt")
CANONICAL_24_LABELS = [
    line.strip()
    for line in LABELS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class TestExternalEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir).resolve()

        # Create a checkpoint with 24 canonical labels
        self.labels = list(CANONICAL_24_LABELS)
        self.model = GestureLSTM(input_dim=FEATURE_DIM, num_classes=len(self.labels))
        self.model_path = self.root / "models_dir" / "gesture_lstm.pt"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            self.model_path,
            self.model,
            self.labels,
            config={
                "input_dim": FEATURE_DIM,
                "sequence_length": SEQUENCE_LENGTH,
                "features_version": 3,
                "feature_contract": "holistic-landmarks-v3-presence-aware-group-local-z",
                "model_architecture_version": 1,
                "pooling": "fwd_last_bwd_first",
                "rejection_contract": "closed-set-reject-v1-calibration-only",
                "segmentation_contract": "timestamp-segment-v2-min-active-no-tail",
                "training_signature": "dummy_signature_123",
                "recording_plan": {"schema_version": 1, "dataset_version": "test"},
                "reject_policy": {
                    "schema_version": 1,
                    "calibrated": False,
                    "source": "not-calibrated",
                },
            },
        )
        self.initial_model_hash = file_sha256(self.model_path)

        # Create a valid dummy training manifest (24 unique 64-hex SHA-256 hashes)
        self.train_manifest_path = self.root / "training_manifest.csv"
        self.train_hashes = [f"a{i:063x}" for i in range(len(self.labels))]
        manifest_lines = [
            "relative_path,person,label,filename,bytes,sha256,width,height,fps,duration"
        ]
        for i, lbl in enumerate(self.labels):
            manifest_lines.append(
                f"P01/{lbl}/001.mov,P01,{lbl},001.mov,1000,{self.train_hashes[i]},1920,1080,60,3.0"
            )
        self.train_manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

        # Create a valid dummy run manifest matching training manifest, checkpoint, and signature
        self.run_manifest_path = self.root / "RUN_MANIFEST.json"
        self.run_manifest_data = {
            "dataset_manifest_sha256": file_sha256(self.train_manifest_path),
            "checkpoint_sha256": self.initial_model_hash,
            "training_signature": "dummy_signature_123",
        }
        self.run_manifest_path.write_text(json.dumps(self.run_manifest_data), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_valid_tree_24(self, root_data_dir: Path, signer: str = "P05", clips_per_label: int = 2):
        """Helper to create a full valid test tree: 24 labels x clips_per_label."""
        for lbl in self.labels:
            folder = root_data_dir / signer / lbl
            folder.mkdir(parents=True, exist_ok=True)
            for c_idx in range(1, clips_per_label + 1):
                clip = folder / f"{c_idx:03d}.mov"
                # distinct unique bytes for every single clip
                clip.write_bytes(f"{signer}_{lbl}_{c_idx}_{hashlib.sha256(f'{signer}_{lbl}_{c_idx}'.encode()).hexdigest()}".encode())

    def test_compute_metrics_accounting(self):
        predictions = [
            {"ground_truth": "Cảm ơn", "predicted_top1": "Cảm ơn", "confidence": 0.90, "correct": True, "accepted": True},
            {"ground_truth": "Cảm ơn", "predicted_top1": "Xin chào", "confidence": 0.80, "correct": False, "accepted": True},
            {"ground_truth": "Xin chào", "predicted_top1": "Xin chào", "confidence": 0.40, "correct": True, "accepted": False},
            {"ground_truth": "Xin chào", "predicted_top1": "Cảm ơn", "confidence": 0.30, "correct": False, "accepted": False},
        ]
        metrics = compute_metrics(predictions)
        self.assertEqual(metrics["total_clips"], 4)
        self.assertEqual(metrics["top1_correct"], 2)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.50)
        self.assertEqual(metrics["accepted_clips"], 2)
        self.assertAlmostEqual(metrics["coverage"], 0.50)
        self.assertAlmostEqual(metrics["rejection_rate"], 0.50)
        self.assertEqual(metrics["accepted_correct"], 1)
        self.assertAlmostEqual(metrics["accepted_accuracy"], 0.50)

    # =========================================================================
    # 1. Output safety
    # =========================================================================
    def test_output_dir_refuses_models_directory(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        models_dir = Path("models").resolve()

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                output_dir=models_dir,
                allow_uncalibrated=True,
            )
        self.assertIn("Safety violation", str(ctx.exception))

    def test_output_dir_refuses_subdirectory_of_models(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        models_sub = Path("models") / "evaluation_run_1"

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                output_dir=models_sub,
                allow_uncalibrated=True,
            )
        self.assertIn("Safety violation", str(ctx.exception))

    def test_output_dir_refuses_checkpoint_parent_directory(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        ckpt_parent = self.model_path.parent

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                output_dir=ckpt_parent,
                allow_uncalibrated=True,
            )
        self.assertIn("Safety violation", str(ctx.exception))

    def test_output_dir_refuses_data_directory(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                output_dir=data_dir,
                allow_uncalibrated=True,
            )
        self.assertIn("Safety violation", str(ctx.exception))

    def test_output_dir_refuses_existing_artifacts_without_overwrite(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        out_dir = self.root / "eval_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions.csv").write_text("old,results\n", encoding="utf-8")

        with self.assertRaises(FileExistsError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                output_dir=out_dir,
                allow_uncalibrated=True,
            )
        self.assertIn("already contains evaluation artifacts", str(ctx.exception))

    # =========================================================================
    # 2. Directory completeness
    # =========================================================================
    def test_directory_mode_enforces_exact_24_labels(self):
        data_dir = self.root / "missing_label_data"
        # Only create 23 labels (leave out 'Xin lỗi')
        for lbl in self.labels[:-1]:
            folder = data_dir / "P05" / lbl
            folder.mkdir(parents=True, exist_ok=True)
            for c_idx in (1, 2):
                (folder / f"{c_idx:03d}.mov").write_bytes(f"{lbl}_{c_idx}".encode())

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("missing label", str(ctx.exception).lower())
        self.assertIn("Xin lỗi", str(ctx.exception))

    def test_directory_mode_refuses_unexpected_label_directory(self):
        data_dir = self.root / "extra_label_data"
        self._create_valid_tree_24(data_dir)
        extra_dir = data_dir / "P05" / "Hẹn gặp lại"
        extra_dir.mkdir(parents=True, exist_ok=True)
        (extra_dir / "001.mov").write_bytes(b"extra")
        (extra_dir / "002.mov").write_bytes(b"extra2")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("not in checkpoint labels", str(ctx.exception))

    def test_directory_mode_enforces_exact_clips_per_label(self):
        data_dir = self.root / "wrong_clip_count_data"
        self._create_valid_tree_24(data_dir)
        # Add 3rd clip to 'Xin chào'
        (data_dir / "P05" / "Xin chào" / "003.mov").write_bytes(b"clip3")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("expected exactly 2", str(ctx.exception))

    def test_directory_mode_enforces_expected_signer(self):
        data_dir = self.root / "wrong_signer_data"
        self._create_valid_tree_24(data_dir, signer="P06")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("unexpected signer", str(ctx.exception).lower())

    # =========================================================================
    # 3. Training manifest mandatory
    # =========================================================================
    def test_training_manifest_is_mandatory_in_directory_mode(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=None,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("training manifest is required", str(ctx.exception).lower())

    def test_training_manifest_is_mandatory_in_single_clip_mode(self):
        clip = self.root / "single.mov"
        clip.write_bytes(b"single clip")

        with self.assertRaises(ValueError) as ctx:
            evaluate_single_clip(
                video_spec=f"Cảm ơn={clip}",
                model_path=self.model_path,
                training_manifest_path=None,
                run_manifest_path=self.run_manifest_path,
                allow_uncalibrated=True,
            )
        self.assertIn("training manifest is required", str(ctx.exception).lower())

    # =========================================================================
    # 4. Manifest validation (64-hex lowercase, non-empty, no duplicates)
    # =========================================================================
    def test_manifest_refuses_invalid_hex_or_non_64_char_sha256(self):
        bad_manifest = self.root / "invalid_hex_manifest.csv"
        bad_manifest.write_text(
            "relative_path,person,label,filename,bytes,sha256,width,height,fps,duration\n"
            "P01/Cảm ơn/001.mov,P01,Cảm ơn,001.mov,100,not_a_valid_sha256_hash_here,1920,1080,60,3.0\n",
            encoding="utf-8",
        )
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=bad_manifest,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("invalid sha256", str(ctx.exception).lower())

    def test_manifest_refuses_duplicate_sha256(self):
        dup_manifest = self.root / "dup_sha_manifest.csv"
        dup_sha = "b" * 64
        dup_manifest.write_text(
            f"relative_path,person,label,filename,bytes,sha256,width,height,fps,duration\n"
            f"P01/Cảm ơn/001.mov,P01,Cảm ơn,001.mov,100,{dup_sha},1920,1080,60,3.0\n"
            f"P01/Cảm ơn/002.mov,P01,Cảm ơn,002.mov,100,{dup_sha},1920,1080,60,3.0\n",
            encoding="utf-8",
        )
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=dup_manifest,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("duplicate sha-256", str(ctx.exception).lower())

    # =========================================================================
    # 5. Test-set duplicate gate
    # =========================================================================
    def test_test_set_duplicate_content_gate(self):
        data_dir = self.root / "duplicate_test_data"
        self._create_valid_tree_24(data_dir)
        # Make two clips have identical bytes
        clip1 = data_dir / "P05" / "Cảm ơn" / "001.mov"
        clip2 = data_dir / "P05" / "Xin chào" / "002.mov"
        identical_content = b"identical_content_across_test_clips"
        clip1.write_bytes(identical_content)
        clip2.write_bytes(identical_content)

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("DUPLICATE TEST CLIPS DETECTED", str(ctx.exception))

    # =========================================================================
    # 6. Unicode normalization (NFC vs NFD equivalence)
    # =========================================================================
    def test_unicode_nfc_nfd_equivalence(self):
        data_dir = self.root / "nfd_test_data"
        # Create directories with NFD encoding
        signer = "P05"
        for lbl in self.labels:
            nfd_lbl = unicodedata.normalize("NFD", lbl)
            folder = data_dir / signer / nfd_lbl
            folder.mkdir(parents=True, exist_ok=True)
            for c_idx in (1, 2):
                clip = folder / f"{c_idx:03d}.mov"
                clip.write_bytes(f"{nfd_lbl}_{c_idx}".encode("utf-8"))

        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            out_dir = self.root / "nfd_out"
            res = evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                output_dir=out_dir,
                allow_uncalibrated=True,
            )
            # All ground_truth in predictions must be canonical NFC
            for p in res["predictions"]:
                self.assertTrue(unicodedata.is_normalized("NFC", p["ground_truth"]))

    # =========================================================================
    # 7. Model invariant (requires_grad, inference_mode, checked in finally)
    # =========================================================================
    def test_model_hash_checked_in_finally_even_on_exception(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        # Tamper the model file during extraction
        def tamper_and_raise(video_path, target_len):
            self.model_path.write_bytes(b"tampered_bytes_to_fail_invariant")
            raise RuntimeError("Extraction crashed unexpectedly")

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", side_effect=tamper_and_raise):
            with self.assertRaises(RuntimeError) as ctx:
                evaluate_dataset(
                    data_dir=data_dir,
                    model_path=self.model_path,
                    training_manifest_path=self.train_manifest_path,
                    run_manifest_path=self.run_manifest_path,
                    expected_signers=["P05"],
                    clips_per_label=2,
                    allow_uncalibrated=True,
                )
            self.assertIn("Model weights file was altered", str(ctx.exception))

    # =========================================================================
    # 8. Evaluation provenance (provenance in predictions and metrics.json)
    # =========================================================================
    def test_evaluation_provenance_and_metrics_metadata(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        out_dir = self.root / "provenance_out"

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            res = evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=["P05"],
                clips_per_label=2,
                output_dir=out_dir,
                allow_uncalibrated=True,
            )

        # Check prediction rows
        for p in res["predictions"]:
            self.assertIn("video_sha256", p)
            self.assertEqual(len(p["video_sha256"]), 64)
            self.assertIn("model_sha256", p)
            self.assertEqual(p["model_sha256"], self.initial_model_hash)

        # Check metrics.json
        metrics_json = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        for field in (
            "num_signers",
            "signers",
            "label_count",
            "clips_per_label",
            "total_clips",
            "test_set_fingerprint",
            "training_manifest_path",
            "training_manifest_sha256",
            "run_manifest_path",
            "run_manifest_sha256",
            "model_sha256",
            "training_signature",
            "confidence_threshold",
            "reject_policy_calibrated",
        ):
            self.assertIn(field, metrics_json, f"Missing provenance field: {field}")

        self.assertEqual(metrics_json["num_signers"], 1)
        self.assertEqual(metrics_json["signers"], ["P05"])
        self.assertEqual(metrics_json["label_count"], 24)
        self.assertEqual(metrics_json["clips_per_label"], 2)
        self.assertEqual(metrics_json["total_clips"], 48)
        self.assertFalse(metrics_json["reject_policy_calibrated"])

    def test_single_video_mode_does_not_apply_48_clip_gate(self):
        clip = self.root / "single_test_clip.mov"
        clip.write_bytes(b"single_clip_content_here")
        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            res = evaluate_single_clip(
                video_spec=f"Cảm ơn={clip}",
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                allow_uncalibrated=True,
            )
            self.assertEqual(res["ground_truth"], "Cảm ơn")
            self.assertIn("video_sha256", res)
            self.assertIn("model_sha256", res)

    def test_cli_main_data_dir_with_overwrite(self):
        data_dir = self.root / "cli_data_dir"
        self._create_valid_tree_24(data_dir, signer="P05", clips_per_label=2)
        out_dir = self.root / "cli_out"
        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        argv = [
            "vslr-eval",
            "--data-dir",
            str(data_dir),
            "--model",
            str(self.model_path),
            "--training-manifest",
            str(self.train_manifest_path),
            "--run-manifest",
            str(self.run_manifest_path),
            "--expected-signer",
            "P05",
            "--clips-per-label",
            "2",
            "--output-dir",
            str(out_dir),
            "--allow-uncalibrated",
            "--confidence",
            "0.50",
            "--overwrite",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq),
        ):
            eval_main()

        self.assertTrue((out_dir / "predictions.csv").exists())
        self.assertTrue((out_dir / "metrics.json").exists())
        self.assertTrue((out_dir / "REPORT.md").exists())
        self.assertTrue((out_dir / "confusion_matrix.png").exists())

    def test_cli_requires_training_manifest(self):
        data_dir = self.root / "cli_no_manifest"
        self._create_valid_tree_24(data_dir)
        argv = [
            "vslr-eval",
            "--data-dir",
            str(data_dir),
            "--model",
            str(self.model_path),
            "--run-manifest",
            str(self.run_manifest_path),
            "--expected-signer",
            "P05",
            "--allow-uncalibrated",
        ]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit) as ctx:
            eval_main()
        self.assertEqual(ctx.exception.code, 2)

    # =========================================================================
    # 9. Run manifest mandatory & binding 3-way checks
    # =========================================================================
    def test_run_manifest_is_mandatory_in_directory_mode(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=None,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("run manifest is required", str(ctx.exception).lower())

    def test_run_manifest_is_mandatory_in_single_clip_mode(self):
        clip = self.root / "single.mov"
        clip.write_bytes(b"single clip")
        with self.assertRaises(ValueError) as ctx:
            evaluate_single_clip(
                video_spec=f"Cảm ơn={clip}",
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=None,
                allow_uncalibrated=True,
            )
        self.assertIn("run manifest is required", str(ctx.exception).lower())

    def test_run_manifest_refuses_mismatched_dataset_manifest_sha256(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        bad_run_manifest = self.root / "bad_dataset_run_manifest.json"
        bad_data = dict(self.run_manifest_data)
        bad_data["dataset_manifest_sha256"] = "0" * 64
        bad_run_manifest.write_text(json.dumps(bad_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=bad_run_manifest,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("dataset_manifest_sha256", str(ctx.exception).lower())

    def test_run_manifest_refuses_mismatched_checkpoint_sha256(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        bad_run_manifest = self.root / "bad_ckpt_run_manifest.json"
        bad_data = dict(self.run_manifest_data)
        bad_data["checkpoint_sha256"] = "1" * 64
        bad_run_manifest.write_text(json.dumps(bad_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=bad_run_manifest,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("checkpoint_sha256", str(ctx.exception).lower())

    def test_run_manifest_refuses_mismatched_training_signature(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        bad_run_manifest = self.root / "bad_sig_run_manifest.json"
        bad_data = dict(self.run_manifest_data)
        bad_data["training_signature"] = "mismatched_signature"
        bad_run_manifest.write_text(json.dumps(bad_data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=bad_run_manifest,
                expected_signers=["P05"],
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("training_signature", str(ctx.exception).lower())

    # =========================================================================
    # 10. Expected signer & clips-per-label validation
    # =========================================================================
    def test_directory_mode_enforces_expected_signer_mandatory(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=self.train_manifest_path,
                run_manifest_path=self.run_manifest_path,
                expected_signers=None,
                clips_per_label=2,
                allow_uncalibrated=True,
            )
        self.assertIn("expected-signer", str(ctx.exception).lower())

    def test_directory_mode_enforces_positive_clips_per_label(self):
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)

        for invalid_val in (0, -1):
            with self.assertRaises(ValueError) as ctx:
                evaluate_dataset(
                    data_dir=data_dir,
                    model_path=self.model_path,
                    training_manifest_path=self.train_manifest_path,
                    run_manifest_path=self.run_manifest_path,
                    expected_signers=["P05"],
                    clips_per_label=invalid_val,
                    allow_uncalibrated=True,
                )
            self.assertIn("positive integer", str(ctx.exception).lower())

    # =========================================================================
    # 11. Proof: P04 training clip cannot leak with P01-P03 manifest
    # =========================================================================
    def test_p04_train_clip_cannot_leak_with_p01_p03_manifest(self):
        """Proof: A P04 clip cannot leak through evaluation even if user passes a P01-P03 manifest.

        The run manifest binds the model to the authoritative 4-signer training manifest SHA-256.
        When a partial P01-P03 manifest is passed, it is rejected before extractor or inference.
        """
        p01_p03_manifest = self.root / "manifest_p01_p03.csv"
        p01_p03_manifest.write_text(
            "relative_path,person,label,filename,bytes,sha256,width,height,fps,duration\n"
            f"P01/Cảm ơn/001.mov,P01,Cảm ơn,001.mov,1000,{self.train_hashes[0]},1920,1080,60,3.0\n",
            encoding="utf-8",
        )

        p04_clip = self.root / "p04_test_clip.mov"
        p04_clip.write_bytes(b"p04_clip_content")

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video") as mock_extract:
            with self.assertRaises(ValueError) as ctx:
                evaluate_single_clip(
                    video_spec=f"Cảm ơn={p04_clip}",
                    model_path=self.model_path,
                    training_manifest_path=p01_p03_manifest,
                    run_manifest_path=self.run_manifest_path,
                    allow_uncalibrated=True,
                )
            # Extractor must never have been called
            mock_extract.assert_not_called()
            self.assertIn("dataset_manifest_sha256", str(ctx.exception).lower())

    # =========================================================================
    # 12. CLI gate error format: "error: ..." without traceback, exit 2
    # =========================================================================
    def test_cli_gate_error_prints_error_prefix_without_traceback_and_exits_2(self):
        import io
        from contextlib import redirect_stderr

        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        argv = [
            "vslr-eval",
            "--data-dir",
            str(data_dir),
            "--model",
            str(self.model_path),
            "--training-manifest",
            str(self.train_manifest_path),
            "--run-manifest",
            str(self.run_manifest_path),
            # Missing --expected-signer
            "--allow-uncalibrated",
        ]
        stderr_buf = io.StringIO()
        with mock.patch("sys.argv", argv), redirect_stderr(stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                eval_main()
            self.assertEqual(ctx.exception.code, 2)

        err_output = stderr_buf.getvalue()
        self.assertTrue(err_output.startswith("error:") or "\nerror:" in err_output, f"Got: {err_output}")
        self.assertNotIn("Traceback (most recent call last)", err_output)

    def test_cli_unexpected_programming_error_not_swallowed(self):
        """Unexpected bugs (e.g. TypeError) must not be masked as user error."""
        data_dir = self.root / "test_data"
        self._create_valid_tree_24(data_dir)
        argv = [
            "vslr-eval",
            "--data-dir",
            str(data_dir),
            "--model",
            str(self.model_path),
            "--training-manifest",
            str(self.train_manifest_path),
            "--run-manifest",
            str(self.run_manifest_path),
            "--expected-signer",
            "P05",
            "--clips-per-label",
            "2",
            "--allow-uncalibrated",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("prototype_3_gestures.evaluate.evaluate_dataset", side_effect=TypeError("Unexpected code bug")),
        ):
            with self.assertRaises(TypeError):
                eval_main()


if __name__ == "__main__":
    unittest.main()
