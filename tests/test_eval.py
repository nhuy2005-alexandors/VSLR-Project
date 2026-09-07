import shutil
import tempfile
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


class TestExternalEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

        # Create a dummy checkpoint with 3 labels
        self.labels = ["Bạn có cần giúp đỡ không", "Cảm ơn", "Xin chào"]
        self.model = GestureLSTM(input_dim=FEATURE_DIM, num_classes=len(self.labels))
        self.model_path = self.root / "test_model.pt"
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

        # Create a dummy training manifest
        self.train_manifest_path = self.root / "training_manifest.csv"
        self.dummy_train_hash = "1111111111111111111111111111111111111111111111111111111111111111"
        self.train_manifest_path.write_text(
            f"relative_path,person,label,filename,bytes,sha256,width,height,fps,duration\n"
            f"P01/Cảm ơn/001.mov,P01,Cảm ơn,001.mov,100,{self.dummy_train_hash},1920,1080,60,3.0\n",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_metrics_computation_top1_accepted_rejection_coverage(self):
        # 4 samples:
        # Sample 1: GT=Cảm ơn, Pred=Cảm ơn, Conf=0.90 (Accepted, Correct)
        # Sample 2: GT=Cảm ơn, Pred=Xin chào, Conf=0.80 (Accepted, Wrong)
        # Sample 3: GT=Xin chào, Pred=Xin chào, Conf=0.40 (Rejected, Correct)
        # Sample 4: GT=Xin chào, Pred=Cảm ơn, Conf=0.30 (Rejected, Wrong)
        predictions = [
            {"ground_truth": "Cảm ơn", "predicted": "Cảm ơn", "confidence": 0.90, "correct": True, "accepted": True},
            {"ground_truth": "Cảm ơn", "predicted": "Xin chào", "confidence": 0.80, "correct": False, "accepted": True},
            {"ground_truth": "Xin chào", "predicted": "Xin chào", "confidence": 0.40, "correct": True, "accepted": False},
            {"ground_truth": "Xin chào", "predicted": "Cảm ơn", "confidence": 0.30, "correct": False, "accepted": False},
        ]
        metrics = compute_metrics(predictions)
        self.assertEqual(metrics["total_clips"], 4)
        self.assertEqual(metrics["top1_correct"], 2)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.50)  # 2/4
        self.assertEqual(metrics["accepted_clips"], 2)
        self.assertAlmostEqual(metrics["coverage"], 0.50)  # 2/4
        self.assertAlmostEqual(metrics["rejection_rate"], 0.50)  # 2/4
        self.assertEqual(metrics["accepted_correct"], 1)
        self.assertAlmostEqual(metrics["accepted_accuracy"], 0.50)  # 1/2

    def test_detects_test_train_hash_overlap(self):
        # Create a test dataset where one clip has the exact same bytes as dummy_train_hash
        data_dir = self.root / "test_data"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        overlap_clip = label_dir / "001.mov"
        # We write dummy bytes whose sha256 matches dummy_train_hash or mock file_sha256
        overlap_clip.write_bytes(b"dummy video content")

        with mock.patch("prototype_3_gestures.evaluate.file_sha256", return_value=self.dummy_train_hash):
            with self.assertRaises(ValueError) as ctx:
                evaluate_dataset(
                    data_dir=data_dir,
                    model_path=self.model_path,
                    training_manifest_path=self.train_manifest_path,
                    allow_uncalibrated=True,
                )
            self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))

    def test_rejects_label_not_in_checkpoint(self):
        # Create a test dataset with an unknown label
        data_dir = self.root / "test_data_invalid_label"
        label_dir = data_dir / "P05" / "Nhãn Không Tồn Tại"
        label_dir.mkdir(parents=True)
        clip = label_dir / "001.mov"
        clip.write_bytes(b"some content")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                allow_uncalibrated=True,
            )
        self.assertIn("not in checkpoint labels", str(ctx.exception))

    def test_rejects_empty_data_dir(self):
        data_dir = self.root / "empty_data"
        data_dir.mkdir()
        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                allow_uncalibrated=True,
            )
        self.assertIn("No .mov/.mp4 clips found", str(ctx.exception))

    def test_prediction_row_completeness_and_top3(self):
        data_dir = self.root / "test_data_valid"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        clip = label_dir / "001.mov"
        clip.write_bytes(b"valid clip bytes")

        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            output_dir = self.root / "eval_output"
            result = evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                output_dir=output_dir,
                allow_uncalibrated=True,
                confidence_threshold=0.50,
            )
            preds = result["predictions"]
            self.assertEqual(len(preds), 1)
            row = preds[0]
            for key in ("video_path", "signer", "ground_truth", "predicted", "confidence", "top3", "accepted", "correct"):
                self.assertIn(key, row)
            self.assertEqual(len(row["top3"]), 3)
            self.assertEqual(row["ground_truth"], "Cảm ơn")
            self.assertEqual(row["signer"], "P05")

    def test_model_weights_invariant_after_evaluation(self):
        data_dir = self.root / "test_data_invariant"
        label_dir = data_dir / "P05" / "Xin chào"
        label_dir.mkdir(parents=True)
        clip = label_dir / "001.mov"
        clip.write_bytes(b"clip bytes")

        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            output_dir = self.root / "eval_output"
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                output_dir=output_dir,
                allow_uncalibrated=True,
            )

        # Verify model file SHA-256 is strictly unchanged
        after_hash = file_sha256(self.model_path)
        self.assertEqual(after_hash, self.initial_model_hash)

    def test_output_artifacts_created_in_output_dir(self):
        data_dir = self.root / "test_artifacts"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        clip = label_dir / "001.mov"
        clip.write_bytes(b"clip bytes")

        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        output_dir = self.root / "eval_results_dir"

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                output_dir=output_dir,
                allow_uncalibrated=True,
            )

        self.assertTrue((output_dir / "predictions.csv").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        self.assertTrue((output_dir / "REPORT.md").exists())
        self.assertTrue((output_dir / "confusion_matrix.png").exists())
        # Verify nothing written to models/
        self.assertFalse((Path("models") / "predictions.csv").exists())

    def test_windows_unicode_paths(self):
        # Unicode folder name with Vietnamese accents
        data_dir = self.root / "bộ_kiểm_thử_tiếng_việt"
        label_dir = data_dir / "Người_ký_05" / "Bạn có cần giúp đỡ không"
        label_dir.mkdir(parents=True)
        clip = label_dir / "001_cử_chỉ.mov"
        clip.write_bytes(b"unicode clip")

        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            output_dir = self.root / "kết_quả_đánh_giá"
            result = evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                output_dir=output_dir,
                allow_uncalibrated=True,
            )
            self.assertEqual(len(result["predictions"]), 1)
            self.assertEqual(result["predictions"][0]["ground_truth"], "Bạn có cần giúp đỡ không")

    def test_single_video_mode(self):
        clip = self.root / "single_test.mov"
        clip.write_bytes(b"single clip")

        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        with mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq):
            pred = evaluate_single_clip(
                video_spec="Xin chào=" + str(clip),
                model_path=self.model_path,
                allow_uncalibrated=True,
                confidence_threshold=0.50,
            )
            self.assertEqual(pred["ground_truth"], "Xin chào")
            self.assertIn("predicted", pred)
            self.assertIn("confidence", pred)
            self.assertIn("top3", pred)
            self.assertIn("accepted", pred)
            self.assertIn("correct", pred)

    def test_cli_main_single_video(self):
        clip = self.root / "single_cli_test.mov"
        clip.write_bytes(b"single clip bytes")
        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        argv = [
            "vslr-eval",
            "--video",
            f"Cảm ơn={clip}",
            "--model",
            str(self.model_path),
            "--allow-uncalibrated",
            "--confidence",
            "0.50",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq),
        ):
            eval_main()

    def test_cli_main_data_dir(self):
        data_dir = self.root / "cli_data"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        clip = label_dir / "001.mov"
        clip.write_bytes(b"cli test clip")
        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
        output_dir = self.root / "cli_output"

        argv = [
            "vslr-eval",
            "--data-dir",
            str(data_dir),
            "--model",
            str(self.model_path),
            "--output-dir",
            str(output_dir),
            "--allow-uncalibrated",
            "--confidence",
            "0.50",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq),
        ):
            eval_main()

        self.assertTrue((output_dir / "predictions.csv").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        self.assertTrue((output_dir / "REPORT.md").exists())

    def test_uncalibrated_reject_policy_raises_without_flag(self):
        data_dir = self.root / "test_uncalibrated"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        (label_dir / "001.mov").write_bytes(b"content")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                allow_uncalibrated=False,
            )
        self.assertIn("Reject policy is not calibrated", str(ctx.exception))

    def test_manifest_without_sha256_column_raises(self):
        bad_manifest = self.root / "bad_manifest.csv"
        bad_manifest.write_text("video_path,label\n001.mov,Cảm ơn\n", encoding="utf-8")
        data_dir = self.root / "test_manifest_err"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        (label_dir / "001.mov").write_bytes(b"content")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=bad_manifest,
                allow_uncalibrated=True,
            )
        self.assertIn("missing required 'sha256' column", str(ctx.exception))

    def test_empty_manifest_raises(self):
        empty_manifest = self.root / "empty_manifest.csv"
        empty_manifest.write_text("relative_path,sha256\n", encoding="utf-8")
        data_dir = self.root / "test_empty_manifest"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        (label_dir / "001.mov").write_bytes(b"content")

        with self.assertRaises(ValueError) as ctx:
            evaluate_dataset(
                data_dir=data_dir,
                model_path=self.model_path,
                training_manifest_path=empty_manifest,
                allow_uncalibrated=True,
            )
        self.assertIn("No valid SHA-256 hashes found", str(ctx.exception))

    def test_detects_test_train_hash_overlap_single_video_mode(self):
        clip = self.root / "single_overlap.mov"
        clip.write_bytes(b"overlap clip")

        with mock.patch("prototype_3_gestures.evaluate.file_sha256", return_value=self.dummy_train_hash):
            with self.assertRaises(ValueError) as ctx:
                evaluate_single_clip(
                    video_spec=f"Cảm ơn={clip}",
                    model_path=self.model_path,
                    training_manifest_path=self.train_manifest_path,
                    allow_uncalibrated=True,
                )
            self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))

    def test_model_file_tampering_raises_runtime_error(self):
        data_dir = self.root / "test_tampering"
        label_dir = data_dir / "P05" / "Cảm ơn"
        label_dir.mkdir(parents=True)
        (label_dir / "001.mov").write_bytes(b"clip")
        dummy_seq = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)

        hashes = [self.initial_model_hash, "tampered_hash_999"]

        def mock_sha(path):
            if Path(path) == self.model_path:
                return hashes.pop(0) if hashes else "tampered_hash_999"
            return "regular_hash"

        with (
            mock.patch("prototype_3_gestures.evaluate.file_sha256", side_effect=mock_sha),
            mock.patch("prototype_3_gestures.evaluate.extract_sequence_from_video", return_value=dummy_seq),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                evaluate_dataset(
                    data_dir=data_dir,
                    model_path=self.model_path,
                    allow_uncalibrated=True,
                )
            self.assertIn("Model weights file was altered", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
