import csv
import json
import os
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from prototype_3_gestures.prepare_train import Clip, validate_recording_tree
from prototype_3_gestures.realtime import Segment, SegmentTracker, classify_segment_frames
from prototype_3_gestures.sentence import edit_distance, parse_sentence_manifest, word_error_rate
from prototype_3_gestures.vsl3.features import (
    FEATURE_DIM,
    LANDMARK_FEATURE_DIM,
    SEQUENCE_LENGTH,
    bridge_missing_hand_gaps,
    preprocess_sequence,
    normalize_landmarks,
    require_valid_video_fps,
    trim_active_frames,
    _sample_presence_aware,
)
from prototype_3_gestures.vsl3.model import GestureLSTM, load_checkpoint, save_checkpoint
from prototype_3_gestures.vsl3.recording_plan import (
    RecordingPlan,
    load_recording_plan,
    save_recording_plan,
)
from prototype_3_gestures.vsl3.reject import (
    CalibrationLeakageError,
    RejectPolicy,
    fit_reject_policy,
    load_reject_policy,
    reject_prediction,
    save_reject_policy,
)


class RecordingPlanTests(unittest.TestCase):
    def _plan(self, root: Path) -> RecordingPlan:
        return RecordingPlan(
            schema_version=1,
            dataset_version="recordings_v1",
            labels_file="labels.txt",
            people=("P01", "P02", "P03", "P04"),
            clips_per_label=6,
        )

    def test_plan_is_versioned_and_does_not_duplicate_labels(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            path = root / "recording_plan.json"
            save_recording_plan(path, plan)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["people"], ["P01", "P02", "P03", "P04"])
            self.assertNotIn("labels", raw)
            self.assertEqual(load_recording_plan(path), plan)

    def test_tree_gate_sees_empty_signer_and_exact_counts(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            labels = ["A", "B"]
            (root / "P01" / "A").mkdir(parents=True)
            (root / "P01" / "B").mkdir()
            (root / "P02").mkdir()
            result = validate_recording_tree(root, plan, labels, strict_counts=True)
            self.assertTrue(any("P02" in error for error in result.errors))
            self.assertTrue(any("P03" in error for error in result.errors))
            self.assertTrue(any("P01" in error and "A" in error for error in result.errors))

    def test_tree_gate_rejects_overfull_and_duplicate_content_before_extraction(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = RecordingPlan(1, "v1", "labels.txt", ("P01", "P02", "P03", "P04"), 6)
            for person in plan.people:
                (root / person / "A").mkdir(parents=True)
            for index in range(1, 8):
                (root / "P01" / "A" / f"{index:03d}.mov").write_bytes(
                    b"same" if index == 1 else f"P01-{index}".encode()
                )
            for index in range(1, 7):
                (root / "P02" / "A" / f"{index:03d}.mov").write_bytes(
                    b"same" if index == 1 else f"P02-{index}".encode()
                )
            for person in ("P03", "P04"):
                for index in range(1, 7):
                    (root / person / "A" / f"{index:03d}.mov").write_bytes(f"{person}-{index}".encode())
            result = validate_recording_tree(root, plan, ["A"], strict_counts=True)
            self.assertTrue(any("overfull" in error for error in result.errors))
            duplicate = [error for error in result.errors if "duplicate" in error.lower()]
            self.assertEqual(len(duplicate), 1)
            self.assertIn("P01", duplicate[0])
            self.assertIn("P02", duplicate[0])

    def test_zero_clip_quota_is_refused(self):
        with self.assertRaises(ValueError):
            RecordingPlan(1, "v1", "labels.txt", ("P01", "P02", "P03", "P04"), 0)

    def test_plan_rejects_non_primitive_or_non_v1_quota_values(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            for value in (6.9, "6", True, 4):
                payload = {
                    "schema_version": 1,
                    "dataset_version": "recordings_v1",
                    "labels_file": "labels.txt",
                    "people": ["P01", "P02", "P03", "P04"],
                    "clips_per_label": value,
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        load_recording_plan(path)


class PresencePreprocessingTests(unittest.TestCase):
    def _sequence(self, n=9):
        data = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        data[:, 0] = np.arange(n, dtype=np.float32)
        data[:, 50:LANDMARK_FEATURE_DIM] = 2.0
        return data

    def test_short_bounded_gap_is_bridged_only_for_that_hand(self):
        sequence = self._sequence()
        left = np.array([True, True, False, False, True, True, True, True, True])
        right = np.ones(9, dtype=bool)
        output, left_out, right_out = bridge_missing_hand_gaps(
            sequence, left, right, timestamps=np.arange(9, dtype=float) / 30.0, max_gap_seconds=0.1
        )
        self.assertTrue(left_out[2:4].all())
        self.assertTrue(right_out.all())
        self.assertTrue(np.any(output[2, LANDMARK_FEATURE_DIM - 2 : LANDMARK_FEATURE_DIM] > 0))

    def test_long_and_one_sided_gaps_remain_missing_without_ghost_values(self):
        sequence = self._sequence(12)
        left = np.array([True, True, False, False, False, False, False, False, False, True, True, True])
        right = np.array([True, True, True, True, True, False, False, False, False, False, False, False])
        output, left_out, right_out = bridge_missing_hand_gaps(
            sequence, left, right, timestamps=np.arange(12, dtype=float) / 30.0, max_gap_seconds=0.1
        )
        self.assertFalse(left_out[2:9].any())
        self.assertFalse(right_out[5:].any())
        self.assertTrue(np.all(output[2:9, 75:138] == 0.0))

    def test_presence_aware_resample_does_not_interpolate_long_missing_hand(self):
        sequence = self._sequence(10)
        left = np.array([True, True, False, False, False, False, False, False, True, True])
        right = np.ones(10, dtype=bool)
        output = preprocess_sequence(
            sequence,
            left_hand_present=left,
            right_hand_present=right,
            timestamps=np.arange(10, dtype=float) / 30.0,
            target_len=SEQUENCE_LENGTH,
            max_gap_seconds=0.05,
        )
        self.assertTrue(np.all(output[15:45, 75:138] == 0.0))
        self.assertTrue(np.all(output[15:45, -2] == 0.0))
        self.assertTrue(np.all(output[:, -1] == 1.0))

    def test_realtime_and_offline_call_the_same_preprocessing_function(self):
        import prototype_3_gestures.realtime as realtime
        import prototype_3_gestures.vsl3.features as features

        self.assertIs(realtime.preprocess_sequence, features.preprocess_sequence)

    def test_realtime_and_video_trim_have_identical_frames_masks_and_times(self):
        sequence = self._sequence(6)
        left = np.array([False, True, False, False, True, False])
        right = np.array([False, False, True, False, False, False])
        times = np.arange(6, dtype=np.float64) / 30.0
        video_trimmed = trim_active_frames(sequence, left, right, times)
        segment = Segment(
            features=[row for row in sequence],
            start_time=times[1],
            active_end_time=times[4],
            end_time=times[5],
            forced=False,
            timestamps=times.tolist(),
            left_hand_present=left.tolist(),
            right_hand_present=right.tolist(),
        )
        realtime_trimmed = segment.active_frames()
        np.testing.assert_array_equal(video_trimmed[0], np.stack(realtime_trimmed[0]))
        np.testing.assert_array_equal(video_trimmed[1], realtime_trimmed[1])
        np.testing.assert_array_equal(video_trimmed[2], realtime_trimmed[2])
        np.testing.assert_array_equal(video_trimmed[3], realtime_trimmed[3])
        self.assertEqual(
            preprocess_sequence(video_trimmed[0], left_hand_present=video_trimmed[1], right_hand_present=video_trimmed[2], timestamps=video_trimmed[3]).tolist(),
            preprocess_sequence(np.stack(realtime_trimmed[0]), left_hand_present=realtime_trimmed[1], right_hand_present=realtime_trimmed[2], timestamps=realtime_trimmed[3]).tolist(),
        )

    def test_invalid_video_fps_is_not_replaced_with_a_fallback(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "invalid FPS"):
                    require_valid_video_fps(value, "clip.mov")

    def test_presence_aware_sampler_keeps_exact_integer_observations(self):
        sequence = self._sequence(3)
        left = np.array([True, True, False])
        right = np.array([False, True, True])
        output = _sample_presence_aware(
            sequence, np.array([0.0, 1.0, 2.0]), left, right
        )
        self.assertEqual(output[:, -2].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(output[:, -1].tolist(), [0.0, 1.0, 1.0])

    def test_gap_between_active_endpoints_is_retained_and_not_bridged_when_long(self):
        sequence = np.zeros((5, FEATURE_DIM), dtype=np.float32)
        sequence[0, 75:138] = 1.0
        sequence[4, 75:138] = 5.0
        sequence[[0, 4], -2] = 1.0
        segment = Segment(
            features=[row for row in sequence],
            start_time=0.0,
            active_end_time=0.4,
            end_time=0.8,
            forced=False,
            timestamps=[0.0, 0.1, 0.2, 0.3, 0.4],
            left_hand_present=[True, False, False, False, True],
            right_hand_present=[False] * 5,
        )
        active, left, right, times = classify_segment_frames(segment)
        self.assertEqual(len(active), 5, "internal missing frames must reach shared preprocessing")
        output = preprocess_sequence(
            np.stack(active),
            left_hand_present=left,
            right_hand_present=right,
            timestamps=times,
            max_gap_seconds=0.2,
            target_len=5,
        )
        self.assertTrue(np.all(output[1:4, 75:138] == 0.0))
        self.assertTrue(np.all(output[1:4, -2] == 0.0))

    def test_missing_hand_coordinates_are_zero_even_when_explicit_origin_is_nonzero(self):
        points = np.zeros((25 + 21 * 2, 3), dtype=np.float32)
        points[11] = [0.4, 0.5, 2.0]
        points[12] = [0.6, 0.5, 2.0]
        points[25:46] = 3.0
        output = normalize_landmarks(points, left_hand_present=False, right_hand_present=True)
        self.assertTrue(np.all(output[75:138] == 0.0))
        self.assertEqual(float(output[-2]), 0.0)
        self.assertEqual(float(output[-1]), 1.0)

    def test_partial_malformed_missing_hand_does_not_affect_normalization(self):
        points = np.zeros((25 + 21 * 2, 3), dtype=np.float32)
        points[11] = [0.4, 0.5, 1.0]
        points[12] = [0.6, 0.5, 1.0]
        # Only a bogus left wrist-z is non-zero; explicit presence=false must exclude the entire
        # group from valid/center/scale/origin and leave every left coordinate zero.
        points[25, 2] = 99.0
        points[46:67] = [0.8, 0.7, 2.0]
        output = normalize_landmarks(points, left_hand_present=False, right_hand_present=True)
        self.assertTrue(np.all(output[75:138] == 0.0))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.all(np.abs(output[138:201]) <= 6.0))

    def test_augmentation_marks_hand_missing_when_drop_neighbors_are_missing(self):
        sequence = np.zeros((60, FEATURE_DIM), dtype=np.float32)
        sequence[:, 0] = np.linspace(0.0, 1.0, 60)
        sequence[:, -2] = 0.0
        sequence[:, -1] = 1.0
        output = __import__("prototype_3_gestures.vsl3.features", fromlist=["augment_sequence"]).augment_sequence(
            sequence, np.random.default_rng(4)
        )
        # The invariant is checked on every output frame: absent left hand never has coordinates
        # or a stale presence bit. (All frames are absent in this synthetic input.)
        self.assertTrue(np.all(output[:, 75:138] == 0.0))
        self.assertTrue(np.all(output[:, -2] == 0.0))


class SegmentationHardeningTests(unittest.TestCase):
    def test_idle_blip_does_not_close_as_active_segment_when_min_evidence_is_set(self):
        tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0, min_active_seconds=0.2)
        frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        frame[-2] = 1.0
        self.assertIsNone(tracker.feed(True, frame, 0.0))
        self.assertIsNone(tracker.feed(False, np.zeros(FEATURE_DIM, dtype=np.float32), 0.1))
        self.assertIsNone(tracker.feed(False, np.zeros(FEATURE_DIM, dtype=np.float32), 0.6))
        self.assertFalse(tracker.in_segment)

    def test_no_hand_tail_is_excluded_before_model_preprocessing(self):
        features = [np.ones(FEATURE_DIM, dtype=np.float32) for _ in range(4)]
        left = [True, True, True, False]
        right = [False, False, False, False]
        segment = Segment(
            features=features,
            start_time=0.0,
            active_end_time=0.2,
            end_time=0.65,
            forced=False,
            timestamps=[0.0, 0.1, 0.2, 0.65],
            left_hand_present=left,
            right_hand_present=right,
        )
        active, active_left, active_right, times = segment.active_frames()
        self.assertEqual(len(active), 3)
        self.assertEqual(active_left, [True, True, True])
        self.assertEqual(active_right, [False, False, False])

    def test_tracker_rejects_non_monotonic_timestamps(self):
        tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0)
        frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        frame[-2] = 1.0
        tracker.feed(True, frame, 1.0)
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            tracker.feed(True, np.zeros(FEATURE_DIM, dtype=np.float32), 0.9)

    def test_tracker_rejects_hands_present_mismatch_with_authoritative_masks(self):
        frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        frame[-2] = 1.0
        tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0)
        with self.assertRaisesRegex(ValueError, "presence"):
            tracker.feed(False, frame, 0.0)

    def test_tracker_rejects_non_monotonic_force_boundary_and_invalid_presence_channel(self):
        frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        frame[-2] = 1.0
        tracker = SegmentTracker(word_gap=0.45, max_seconds=5.0)
        tracker.feed(True, frame, 1.0)
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            tracker.force_boundary(0.9)
        bad = frame.copy()
        bad[-2] = 2.0
        fresh = SegmentTracker(word_gap=0.45, max_seconds=5.0)
        with self.assertRaisesRegex(ValueError, "presence"):
            fresh.feed(True, bad, 0.0)

    def test_any_max_boundary_waits_for_a_true_hand_drop(self):
        tracker = SegmentTracker(word_gap=2.0, max_seconds=1.0)
        frame = np.zeros(FEATURE_DIM, dtype=np.float32)
        frame[-2] = 1.0
        self.assertIsNone(tracker.feed(True, frame, 0.0))
        done = tracker.feed(False, np.zeros(FEATURE_DIM, dtype=np.float32), 1.1)
        self.assertIsNotNone(done)
        self.assertFalse(done.forced, "the cap was crossed in the no-hand tail; boundary is not forced")
        self.assertTrue(tracker.awaiting_hand_drop)
        self.assertIsNone(tracker.feed(True, frame, 1.2))
        self.assertFalse(tracker.in_segment)


class RejectCalibrationTests(unittest.TestCase):
    def test_calibration_rejects_loso_test_rows_as_leakage(self):
        rows = [
            {"confidence": 0.9, "is_negative": False, "split": "calibration", "source": "calib-a"},
            {"confidence": 0.8, "is_negative": True, "split": "loso_test", "source": "calib-a"},
        ]
        with self.assertRaises(CalibrationLeakageError):
            fit_reject_policy(rows)

    def test_calibration_requires_negative_data(self):
        with self.assertRaises(ValueError):
            fit_reject_policy([{"confidence": 0.9, "is_negative": False, "split": "calibration"}])

    def test_reject_policy_fails_closed_for_nonfinite_or_invalid_probabilities(self):
        policy = RejectPolicy(True, threshold=0.5)
        cases = [
            (float("nan"), np.array([0.9, 0.1])),
            (float("inf"), np.array([0.9, 0.1])),
            (0.9, np.array([np.nan, 0.1])),
            (0.9, np.array([1.2, -0.2])),
            (0.9, np.array([[0.9, 0.1]])),
            (0.9, np.array([0.9, 0.2])),
        ]
        for confidence, probabilities in cases:
            with self.subTest(confidence=confidence, probabilities=probabilities):
                accepted, reason = reject_prediction(
                    confidence,
                    probabilities,
                    policy,
                    manual_threshold=0.5,
                )
                self.assertFalse(accepted)
                self.assertTrue(reason)

    def test_calibration_requires_explicit_non_test_source(self):
        rows = [
            {"confidence": 0.9, "is_negative": False, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "pos.mov", "source_sha256": "a" * 64, "category": "in_dictionary"},
            {"confidence": 0.1, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "idle.mov", "source_sha256": "b" * 64, "category": "idle_stationary"},
            {"confidence": 0.2, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "oov.mov", "source_sha256": "c" * 64, "category": "oov_motion"},
        ]
        with self.assertRaisesRegex(ValueError, "feature_contract"):
            fit_reject_policy(rows)

    def test_calibration_parses_false_as_false_not_true(self):
        rows = [
            {"confidence": 0.9, "is_negative": "false", "split": "calibration", "source": "calib-a", "person": "P01", "clip": "pos.mov", "source_sha256": "a" * 64, "category": "in_dictionary"},
            {"confidence": 0.1, "is_negative": "true", "split": "calibration", "source": "calib-a", "person": "P01", "clip": "idle.mov", "source_sha256": "b" * 64, "category": "idle_stationary"},
            {"confidence": 0.2, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "oov.mov", "source_sha256": "c" * 64, "category": "oov_motion"},
        ]
        policy = fit_reject_policy(rows, feature_contract="f", rejection_contract="r", training_signature="t", checkpoint_sha256="d" * 64)
        self.assertTrue(policy.calibrated)

    def test_calibration_rejects_ambiguous_boolean_text(self):
        rows = [
            {"confidence": 0.9, "is_negative": "FALSE?", "split": "calibration", "source": "calib-a", "person": "P01", "clip": "pos.mov", "source_sha256": "a" * 64, "category": "in_dictionary"},
            {"confidence": 0.1, "is_negative": "true", "split": "calibration", "source": "calib-a", "person": "P01", "clip": "idle.mov", "source_sha256": "b" * 64, "category": "idle_stationary"},
        ]
        with self.assertRaises(ValueError):
            fit_reject_policy(rows, feature_contract="f", rejection_contract="r", training_signature="t", checkpoint_sha256="d" * 64)

    def test_saved_policy_has_receipt_and_is_bound_to_checkpoint_contract(self):
        rows = [
            {"confidence": 0.9, "is_negative": False, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "pos.mov", "source_sha256": "a" * 64, "category": "in_dictionary"},
            {"confidence": 0.1, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "idle.mov", "source_sha256": "b" * 64, "category": "idle_stationary"},
            {"confidence": 0.2, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "oov.mov", "source_sha256": "c" * 64, "category": "oov_motion"},
        ]
        policy = fit_reject_policy(rows, feature_contract="f", rejection_contract="r", training_signature="t", checkpoint_sha256="d" * 64)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reject.json"
            save_reject_policy(path, policy)
            loaded = load_reject_policy(path, expected_feature_contract="f", expected_training_signature="t")
            self.assertTrue(loaded.has_receipt)
            with self.assertRaisesRegex(ValueError, "training_signature"):
                load_reject_policy(path, expected_feature_contract="f", expected_training_signature="other")
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                load_reject_policy(path, expected_checkpoint_sha256="e" * 64)

    def test_calibration_rejects_forbidden_identity_or_content_hash_overlap(self):
        rows = [
            {"confidence": 0.9, "is_negative": False, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "pos.mov", "source_sha256": "a" * 64, "category": "in_dictionary"},
            {"confidence": 0.1, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "idle.mov", "source_sha256": "b" * 64, "category": "idle_stationary"},
            {"confidence": 0.2, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "oov.mov", "source_sha256": "c" * 64, "category": "oov_motion"},
        ]
        kwargs = dict(feature_contract="f", rejection_contract="r", training_signature="t", checkpoint_sha256="d" * 64)
        with self.assertRaises(CalibrationLeakageError):
            fit_reject_policy(rows, forbidden_identities=[("P01", "pos.mov")], **kwargs)
        with self.assertRaises(CalibrationLeakageError):
            fit_reject_policy(rows, forbidden_source_sha256=["b" * 64], **kwargs)

    def test_calibration_requires_idle_and_oov_categories(self):
        rows = [
            {"confidence": 0.9, "is_negative": False, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "pos.mov", "source_sha256": "a" * 64, "category": "in_dictionary"},
            {"confidence": 0.1, "is_negative": True, "split": "calibration", "source": "calib-a", "person": "P01", "clip": "idle.mov", "source_sha256": "b" * 64, "category": "idle_stationary"},
        ]
        with self.assertRaisesRegex(ValueError, "oov_motion"):
            fit_reject_policy(rows, feature_contract="f", rejection_contract="r", training_signature="t", checkpoint_sha256="d" * 64)


class CheckpointContractTests(unittest.TestCase):
    def test_same_feature_version_without_contract_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-contract.pt"
            save_checkpoint(
                path,
                GestureLSTM(FEATURE_DIM, 2),
                ["A", "B"],
                {
                    "input_dim": FEATURE_DIM,
                    "sequence_length": SEQUENCE_LENGTH,
                    "features_version": 3,
                },
            )
            with self.assertRaisesRegex(ValueError, "feature_contract"):
                load_checkpoint(path)

    def test_wrong_input_dim_is_refused_even_when_version_matches(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong-dim.pt"
            save_checkpoint(
                path,
                GestureLSTM(FEATURE_DIM, 2),
                ["A", "B"],
                {
                    "input_dim": FEATURE_DIM - 1,
                    "sequence_length": SEQUENCE_LENGTH,
                    "features_version": 3,
                    "feature_contract": "holistic-landmarks-v3-presence-aware-group-local-z",
                },
            )
            with self.assertRaisesRegex(ValueError, "input_dim"):
                load_checkpoint(path)

    def test_legacy_incompatible_override_is_no_longer_a_public_fallback(self):
        from prototype_3_gestures.realtime import build_parser as build_realtime_parser
        from prototype_3_gestures.sentence import build_parser as build_sentence_parser

        with self.assertRaises(SystemExit):
            build_realtime_parser().parse_args(["--allow-incompatible-model"])
        with self.assertRaises(SystemExit):
            build_sentence_parser().parse_args(["--clip", "clip.mov", "--allow-incompatible-model"])


class SentenceHelpersTests(unittest.TestCase):
    def test_edit_distance_and_wer_and_csv_ground_truth(self):
        self.assertEqual(edit_distance(["A", "B"], ["A", "C", "B"]), 1)
        self.assertAlmostEqual(word_error_rate(["A", "B"], ["A", "C"]), 0.5)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentences.csv"
            path.write_text("person,clip,words\nP01,cau_01.mov,A|B\nP01,idle.mov,\n", encoding="utf-8")
            rows = parse_sentence_manifest(path)
        self.assertEqual(rows[0].words, ["A", "B"])
        self.assertTrue(rows[1].negative)

    def test_sentence_manifest_rejects_duplicate_and_traversal_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate = root / "duplicate.csv"
            duplicate.write_text(
                "person,clip,words\nP01,cau_01.mov,A\nP01,cau_01.mov,B\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_sentence_manifest(duplicate)
            traversal = root / "traversal.csv"
            traversal.write_text("person,clip,words\nP01,../P02/cau.mov,A\n", encoding="utf-8")
            rows = parse_sentence_manifest(traversal)
            from prototype_3_gestures.sentence import _resolve_manifest_clip

            with self.assertRaises(ValueError):
                _resolve_manifest_clip(root, rows[0])


if __name__ == "__main__":
    unittest.main()
