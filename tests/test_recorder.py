from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from prototype_3_gestures.recorder import (
    GestureVideoRecorder,
    RecordedFrame,
    draw_unicode_text,
    resolve_record_dir,
    slugify,
)


class TestRecorder(TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Xin chào"), "xin_chao")
        self.assertEqual(slugify("Cảm ơn"), "cam_on")
        self.assertEqual(slugify("Bạn có cần giúp đỡ không"), "ban_co_can_giup_do_khong")
        self.assertEqual(slugify("123 Test!"), "123_test")
        self.assertEqual(slugify(""), "unknown")

    def test_draw_unicode_text(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        drawn = draw_unicode_text(img, "Xin chào", (10, 20), font_size=18, color_bgr=(255, 255, 255))
        self.assertEqual(drawn.shape, img.shape)
        self.assertGreater(np.sum(drawn), 0)

    def test_recorder_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            recorder = GestureVideoRecorder(record_dir=tmp_path, fps=30.0, enabled=True, pre_roll_seconds=0.2)

            class DummyObs:
                results = None
                left_hand_present = True
                right_hand_present = False

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            obs = DummyObs()

            # 1. Feed pre-roll frames
            t0 = 100.0
            for i in range(5):
                recorder.feed_frame(frame, obs, in_segment=False, now=t0 + i * 0.033)

            self.assertFalse(recorder.recording_active)
            self.assertEqual(len(recorder.pre_roll_buffer), 5)

            # 2. Start gesture
            for i in range(5, 15):
                recorder.feed_frame(frame, obs, in_segment=True, now=t0 + i * 0.033)

            self.assertTrue(recorder.recording_active)
            # Should contain 5 pre-roll frames + 10 gesture frames = 15 frames
            self.assertEqual(len(recorder.current_gesture_frames), 15)

            # 3. Save gesture
            class DummyDecision:
                label = "Xin chào"
                confidence = 0.885
                accepted = True
                reason = "accepted"
                probabilities = np.array([0.885, 0.065, 0.05], dtype=np.float32)

            labels = ["Xin chào", "Tạm biệt", "Cảm ơn"]
            recorder.save_gesture(DummyDecision(), labels=labels, duration=0.5)

            self.assertFalse(recorder.recording_active)
            self.assertEqual(len(recorder.current_gesture_frames), 0)

            # Close recorder to wait for worker thread to finish
            recorder.close()

            # Verify files created in tmp_path (both skeleton and raw)
            mp4_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
            json_files = list(tmp_path.glob("*.json"))

            self.assertEqual(len(mp4_files), 2, f"Expected 2 video files (skeleton + raw), found {mp4_files}")
            self.assertEqual(len(json_files), 1, f"Expected 1 json file, found {json_files}")

            video_names = [f.name for f in mp4_files]
            self.assertTrue(any("_skeleton" in name for name in video_names))
            self.assertTrue(any("_raw" in name for name in video_names))

            json_file = json_files[0]
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual(meta["label"], "Xin chào")
            self.assertEqual(meta["status"], "ACCEPTED")
            self.assertAlmostEqual(meta["confidence"], 0.885, places=2)
            self.assertEqual(len(meta["top_candidates"]), 3)
            self.assertEqual(meta["top_candidates"][0]["label"], "Xin chào")
            self.assertEqual(meta["hand_tracking"]["left_frames"], 15)
            self.assertEqual(meta["hand_tracking"]["right_frames"], 0)
            self.assertIn("fps", meta)

    def test_recorder_cancellation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            recorder = GestureVideoRecorder(record_dir=tmp_path, fps=30.0, enabled=True)

            class DummyObs:
                results = None
                left_hand_present = True
                right_hand_present = True

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            obs = DummyObs()

            recorder.feed_frame(frame, obs, in_segment=True, now=1.0)
            self.assertTrue(recorder.recording_active)

            recorder.cancel_gesture()
            self.assertFalse(recorder.recording_active)
            self.assertEqual(len(recorder.current_gesture_frames), 0)

    def test_rejected_gesture_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            recorder = GestureVideoRecorder(record_dir=tmp_path, fps=30.0, enabled=True)

            class DummyObs:
                results = None
                left_hand_present = True
                right_hand_present = True

            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            obs = DummyObs()

            for i in range(10):
                recorder.feed_frame(frame, obs, in_segment=True, now=1.0 + i * 0.05)

            class RejectedDecision:
                label = "Tạm biệt"
                confidence = 0.52
                accepted = False
                reason = "confidence 0.52 < threshold 0.72"
                probabilities = np.array([0.2, 0.52, 0.28], dtype=np.float32)

            labels = ["Xin chào", "Tạm biệt", "Cảm ơn"]
            recorder.save_gesture(RejectedDecision(), labels=labels, duration=0.5)
            recorder.close()

            mp4_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
            json_files = list(tmp_path.glob("*.json"))

            self.assertEqual(len(mp4_files), 2)
            self.assertEqual(len(json_files), 1)

            video_names = [f.name for f in mp4_files]
            self.assertTrue(any("REJECTED" in name and "tam_biet" in name for name in video_names))

            json_file = json_files[0]
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.assertEqual(meta["status"], "REJECTED")
            self.assertEqual(meta["label"], "Tạm biệt")
            self.assertIn("0.52 < threshold", meta["reason"])
            self.assertEqual(len(meta["top_candidates"]), 3)

    def test_tracker_and_recorder_loop_integration(self):
        """Mô phỏng chính xác vòng lặp main() trong realtime.py để kiểm tra việc lưu video."""
        from prototype_3_gestures.realtime import SegmentTracker, FEATURE_DIM

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            recorder = GestureVideoRecorder(record_dir=tmp_path, fps=30.0, enabled=True)
            tracker = SegmentTracker(word_gap=0.3, max_seconds=3.0, min_active_seconds=0.2)

            class DummyObs:
                results = None
                left_hand_present = True
                right_hand_present = False
                hands_present = True
                features = np.zeros(FEATURE_DIM, dtype=np.float32)

            obs = DummyObs()
            frame = np.zeros((240, 320, 3), dtype=np.uint8)

            saved_decisions = []

            def handle_segment(segment):
                if segment is None:
                    return
                if segment.duration < 0.2:
                    recorder.cancel_gesture()
                    return
                class Dec:
                    label = "Cảm ơn"
                    confidence = 0.95
                    accepted = True
                    reason = "accepted"
                    probabilities = np.array([0.95, 0.05], dtype=np.float32)
                recorder.save_gesture(Dec(), labels=["Cảm ơn", "Xin chào"], duration=segment.duration)
                saved_decisions.append(segment)

            # Mô phỏng vòng lặp:
            # 1. 5 frames nghỉ (idle)
            now = 10.0
            for _ in range(5):
                now += 0.04
                obs.hands_present = False
                obs.left_hand_present = False
                done = tracker.feed(False, obs.features, now, False, False, gesture_active=False)
                in_gesture = tracker.in_segment or (done is not None)
                recorder.feed_frame(frame, obs, in_segment=in_gesture, now=now)
                if done is not None:
                    handle_segment(done)

            self.assertFalse(tracker.in_segment)

            # 2. 10 frames làm cử chỉ (active)
            for _ in range(10):
                now += 0.04
                obs.hands_present = True
                obs.left_hand_present = True
                done = tracker.feed(True, obs.features, now, True, False, gesture_active=True)
                in_gesture = tracker.in_segment or (done is not None)
                recorder.feed_frame(frame, obs, in_segment=in_gesture, now=now)
                if done is not None:
                    handle_segment(done)

            self.assertTrue(tracker.in_segment)
            self.assertTrue(recorder.recording_active)

            # 3. 10 frames hạ tay (idle gap > word_gap 0.3s)
            for _ in range(10):
                now += 0.04
                obs.hands_present = False
                obs.left_hand_present = False
                done = tracker.feed(False, obs.features, now, False, False, gesture_active=False)
                in_gesture = tracker.in_segment or (done is not None)
                recorder.feed_frame(frame, obs, in_segment=in_gesture, now=now)
                if done is not None:
                    handle_segment(done)

            self.assertFalse(tracker.in_segment)
            self.assertEqual(len(saved_decisions), 1)

            recorder.close()

            mp4_files = list(tmp_path.glob("*.mp4")) + list(tmp_path.glob("*.avi"))
            json_files = list(tmp_path.glob("*.json"))
            self.assertEqual(len(mp4_files), 2, "Both skeleton and raw videos must be saved after gesture ends!")
            self.assertEqual(len(json_files), 1, "Metadata JSON must be saved after gesture ends!")


