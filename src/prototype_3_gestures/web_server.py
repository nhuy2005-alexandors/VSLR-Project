from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncGenerator

import cv2
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import mediapipe as mp

from .vsl3.console import configure_utf8_stdio
from .vsl3.features import (
    FEATURE_DIM,
    LANDMARK_FEATURE_DIM,
    LEFT_PRESENCE_INDEX,
    MAX_BRIDGE_GAP_SECONDS,
    N_HAND,
    N_POSE,
    RIGHT_PRESENCE_INDEX,
    HolisticExtractor,
    preprocess_sequence,
)
from .vsl3.model import load_checkpoint, predict_sequence
from .vsl3.reject import (
    RejectPolicy,
    load_reject_policy,
    reject_prediction,
    sha256_file,
)
from .recorder import (
    GestureVideoRecorder,
    draw_body_pose_landmarks,
    draw_unicode_text,
    resolve_record_dir,
)
from .realtime import (
    Segment,
    SegmentTracker,
    SegmentDecision,
    decide_segment,
    gesture_activity_from_features,
    list_available_cameras,
    resolve_camera,
)
from .tts import TTSManager, normalize_speech_text

configure_utf8_stdio()


def calculate_segment_motion(
    features_list: list[np.ndarray],
    left_list: list[bool],
    right_list: list[bool],
) -> float:
    """Tính biên độ dịch chuyển thực sự của các bàn tay đang có mặt (loại bỏ trạng thái đứng yên)."""
    if len(features_list) < 3:
        return 0.0
    arr = np.asarray(features_list, dtype=np.float32)[:, :LANDMARK_FEATURE_DIM].reshape(-1, N_POSE + N_HAND * 2, 3)
    l_mask = np.asarray(left_list, dtype=bool)
    r_mask = np.asarray(right_list, dtype=bool)
    max_motion = 0.0
    if np.sum(l_mask) >= 2:
        lh = arr[l_mask, N_POSE : N_POSE + N_HAND, :2]
        max_motion = max(max_motion, float(np.max(np.ptp(lh, axis=0))))
    if np.sum(r_mask) >= 2:
        rh = arr[r_mask, N_POSE + N_HAND :, :2]
        max_motion = max(max_motion, float(np.max(np.ptp(rh, axis=0))))
    return max_motion


class RealtimeVSLRPipeline:
    """Core realtime engine bridging OpenCV, MediaPipe Holistic, PyTorch BiLSTM, and VieNeu-TTS."""

    def __init__(
        self,
        model_path: str | Path,
        camera_id: str | int = 0,
        confidence_threshold: float = 0.70,
        cooldown: float = 1.0,
        word_gap: float = 0.28,
        sentence_gap: float = 1.8,
        min_seconds: float = 0.18,
        max_seconds: float = 5.0,
        tts_voice: str = "Trúc Ly",
        tts_engine: str = "vieneu",
        record_dir: str | Path | None = None,
        no_record: bool = False,
        allow_uncalibrated: bool = True,
    ) -> None:
        is_cloud_env = bool(os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACE_ID"))
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not is_cloud_env) else "cpu")
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown
        self.word_gap = word_gap
        self.sentence_gap = sentence_gap
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.tts_voice = tts_voice
        self.tts_engine = tts_engine
        self.allow_uncalibrated = allow_uncalibrated
        self.record_dir = resolve_record_dir(record_dir)

        # 1. Load Model & Reject Policy
        self.model, self.labels, self.config = load_checkpoint(self.model_path, self.device)
        self.seq_len = int(self.config.get("sequence_length", 30))
        checkpoint_sha = sha256_file(self.model_path)
        self.policy = load_reject_policy(
            None,
            expected_feature_contract=self.config.get("feature_contract"),
            expected_training_signature=self.config.get("training_signature"),
            expected_checkpoint_sha256=checkpoint_sha,
            expected_rejection_contract=self.config.get("rejection_contract"),
        )

        # 2. TTS & Recorder
        self.tts_mgr = TTSManager(engine=self.tts_engine, voice=self.tts_voice, enabled=True)
        self.recorder = GestureVideoRecorder(
            record_dir=self.record_dir,
            fps=30.0,
            enabled=not no_record,
            record_mode="both",
        )

        # 3. Camera & State
        self.requested_camera = camera_id
        self.current_camera_idx = 0
        self.current_camera_name = "Camera 0"
        self.cap: cv2.VideoCapture | None = None
        self.camera_lock = threading.Lock()
        self.camera_enabled = False
        self.current_signer: str = "Khách"
        self.last_external_frame_time: float = 0.0

        self.running = False
        self.camera_thread: threading.Thread | None = None
        self.ai_thread: threading.Thread | None = None

        # Decoupled high-speed pipeline sync primitives
        self.ai_input_frame: np.ndarray | None = None
        self.ai_frame_lock = threading.Lock()
        self.new_ai_frame_event = threading.Event()
        self.latest_results: Any = None

        self.show_hands = True
        self.rec_mode = "auto"  # 'auto' or 'manual'
        self.manual_recording = False
        self.manual_rec_start = 0.0

        self.sentence: list[str] = []
        self.last_accepted_label: str | None = None
        self.last_accepted_time: float = 0.0
        self.last_sentence_activity = time.monotonic()
        self.last_saved_info: str | None = None
        self.last_saved_time: float = 0.0

        # Realtime metrics
        self.fps = 0.0
        self.in_segment = False
        self.hands_count = 0
        self.latest_decision: dict[str, Any] = {}

        # Frame cache for MJPEG streaming
        self.latest_jpeg: bytes | None = None
        self.frame_id: int = 0
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

        # SSE Event queues
        self.event_subscribers: list[asyncio.Queue] = []
        self.event_loop: asyncio.AbstractEventLoop | None = None

        # Bộ lọc chuyển động chống nhận diện nhầm khi người dùng đứng yên
        self.recent_hand_window: deque = deque(maxlen=6)
        self.min_gesture_motion: float = 0.50

        # Kích hoạt nhạy sau 0.10s chuyển động (lọc bỏ nhiễu đứng yên 1-2 frame)
        self.tracker = SegmentTracker(self.word_gap, self.max_seconds, min_active_seconds=0.10)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = loop

    def broadcast_event(self, event: dict[str, Any]) -> None:
        """Push real-time event to all connected SSE clients."""
        if not self.event_subscribers or not self.event_loop:
            return
        payload = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        for q in list(self.event_subscribers):
            try:
                self.event_loop.call_soon_threadsafe(q.put_nowait, payload)
            except Exception:
                pass

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.camera_enabled = False
        self.camera_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.camera_thread.start()
        self.ai_thread.start()
        print(f"[Pipeline] Hệ thống AI VSLR đã sẵn sàng (Kiến trúc Đa luồng Decoupled 30 FPS mượt mà).")

    def stop(self) -> None:
        self.running = False
        self.camera_enabled = False
        self.new_ai_frame_event.set()
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        with self.camera_lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
        self.recorder.close()
        self.tts_mgr.stop()
        print(f"[Pipeline] Đã dừng toàn bộ pipeline.")

    def start_camera(self) -> bool:
        if self.camera_enabled and self.cap is not None and self.cap.isOpened():
            return True
        ok = self._init_camera(self.requested_camera)
        if ok:
            self.camera_enabled = True
            self.tracker.reset()
            self.in_segment = False
            self.hands_count = 0
            self.broadcast_event({
                "type": "camera_state",
                "enabled": True,
                "index": self.current_camera_idx,
                "name": self.current_camera_name,
            })
            print(f"[Pipeline] Đã BẬT Webcam & Nhận diện: [{self.current_camera_idx}] {self.current_camera_name}")
            return True
        else:
            # Fallback chế độ Cloud / Trình duyệt từ xa: Cho phép nhận luồng từ Webcam Client (Browser)
            self.camera_enabled = True
            self.current_camera_idx = -1
            self.current_camera_name = "Webcam Trình duyệt (Client Stream)"
            self.tracker.reset()
            self.in_segment = False
            self.hands_count = 0
            self.broadcast_event({
                "type": "camera_state",
                "enabled": True,
                "index": -1,
                "name": self.current_camera_name,
                "client_mode": True,
            })
            print(f"[Pipeline] Chuyển sang chế độ Camera Trình duyệt (Client Stream). Sẵn sàng nhận frame từ Web.")
            return True

    def inject_external_frame(self, frame: np.ndarray, signer: str | None = None) -> None:
        """Đẩy frame từ Web Client (trình duyệt của bạn bè) vào luồng AI nhận diện thời gian thực."""
        if signer and signer.strip():
            self.current_signer = signer.strip()
        self.camera_enabled = True
        self.last_external_frame_time = time.monotonic()
        with self.ai_frame_lock:
            self.ai_input_frame = frame
        self.new_ai_frame_event.set()

    def stop_camera(self) -> None:
        self.camera_enabled = False
        self.latest_results = None
        with self.camera_lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
        self.in_segment = False
        self.hands_count = 0
        self.fps = 0.0
        self.manual_recording = False
        self.tracker.reset()
        self.recorder.cancel_gesture()
        self.broadcast_event({
            "type": "camera_state",
            "enabled": False,
        })
        print(f"[Pipeline] Đã TẮT Webcam & Tạm dừng nhận diện.")

    def _init_camera(self, camera_spec: str | int) -> bool:
        with self.camera_lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            try:
                idx, name = resolve_camera(camera_spec)
                self.current_camera_idx = idx
                self.current_camera_name = name
            except Exception as exc:
                print(f"[Pipeline Warning] Không thể mở camera '{camera_spec}': {exc}. Dùng camera 0...")
                self.current_camera_idx = 0
                self.current_camera_name = "Camera 0"

            self.cap = cv2.VideoCapture(self.current_camera_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.current_camera_idx)
            if not self.cap.isOpened():
                print(f"[Pipeline Error] Không thể mở camera [{self.current_camera_idx}] {self.current_camera_name}!", file=sys.stderr)
                return False

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[Pipeline] Đang kết nối camera: [{self.current_camera_idx}] {self.current_camera_name}")
            return True

    def switch_camera(self, camera_spec: str | int) -> dict[str, Any]:
        self.requested_camera = camera_spec
        ok = False
        if self.camera_enabled:
            ok = self._init_camera(camera_spec)
        self.tracker.reset()
        self.broadcast_event({
            "type": "camera_switched",
            "index": self.current_camera_idx,
            "name": self.current_camera_name,
            "success": ok,
        })
        return {"index": self.current_camera_idx, "name": self.current_camera_name, "success": ok}

    def _handle_segment(self, segment: Segment | None) -> None:
        if segment is None:
            return
        if segment.duration < self.min_seconds:
            self.recorder.cancel_gesture()
            return

        # Kiểm tra biên độ chuyển động thực sự của bàn tay (chặn đứng yên bị dịch nhầm 'Bạn quê ở đâu')
        if not self.rec_mode == "manual":
            seg_motion = calculate_segment_motion(
                segment.features,
                segment.left_hand_present,
                segment.right_hand_present,
            )
            if seg_motion < self.min_gesture_motion:
                self.recorder.cancel_gesture()
                return

        decision = decide_segment(
            self.model,
            self.labels,
            self.device,
            segment,
            self.seq_len,
            confidence_threshold=self.confidence_threshold,
            reject_policy=self.policy,
            allow_uncalibrated=self.allow_uncalibrated,
        )

        if self.recorder.enabled:
            self.recorder.save_gesture(
                decision,
                labels=self.labels,
                duration=segment.duration,
                signer_name=self.current_signer,
            )
            status_desc = "ACCEPTED" if decision.accepted else "REJECTED"
            self.last_saved_info = f"Đã lưu video: {decision.label} [{status_desc}]"
            self.last_saved_time = time.monotonic()

        now_seg = time.monotonic()
        conf_pct = round(decision.confidence * 100)

        if decision.accepted:
            if (
                self.cooldown > 0
                and decision.label == self.last_accepted_label
                and (now_seg - self.last_accepted_time) < self.cooldown
            ):
                print(f"[COOLDOWN] Bỏ qua lặp từ '{decision.label}' trong {self.cooldown}s")
                return

            self.last_accepted_label = decision.label
            self.last_accepted_time = now_seg
            self.sentence.append(decision.label)
            self.last_sentence_activity = now_seg

            print(f"[WORD RECOGNIZED] ({self.current_signer}) -> {decision.label} ({conf_pct}%) | Câu: {' '.join(self.sentence)}")

            self.latest_decision = {
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": True,
                "reason": "",
                "sentence": list(self.sentence),
                "signer": self.current_signer,
            }

            self.broadcast_event({
                "type": "prediction",
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": True,
                "reason": "",
                "sentence": list(self.sentence),
                "signer": self.current_signer,
            })

            # Auto-speak current word nếu không phải môi trường Cloud / Headless
            if not os.environ.get("SPACE_ID"):
                self.tts_mgr.speak(decision.label)

        else:
            self.last_accepted_label = None
            print(f"[REJECT] ({self.current_signer}) -> {decision.label} ({conf_pct}%) — {decision.reason}")
            self.latest_decision = {
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": False,
                "reason": decision.reason,
                "signer": self.current_signer,
            }
            self.broadcast_event({
                "type": "prediction",
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": False,
                "reason": decision.reason,
                "sentence": list(self.sentence),
                "signer": self.current_signer,
            })

    def speak_sentence_now(self) -> None:
        if not self.sentence:
            if not os.environ.get("SPACE_ID"):
                self.tts_mgr.speak("Xin chào")
            self.broadcast_event({
                "type": "sentence_spoken",
                "text": "Xin chào",
            })
            return
        text = " ".join(self.sentence)
        print(f"[TTS SENTENCE] -> {text}")
        if not os.environ.get("SPACE_ID"):
            self.tts_mgr.speak(text)
        self.broadcast_event({
            "type": "sentence_spoken",
            "text": text,
        })
        self.sentence.clear()
        self.last_sentence_activity = time.monotonic()
        self.broadcast_event({
            "type": "sentence_updated",
            "sentence": [],
        })

    def clear_sentence_now(self) -> None:
        self.tracker.reset()
        self.recorder.cancel_gesture()
        self.sentence.clear()
        self.last_accepted_label = None
        self.last_accepted_time = 0.0
        self.manual_recording = False
        self.broadcast_event({
            "type": "sentence_updated",
            "sentence": [],
        })
        print(f"[Sentence] Đã xóa toàn bộ câu.")

    def trigger_space(self) -> None:
        now = time.monotonic()
        if self.rec_mode == "manual":
            if not self.manual_recording:
                # Bắt đầu ghi thủ công
                self.manual_recording = True
                self.manual_rec_start = now
                self.tracker.reset()
                self.broadcast_event({
                    "type": "rec_state",
                    "recording": True,
                    "mode": "manual",
                })
            else:
                # Kết thúc ghi thủ công & phân loại
                self.manual_recording = False
                seg = self.tracker.force_boundary(now)
                self._handle_segment(seg)
                self.broadcast_event({
                    "type": "rec_state",
                    "recording": False,
                    "mode": "manual",
                })
        else:
            # Chế độ tự động: chốt ngay boundary
            seg = self.tracker.force_boundary(now)
            if seg:
                self._handle_segment(seg)

    def _draw_status_badges(self, display_frame: np.ndarray, now: float) -> np.ndarray:
        if self.in_segment:
            cv2.circle(display_frame, (24, 24), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC GESTURE", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.circle(display_frame, (24, 24), 7, (120, 120, 120), -1)
            cv2.putText(display_frame, "STANDBY", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        if self.last_saved_info and (now - self.last_saved_time) < 3.0:
            display_frame = draw_unicode_text(
                display_frame, self.last_saved_info, (20, 52), font_size=15, color_bgr=(50, 255, 50), bg_color_bgr=(20, 20, 20)
            )
        return display_frame

    def _publish_jpeg(self, display_frame: np.ndarray) -> None:
        _, jpeg_buf = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        with self.frame_lock:
            self.latest_jpeg = jpeg_buf.tobytes()
            self.frame_id += 1
            self.new_frame_event.set()

    def _camera_capture_loop(self) -> None:
        """Luồng 1: Liên tục rút (drain) frame mới nhất từ phần cứng camera để triệt tiêu độ trễ buffer."""
        while self.running:
            if not self.camera_enabled:
                time.sleep(0.04)
                continue

            # Nếu đang có luồng frame từ trình duyệt (client feed) trong 3s qua, nhường luồng AI cho client
            if (time.monotonic() - getattr(self, "last_external_frame_time", 0.0)) < 3.0:
                time.sleep(0.02)
                continue

            with self.camera_lock:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.04)
                    continue
                ok, frame = self.cap.read()

            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Đẩy frame mới nhất sang Luồng AI (luôn ghi đè frame mới nhất, không bao giờ ứ đọng)
            with self.ai_frame_lock:
                self.ai_input_frame = frame
            self.new_ai_frame_event.set()

            # Khi TẮT khung xương: Xuất trực tiếp luồng hình thô 30 FPS siêu mượt
            if not self.show_hands:
                now = time.monotonic()
                raw_out = self._draw_status_badges(frame.copy(), now)
                self._publish_jpeg(raw_out)

            time.sleep(0.004)

    def _ai_worker_loop(self) -> None:
        """Luồng 2: Xử lý MediaPipe Holistic & Vẽ đồng bộ 100% khung xương lên chính frame đó (0 pixel drift)."""
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        last_telemetry_time = 0.0
        fps_calc_time = time.monotonic()
        frame_counter = 0

        with HolisticExtractor() as extractor:
            while self.running:
                if not self.camera_enabled:
                    time.sleep(0.05)
                    continue

                got = self.new_ai_frame_event.wait(timeout=0.1)
                if not got:
                    continue
                self.new_ai_frame_event.clear()

                with self.ai_frame_lock:
                    if self.ai_input_frame is None:
                        continue
                    frame = self.ai_input_frame.copy()

                now = time.monotonic()
                frame_counter += 1
                if now - fps_calc_time >= 1.0:
                    self.fps = frame_counter / (now - fps_calc_time)
                    frame_counter = 0
                    fps_calc_time = now

                # 1. Holistic feature extraction (kèm ổn định ngón tay chống rớt điểm)
                obs = extractor.process_frame(frame)
                res = getattr(obs, "results", None)
                self.latest_results = res

                # 2. Hands presence calculation
                num_hands = int(obs.left_hand_present) + int(obs.right_hand_present)
                self.hands_count = num_hands

                # 3. Gesture tracking state machine (Kết hợp vị trí cổ tay + vận tốc chuyển động thực sự)
                if obs.hands_present:
                    self.recent_hand_window.append((obs.features, obs.left_hand_present, obs.right_hand_present))
                else:
                    self.recent_hand_window.clear()

                recent_motion = 0.0
                if len(self.recent_hand_window) >= 3:
                    recent_motion = calculate_segment_motion(
                        [x[0] for x in self.recent_hand_window],
                        [x[1] for x in self.recent_hand_window],
                        [x[2] for x in self.recent_hand_window],
                    )

                wrist_gate = gesture_activity_from_features(
                    obs.features, obs.left_hand_present, obs.right_hand_present, wrist_above_hip=0.30
                )
                # Khi đứng yên (recent_motion < 0.22): giữ nguyên STANDBY, không kích hoạt ghi nhận
                is_moving = recent_motion >= 0.22
                gesture_active = bool(
                    obs.hands_present
                    and wrist_gate
                    and (is_moving or (self.in_segment and recent_motion >= 0.14))
                )

                if self.rec_mode == "manual":
                    if self.manual_recording:
                        done = self.tracker.feed(
                            True,
                            obs.features,
                            now,
                            obs.left_hand_present,
                            obs.right_hand_present,
                            gesture_active=True,
                        )
                        if now - self.manual_rec_start >= self.max_seconds:
                            self.manual_recording = False
                            done = self.tracker.force_boundary(now)
                    else:
                        done = None
                else:
                    done = self.tracker.feed(
                        obs.hands_present,
                        obs.features,
                        now,
                        obs.left_hand_present,
                        obs.right_hand_present,
                        gesture_active=gesture_active,
                    )

                self.in_segment = self.tracker.in_segment or self.manual_recording
                in_gesture = self.in_segment or (done is not None)
                self.recorder.feed_frame(frame, obs, in_segment=in_gesture, now=now)

                # Dự đoán sớm thời gian thực (Live Preview) chỉ khi thực sự có chuyển động mạnh (>= 0.50)
                if self.in_segment and done is None and len(self.tracker.segment) >= 10 and (len(self.tracker.segment) % 4 == 0):
                    try:
                        live_motion = calculate_segment_motion(
                            self.tracker.segment,
                            self.tracker.segment_left,
                            self.tracker.segment_right,
                        )
                        if live_motion >= self.min_gesture_motion:
                            seg_arr = np.asarray(self.tracker.segment, dtype=np.float32)
                            prep_live = preprocess_sequence(
                                seg_arr,
                                left_hand_present=self.tracker.segment_left,
                                right_hand_present=self.tracker.segment_right,
                                timestamps=self.tracker.segment_times if len(self.tracker.segment_times) == len(seg_arr) else None,
                                target_len=self.seq_len,
                            )
                            live_lbl, live_conf, _ = predict_sequence(self.model, prep_live, self.labels, self.device)
                            if live_conf >= 0.65:
                                self.broadcast_event({
                                    "type": "prediction_preview",
                                    "label": live_lbl,
                                    "confidence": round(live_conf * 100),
                                })
                    except Exception:
                        pass

                if done is not None:
                    self._handle_segment(done)

                # 4. Auto sentence speak when gap expires (chỉ trong chế độ auto)
                if (
                    self.rec_mode == "auto"
                    and self.sentence
                    and not self.tracker.in_segment
                    and (now - self.last_sentence_activity) >= self.sentence_gap
                ):
                    self.speak_sentence_now()

                # 5. Khi BẬT khung xương: Vẽ trực tiếp lên CHÍNH frame vừa phân tích để khớp 100% vị trí bàn tay
                if self.show_hands:
                    if res is not None:
                        if hasattr(res, "pose_landmarks") and res.pose_landmarks:
                            draw_body_pose_landmarks(
                                frame,
                                res.pose_landmarks,
                                point_color=(60, 180, 75),
                                line_color=(255, 225, 25),
                                thickness=1,
                                circle_radius=1,
                            )
                        if hasattr(res, "left_hand_landmarks") and res.left_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                res.left_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(230, 25, 75), thickness=2, circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 130, 48), thickness=2, circle_radius=1),
                            )
                        if hasattr(res, "right_hand_landmarks") and res.right_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                res.right_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 130, 200), thickness=2, circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(70, 240, 240), thickness=2, circle_radius=1),
                            )
                    frame = self._draw_status_badges(frame, now)
                    self._publish_jpeg(frame)

                # 6. Telemetry push (every 250ms)
                if now - last_telemetry_time >= 0.25:
                    last_telemetry_time = now
                    elapsed_rec = round(now - self.manual_rec_start, 1) if self.manual_recording else 0.0
                    self.broadcast_event({
                        "type": "telemetry",
                        "fps": round(self.fps, 1),
                        "camera": f"[{self.current_camera_idx}] {self.current_camera_name}",
                        "camera_enabled": self.camera_enabled,
                        "hands_count": self.hands_count,
                        "in_segment": self.in_segment,
                        "rec_mode": self.rec_mode,
                        "rec_elapsed": elapsed_rec,
                        "sentence": list(self.sentence),
                    })


def create_app(
    pipeline: RealtimeVSLRPipeline,
    web_dir: str | Path,
    static_mount_path: str = "/",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipeline.set_event_loop(asyncio.get_event_loop())
        pipeline.start()
        try:
            yield
        finally:
            pipeline.stop()

    app = FastAPI(
        title="VSLR Web Realtime Studio",
        description="Hệ thống nhận diện cử chỉ thời gian thực VSLR",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Chống trình duyệt cache JS/CSS cũ
    @app.middleware("http")
    async def add_no_cache_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    # 1. Video MJPEG Feed (30+ FPS, Zero-Lag Fresh-Frames Only)
    @app.get("/api/video_feed")
    async def video_feed():
        async def frame_generator():
            last_sent_id = -1
            while pipeline.running:
                if not pipeline.camera_enabled:
                    await asyncio.sleep(0.08)
                    continue
                frame_bytes = None
                current_id = -1
                with pipeline.frame_lock:
                    if pipeline.frame_id != last_sent_id:
                        frame_bytes = pipeline.latest_jpeg
                        current_id = pipeline.frame_id

                if frame_bytes is not None and current_id != last_sent_id:
                    last_sent_id = current_id
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                # Nhịp độ kiểm tra frame tối ưu ~35-40 FPS, nhường CPU cho event loop
                await asyncio.sleep(0.025)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # 2. Server-Sent Events (SSE) for Realtime Predictions & State
    @app.get("/api/events")
    async def event_stream(request: Request):
        q: asyncio.Queue = asyncio.Queue()
        pipeline.event_subscribers.append(q)

        async def stream():
            try:
                # Send initial state
                init_data = json.dumps({
                    "type": "init",
                    "camera": f"[{pipeline.current_camera_idx}] {pipeline.current_camera_name}",
                    "camera_enabled": pipeline.camera_enabled,
                    "rec_mode": pipeline.rec_mode,
                    "show_hands": pipeline.show_hands,
                    "sentence": list(pipeline.sentence),
                    "tts_voice": pipeline.tts_voice,
                    "signer": pipeline.current_signer,
                }, ensure_ascii=False)
                yield f"data: {init_data}\n\n"

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield msg
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                if q in pipeline.event_subscribers:
                    pipeline.event_subscribers.remove(q)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 3. Action API (Space, Clear, Speak, Toggle Hands, Toggle Mode, Start/Stop Camera)
    @app.post("/api/action")
    async def trigger_action(action: str = Query(...)):
        cmd = action.lower().strip()
        if cmd == "start_camera":
            ok = pipeline.start_camera()
            return {"status": "ok", "action": "start_camera", "camera_enabled": ok}
        elif cmd == "stop_camera":
            pipeline.stop_camera()
            return {"status": "ok", "action": "stop_camera", "camera_enabled": False}
        elif cmd == "toggle_camera":
            if pipeline.camera_enabled:
                pipeline.stop_camera()
            else:
                pipeline.start_camera()
            return {"status": "ok", "action": "toggle_camera", "camera_enabled": pipeline.camera_enabled}
        elif cmd == "space":
            if not pipeline.camera_enabled:
                pipeline.start_camera()
                return {"status": "ok", "action": "start_camera", "camera_enabled": pipeline.camera_enabled}
            pipeline.trigger_space()
            return {"status": "ok", "action": "space"}
        elif cmd == "clear":
            pipeline.clear_sentence_now()
            return {"status": "ok", "action": "clear"}
        elif cmd == "speak":
            pipeline.speak_sentence_now()
            return {"status": "ok", "action": "speak"}
        elif cmd == "toggle_hands":
            pipeline.show_hands = not pipeline.show_hands
            pipeline.broadcast_event({
                "type": "settings",
                "show_hands": pipeline.show_hands,
            })
            return {"status": "ok", "show_hands": pipeline.show_hands}
        elif cmd == "toggle_mode":
            pipeline.rec_mode = "manual" if pipeline.rec_mode == "auto" else "auto"
            pipeline.tracker.reset()
            pipeline.manual_recording = False
            pipeline.broadcast_event({
                "type": "settings",
                "rec_mode": pipeline.rec_mode,
            })
            return {"status": "ok", "rec_mode": pipeline.rec_mode}
        else:
            raise HTTPException(status_code=400, detail=f"Lệnh không hợp lệ: {action}")

    # 4. Cameras List & Selection
    @app.get("/api/cameras")
    async def get_cameras():
        cams = list_available_cameras()
        return {
            "cameras": cams,
            "current_index": pipeline.current_camera_idx,
            "current_name": pipeline.current_camera_name,
            "camera_enabled": pipeline.camera_enabled,
        }

    @app.post("/api/camera/start")
    async def start_camera_api():
        ok = pipeline.start_camera()
        return {"success": ok, "camera_enabled": ok, "name": pipeline.current_camera_name}

    @app.post("/api/camera/stop")
    async def stop_camera_api():
        pipeline.stop_camera()
        return {"success": True, "camera_enabled": False}

    @app.post("/api/camera/select")
    async def select_camera(camera: str = Query(...)):
        res = pipeline.switch_camera(camera)
        return res

    # 5. VieNeu-TTS Speech Synthesis API
    @app.get("/api/tts")
    async def tts_speak(text: str = Query(...), voice: str = Query("Trúc Ly"), play_server: bool = Query(True)):
        normalized = normalize_speech_text(text.strip())
        if not normalized:
            raise HTTPException(status_code=400, detail="Văn bản không được để trống")

        # 1. Phát trên loa máy tính nếu yêu cầu: phát xong phản hồi JSON, KHÔNG gửi thêm file âm thanh để tránh echo
        if play_server:
            pipeline.tts_mgr.voice = voice
            pipeline.tts_mgr.speak(normalized)
            return JSONResponse({"status": "played_on_server", "text": normalized, "voice": voice})

        # 2. Sinh dữ liệu WAV trả về trình duyệt (khi client muốn tự phát qua loa trình duyệt / thiết bị di động)
        try:
            from vieneu import Vieneu
            engine = pipeline.tts_mgr._get_vieneu()
            if engine is not None:
                audio = engine.infer(
                    normalized,
                    voice=voice,
                    temperature=pipeline.tts_mgr.temperature,
                    top_p=pipeline.tts_mgr.top_p,
                    top_k=pipeline.tts_mgr.top_k,
                    repetition_penalty=pipeline.tts_mgr.repetition_penalty,
                )
                sr = getattr(engine, "sample_rate", 48000)
                audio_arr = np.asarray(audio, dtype=np.float32)

                # Fade-out & padding
                fade_len = min(len(audio_arr), int(sr * 0.02))
                if fade_len > 0:
                    audio_arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

                buf = io.BytesIO()
                sf.write(buf, audio_arr, sr, format="WAV")
                buf.seek(0)
                return Response(content=buf.read(), media_type="audio/wav")
        except Exception as exc:
            print(f"[TTS API Error] {exc}", file=sys.stderr)

        return JSONResponse({"status": "played_on_server", "text": normalized})

    # 6. Status API
    @app.get("/api/status")
    async def get_status():
        return {
            "fps": round(pipeline.fps, 1),
            "camera": pipeline.current_camera_name,
            "camera_index": pipeline.current_camera_idx,
            "camera_enabled": pipeline.camera_enabled,
            "in_segment": pipeline.in_segment,
            "rec_mode": pipeline.rec_mode,
            "hands_count": pipeline.hands_count,
            "sentence": pipeline.sentence,
            "tts_voice": pipeline.tts_voice,
            "signer": pipeline.current_signer,
            "dataset_repo": os.environ.get("HF_DATASET_REPO", "ntbii305/vslr-remote"),
            "model": "BiLSTM Holistic v3 (24 Classes)",
        }

    # 7. Signer Name Management API
    @app.get("/api/signer")
    async def get_signer():
        return {"signer": pipeline.current_signer}

    @app.post("/api/signer")
    async def set_signer(name: str = Query("Khách")):
        cleaned = name.strip() or "Khách"
        pipeline.current_signer = cleaned
        pipeline.broadcast_event({
            "type": "signer_updated",
            "signer": pipeline.current_signer,
        })
        return {"status": "ok", "signer": pipeline.current_signer}

    # 8. Client Frame Injection API (Webcam từ trình duyệt từ xa gửi về AI)
    @app.post("/api/client_frame")
    async def post_client_frame(request: Request, signer: str = Query(None)):
        content_type = request.headers.get("content-type", "")
        img_bytes = None
        if "application/json" in content_type:
            try:
                body = await request.json()
                img_b64 = body.get("image", "")
                if signer is None:
                    signer = body.get("signer", None)
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                import base64
                img_bytes = base64.b64decode(img_b64)
            except Exception:
                img_bytes = None
        else:
            img_bytes = await request.body()

        if img_bytes:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                pipeline.inject_external_frame(frame, signer=signer)
                return {
                    "status": "ok",
                    "hands_count": pipeline.hands_count,
                    "in_segment": pipeline.in_segment,
                    "fps": round(pipeline.fps, 1),
                }
        return JSONResponse({"status": "error", "detail": "Invalid frame"}, status_code=400)

    # 9. WebSocket Client Stream (Tốc độ cao cho Client Webcam)
    @app.websocket("/api/ws/client_feed")
    async def websocket_client_feed(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_bytes()
                if data:
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        pipeline.inject_external_frame(frame)
                        await websocket.send_json({
                            "type": "telemetry",
                            "hands": pipeline.hands_count,
                            "in_segment": pipeline.in_segment,
                            "fps": round(pipeline.fps, 1),
                        })
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Mount static files (Frontend HTML, CSS, JS)
    web_path = Path(web_dir).resolve()
    if not web_path.is_dir():
        # Fallback search
        candidate = Path.cwd() / "vslr-web"
        if candidate.is_dir():
            web_path = candidate
        else:
            candidate2 = Path.cwd() / "VSLR-Project-pipeline-signer-split" / "vslr-web"
            if candidate2.is_dir():
                web_path = candidate2

    if static_mount_path:
        print(f"[Static Files] Mount thư mục web tại '{static_mount_path}': {web_path}")
        app.mount(static_mount_path, StaticFiles(directory=str(web_path), html=True), name="static")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VSLR Web Realtime Studio Server")
    parser.add_argument("--model", default="artifacts/v3-realtime-test-candidate/gesture_lstm.pt")
    parser.add_argument("--camera", default="0", help="Camera index (0, 1) hoặc tên (ví dụ: 'A16')")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--word-gap", type=float, default=0.28)
    parser.add_argument("--sentence-gap", type=float, default=1.8)
    parser.add_argument("--tts-voice", default="Trúc Ly")
    parser.add_argument("--tts-engine", default="vieneu")
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--allow-uncalibrated", action="store_true", default=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--web-dir", default="vslr-web")

    args = parser.parse_args()

    # Resolve model path
    model_p = Path(args.model)
    if not model_p.is_file():
        candidates = [
            Path(__file__).resolve().parent.parent.parent / args.model,
            Path.cwd() / args.model,
            Path.cwd() / "VSLR-Project-pipeline-signer-split" / args.model,
            Path(__file__).resolve().parent.parent.parent / "models" / "gesture_lstm.pt",
        ]
        for c in candidates:
            if c.is_file():
                model_p = c
                break

    if not model_p.is_file():
        raise SystemExit(f"Lỗi: Không tìm thấy model tại {args.model}")

    # Resolve web directory
    web_p = Path(args.web_dir)
    if not web_p.is_dir():
        candidates = [
            Path(__file__).resolve().parent.parent.parent.parent / "vslr-web",
            Path(__file__).resolve().parent.parent.parent / "vslr-web",
            Path.cwd() / "vslr-web",
            Path.cwd() / "VSLR-Project-pipeline-signer-split" / "vslr-web",
        ]
        for c in candidates:
            if c.is_dir():
                web_p = c
                break

    print("=========================================================================")
    print("   VSLR REALTIME WEB STUDIO - HE THONG NHAN DIEN NCKH DAI HOC CAN THO    ")
    print("=========================================================================")
    print(f"  * Model       : {model_p}")
    print(f"  * Camera      : {args.camera}")
    print(f"  * TTS Engine  : {args.tts_engine} (Giọng: {args.tts_voice})")
    print(f"  * Web Port    : http://{args.host}:{args.port}")
    print("=========================================================================")

    pipeline = RealtimeVSLRPipeline(
        model_path=model_p,
        camera_id=args.camera,
        confidence_threshold=args.confidence,
        cooldown=args.cooldown,
        word_gap=args.word_gap,
        sentence_gap=args.sentence_gap,
        tts_voice=args.tts_voice,
        tts_engine=args.tts_engine,
        record_dir=args.record_dir,
        no_record=args.no_record,
        allow_uncalibrated=args.allow_uncalibrated,
    )

    app = create_app(pipeline, web_dir=web_p)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
