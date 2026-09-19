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
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncGenerator

import cv2
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Query, Request
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

configure_utf8_stdio()


class RealtimeVSLRPipeline:
    """Core realtime engine bridging OpenCV, MediaPipe Holistic, PyTorch BiLSTM, and VieNeu-TTS."""

    def __init__(
        self,
        model_path: str | Path,
        camera_id: str | int = 0,
        confidence_threshold: float = 0.72,
        cooldown: float = 1.5,
        word_gap: float = 0.45,
        sentence_gap: float = 2.2,
        min_seconds: float = 0.35,
        max_seconds: float = 5.0,
        tts_voice: str = "Trúc Ly",
        tts_engine: str = "vieneu",
        record_dir: str | Path | None = None,
        no_record: bool = False,
        allow_uncalibrated: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.running = False
        self.thread: threading.Thread | None = None

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

        self.tracker = SegmentTracker(self.word_gap, self.max_seconds, min_active_seconds=self.min_seconds)

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
        self._init_camera(self.requested_camera)
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print(f"[Pipeline] Hệ thống AI VSLR & Camera đã khởi động thành công!")

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.camera_lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
        self.recorder.close()
        self.tts_mgr.stop()
        print(f"[Pipeline] Đã dừng toàn bộ pipeline.")

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

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[Pipeline] Đang kết nối camera: [{self.current_camera_idx}] {self.current_camera_name}")
            return True

    def switch_camera(self, camera_spec: str | int) -> dict[str, Any]:
        self.requested_camera = camera_spec
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
            self.recorder.save_gesture(decision, labels=self.labels, duration=segment.duration)
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

            print(f"[WORD RECOGNIZED] -> {decision.label} ({conf_pct}%) | Câu: {' '.join(self.sentence)}")

            self.latest_decision = {
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": True,
                "reason": "",
                "sentence": list(self.sentence),
            }

            self.broadcast_event({
                "type": "prediction",
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": True,
                "reason": "",
                "sentence": list(self.sentence),
            })

            # Auto-speak current word if sentence is single or immediate feedback
            self.tts_mgr.speak(decision.label)

        else:
            self.last_accepted_label = None
            print(f"[REJECT] -> {decision.label} ({conf_pct}%) — {decision.reason}")
            self.latest_decision = {
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": False,
                "reason": decision.reason,
            }
            self.broadcast_event({
                "type": "prediction",
                "label": decision.label,
                "confidence": conf_pct,
                "accepted": False,
                "reason": decision.reason,
                "sentence": list(self.sentence),
            })

    def speak_sentence_now(self) -> None:
        if not self.sentence:
            self.tts_mgr.speak("Xin chào")
            return
        text = " ".join(self.sentence)
        print(f"[TTS SENTENCE] -> {text}")
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

    def _worker_loop(self) -> None:
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        fps_calc_time = time.monotonic()
        frame_counter = 0
        last_telemetry_time = 0.0

        with HolisticExtractor() as extractor:
            while self.running:
                with self.camera_lock:
                    if self.cap is None or not self.cap.isOpened():
                        time.sleep(0.1)
                        continue
                    ok, frame = self.cap.read()

                if not ok:
                    time.sleep(0.02)
                    continue

                now = time.monotonic()
                frame_counter += 1
                if now - fps_calc_time >= 1.0:
                    self.fps = frame_counter / (now - fps_calc_time)
                    frame_counter = 0
                    fps_calc_time = now

                # 1. Holistic feature extraction
                obs = extractor.process_frame(frame)

                # 2. Hands presence calculation
                num_hands = int(obs.left_hand_present) + int(obs.right_hand_present)
                self.hands_count = num_hands

                # 3. Gesture tracking state machine
                gesture_active = gesture_activity_from_features(
                    obs.features, obs.left_hand_present, obs.right_hand_present
                )

                if self.rec_mode == "manual":
                    if self.manual_recording:
                        # Ghi nhận cử chỉ thủ công
                        done = self.tracker.feed(
                            True,
                            obs.features,
                            now,
                            obs.left_hand_present,
                            obs.right_hand_present,
                            gesture_active=True,
                        )
                        # Giới hạn tối đa 5s
                        if now - self.manual_rec_start >= self.max_seconds:
                            self.manual_recording = False
                            done = self.tracker.force_boundary(now)
                    else:
                        done = None
                else:
                    # Chế độ tự động continuous
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

                # 5. Draw MediaPipe skeleton landmarks directly on frame
                if self.show_hands and getattr(obs, "results", None):
                    res = obs.results
                    # Pose (loại bỏ hoàn toàn các điểm trên khuôn mặt 0-10)
                    if hasattr(res, "pose_landmarks") and res.pose_landmarks:
                        draw_body_pose_landmarks(
                            frame,
                            res.pose_landmarks,
                            point_color=(60, 180, 75),
                            line_color=(255, 225, 25),
                            thickness=1,
                            circle_radius=1,
                        )
                    # Left Hand (Vivid Magenta / Orange)
                    if hasattr(res, "left_hand_landmarks") and res.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(230, 25, 75), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 130, 48), thickness=2, circle_radius=1),
                        )
                    # Right Hand (Vivid Blue / Cyan)
                    if hasattr(res, "right_hand_landmarks") and res.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 130, 200), thickness=2, circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(70, 240, 240), thickness=2, circle_radius=1),
                        )

                # Draw minimal video badges (REC status & Last save info)
                if self.in_segment:
                    cv2.circle(frame, (24, 24), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "REC GESTURE", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (24, 24), 7, (120, 120, 120), -1)
                    cv2.putText(frame, "STANDBY", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

                if self.last_saved_info and (now - self.last_saved_time) < 3.0:
                    frame = draw_unicode_text(
                        frame, self.last_saved_info, (20, 52), font_size=15, color_bgr=(50, 255, 50), bg_color_bgr=(20, 20, 20)
                    )

                # 6. Encode JPEG for high-speed streaming (Quality 75 for ultra low-latency)
                _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                with self.frame_lock:
                    self.latest_jpeg = jpeg_buf.tobytes()
                    self.frame_id += 1
                    self.new_frame_event.set()

                # 7. Telemetry push (every 250ms)
                if now - last_telemetry_time >= 0.25:
                    last_telemetry_time = now
                    elapsed_rec = round(now - self.manual_rec_start, 1) if self.manual_recording else 0.0
                    self.broadcast_event({
                        "type": "telemetry",
                        "fps": round(self.fps, 1),
                        "camera": f"[{self.current_camera_idx}] {self.current_camera_name}",
                        "hands_count": self.hands_count,
                        "in_segment": self.in_segment,
                        "rec_mode": self.rec_mode,
                        "rec_elapsed": elapsed_rec,
                        "sentence": list(self.sentence),
                    })

                time.sleep(0.005)


def create_app(pipeline: RealtimeVSLRPipeline, web_dir: str | Path) -> FastAPI:
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
                # Chờ ngắn 10ms để nhường CPU, kiểm tra frame tiếp theo
                await asyncio.sleep(0.010)

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
                    "rec_mode": pipeline.rec_mode,
                    "show_hands": pipeline.show_hands,
                    "sentence": list(pipeline.sentence),
                    "tts_voice": pipeline.tts_voice,
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

    # 3. Action API (Space, Clear, Speak, Toggle Hands, Toggle Mode)
    @app.post("/api/action")
    async def trigger_action(action: str = Query(...)):
        cmd = action.lower().strip()
        if cmd == "space":
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
        }

    @app.post("/api/camera/select")
    async def select_camera(camera: str = Query(...)):
        res = pipeline.switch_camera(camera)
        return res

    # 5. VieNeu-TTS Speech Synthesis API
    @app.get("/api/tts")
    async def tts_speak(text: str = Query(...), voice: str = Query("Trúc Ly"), play_server: bool = Query(True)):
        normalized = text.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Văn bản không được để trống")

        # 1. Phát trên loa máy tính nếu yêu cầu (giống batch camera)
        if play_server:
            pipeline.tts_mgr.voice = voice
            pipeline.tts_mgr.speak(normalized)

        # 2. Sinh dữ liệu WAV trả về trình duyệt
        try:
            from vieneu import Vieneu
            engine = pipeline.tts_mgr._get_vieneu()
            if engine is not None:
                audio = engine.infer(normalized, voice=voice)
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
            "in_segment": pipeline.in_segment,
            "rec_mode": pipeline.rec_mode,
            "hands_count": pipeline.hands_count,
            "sentence": pipeline.sentence,
            "tts_voice": pipeline.tts_voice,
            "model": "BiLSTM Holistic v3 (24 Classes)",
        }

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

    print(f"[Static Files] Mount thư mục web: {web_path}")
    app.mount("/", StaticFiles(directory=str(web_path), html=True), name="static")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VSLR Web Realtime Studio Server")
    parser.add_argument("--model", default="artifacts/v3-realtime-test-candidate/gesture_lstm.pt")
    parser.add_argument("--camera", default="0", help="Camera index (0, 1) hoặc tên (ví dụ: 'A16')")
    parser.add_argument("--confidence", type=float, default=0.72)
    parser.add_argument("--cooldown", type=float, default=1.5)
    parser.add_argument("--word-gap", type=float, default=0.45)
    parser.add_argument("--sentence-gap", type=float, default=2.2)
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
