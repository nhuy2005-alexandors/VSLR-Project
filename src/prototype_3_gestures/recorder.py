from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def slugify(text: str) -> str:
    """Chuyển đổi chuỗi tiếng Việt có dấu thành slug không dấu an toàn cho tên file."""
    text = str(text or "").strip()
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", text) or "unknown"


def resolve_record_dir(record_dir: str | Path | None = None) -> Path:
    """Xác định thư mục lưu trữ video cử chỉ động, tương thích khi copy sang bất kỳ máy nào."""
    if record_dir is not None:
        p = Path(record_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Tìm kiếm thư mục videos tương đối theo cấu trúc dự án
    src_dir = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / "videos",
        src_dir.parent / "videos",
        src_dir.parent.parent / "videos",
    ]
    for c in candidates:
        if c.is_dir() or c.parent.is_dir():
            c.mkdir(parents=True, exist_ok=True)
            return c

    target = Path.cwd() / "videos"
    target.mkdir(parents=True, exist_ok=True)
    return target


class FontManager:
    """Quản lý font chữ Unicode tiếng Việt hỗ trợ vẽ bằng Pillow."""

    _cached_fonts: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    @classmethod
    def get_font(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if size in cls._cached_fonts:
            return cls._cached_fonts[size]

        bundled_font = str(Path(__file__).resolve().parent / "arial.ttf")
        font_candidates = [
            bundled_font,
            "arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "segoeui.ttf",
            "calibri.ttf",
            "tahoma.ttf",
            "Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
        loaded_font = None
        for candidate in font_candidates:
            try:
                loaded_font = ImageFont.truetype(candidate, size)
                break
            except Exception:
                continue

        if loaded_font is None:
            try:
                loaded_font = ImageFont.load_default()
            except Exception:
                pass

        cls._cached_fonts[size] = loaded_font
        return loaded_font


def draw_unicode_text(
    img_bgr: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_size: int = 20,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
    bg_color_bgr: tuple[int, int, int] | None = None,
    padding: int = 4,
) -> np.ndarray:
    """Vẽ chữ tiếng Việt có dấu lên ảnh OpenCV BGR bằng Pillow với hiệu năng cao (patch blit)."""
    if not text:
        return img_bgr

    font = FontManager.get_font(font_size)
    h, w = img_bgr.shape[:2]
    x, y = pos

    if x >= w or y >= h:
        return img_bgr

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    try:
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])
    except Exception:
        tw, th = max(1, len(text) * font_size // 2), font_size

    patch_w = tw + padding * 2
    patch_h = th + padding * 2

    # Giới hạn kích thước không tràn ra ngoài ảnh gốc
    if x + patch_w > w:
        patch_w = max(1, w - x)
    if y + patch_h > h:
        patch_h = max(1, h - y)

    if bg_color_bgr is not None:
        bg_color_rgb = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0])
        patch = Image.new("RGB", (patch_w, patch_h), bg_color_rgb)
    else:
        crop_bgr = img_bgr[y : y + patch_h, x : x + patch_w]
        if crop_bgr.size == 0:
            return img_bgr
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        patch = Image.fromarray(crop_rgb)

    draw = ImageDraw.Draw(patch)
    text_color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((padding, padding), text, font=font, fill=text_color_rgb)

    patch_bgr = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
    img_bgr[y : y + patch_h, x : x + patch_w] = patch_bgr
    return img_bgr


# Các đường nối khung xương phần thân (bỏ qua hoàn toàn 9 đường nối vùng mặt 0-10)
POSE_BODY_CONNECTIONS = frozenset(
    (start, end)
    for start, end in mp.solutions.holistic.POSE_CONNECTIONS
    if start >= 11 and end >= 11
)


def draw_body_pose_landmarks(
    image: np.ndarray,
    pose_landmarks: Any,
    *,
    point_color: tuple[int, int, int] = (60, 180, 75),
    line_color: tuple[int, int, int] = (255, 225, 25),
    thickness: int = 1,
    circle_radius: int = 1,
    min_visibility: float = 0.5,
) -> None:
    """Vẽ khung xương thân người từ MediaPipe Pose, loại bỏ hoàn toàn các điểm trên khuôn mặt (0-10)."""
    if pose_landmarks is None or not hasattr(pose_landmarks, "landmark"):
        return

    h, w = image.shape[:2]
    coords: dict[int, tuple[int, int]] = {}

    for idx, lm in enumerate(pose_landmarks.landmark):
        if idx < 11:
            # Bỏ qua hoàn toàn các điểm trên khuôn mặt (0: Mũi, 1-3: Mắt T, 4-6: Mắt P, 7-8: Tai, 9-10: Miệng)
            continue
        if lm.HasField("visibility") and lm.visibility < min_visibility:
            continue
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        if 0 <= cx < w and 0 <= cy < h:
            coords[idx] = (cx, cy)

    # 1. Vẽ các đường nối thân người
    for start, end in POSE_BODY_CONNECTIONS:
        if start in coords and end in coords:
            cv2.line(image, coords[start], coords[end], line_color, thickness)

    # 2. Vẽ các điểm khớp thân người (viền trắng + chấm màu)
    circle_border_radius = max(circle_radius + 1, int(circle_radius * 1.2))
    for pt in coords.values():
        cv2.circle(image, pt, circle_border_radius, (255, 255, 255), thickness)
        cv2.circle(image, pt, circle_radius, point_color, -1)



@dataclass
class RecordedFrame:
    frame: np.ndarray
    results: Any
    left_present: bool
    right_present: bool
    timestamp: float


@dataclass
class SaveTask:
    frames: list[RecordedFrame]
    label: str
    confidence: float
    accepted: bool
    reason: str
    top_candidates: list[dict[str, Any]]
    duration_seconds: float
    output_dir: Path
    fps: float
    record_mode: str = "both"
    signer_name: str = "guest"
    target_gesture: str | None = None


class GestureVideoRecorder:
    """Bộ thu và xuất video cử chỉ tự động phục vụ phân tích sai số."""

    def __init__(
        self,
        record_dir: str | Path | None = None,
        fps: float = 30.0,
        enabled: bool = True,
        pre_roll_seconds: float = 0.8,
        record_mode: str = "both",
    ):
        self.enabled = enabled
        self.fps = max(5.0, float(fps))
        self.pre_roll_seconds = float(pre_roll_seconds)
        self.output_dir = resolve_record_dir(record_dir)
        self.pre_roll_len = max(5, int(self.fps * pre_roll_seconds))
        self.record_mode = record_mode if record_mode in ("both", "skeleton", "raw") else "both"

        self.pre_roll_buffer: deque[RecordedFrame] = deque(maxlen=self.pre_roll_len)
        self.current_gesture_frames: list[RecordedFrame] = []
        self.recording_active = False

        self._task_queue: queue.Queue[SaveTask | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_holistic = mp.solutions.holistic

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="GestureVideoRecorderWorker")
        self._worker_thread.start()

    def feed_frame(
        self,
        frame: np.ndarray,
        obs: Any,
        in_segment: bool,
        now: float,
    ) -> None:
        """Nhận từng frame thời gian thực và quản lý bộ đệm cử chỉ."""
        if not self.enabled:
            return

        rec_item = RecordedFrame(
            frame=frame.copy(),
            results=getattr(obs, "results", None),
            left_present=bool(getattr(obs, "left_hand_present", False)),
            right_present=bool(getattr(obs, "right_hand_present", False)),
            timestamp=now,
        )

        if in_segment:
            if not self.recording_active:
                # Bắt đầu cử chỉ mới: chỉ nạp các frame pre-roll thực sự mới trong khoảng pre_roll_seconds gần nhất
                self.recording_active = True
                cutoff = now - max(0.25, self.pre_roll_seconds)
                self.current_gesture_frames = [f for f in self.pre_roll_buffer if f.timestamp >= cutoff]
            self.current_gesture_frames.append(rec_item)
        else:
            if self.recording_active:
                # Cử chỉ vừa kết thúc trong tracker nhưng chưa được quyết định lưu hay hủy
                pass
            self.pre_roll_buffer.append(rec_item)

    def cancel_gesture(self) -> None:
        """Hủy bỏ cử chỉ hiện tại nếu bị reject trước khi phân loại (ví dụ quá ngắn hoặc nhấn Clear)."""
        self.recording_active = False
        self.current_gesture_frames.clear()

    def save_gesture(
        self,
        decision: Any,
        labels: list[str] | None = None,
        duration: float = 0.0,
        signer_name: str | None = None,
        target_gesture: str | None = None,
    ) -> None:
        """Chốt cử chỉ và đẩy sang background worker để xuất video kèm chẩn đoán."""
        if not self.enabled:
            self.cancel_gesture()
            return

        if not self.current_gesture_frames:
            print("[RECORDER WARNING] Bộ đệm rỗng khi kết thúc cử chỉ, không có video để lưu.", file=sys.stderr)
            self.cancel_gesture()
            return

        frames_to_save = list(self.current_gesture_frames)
        self.cancel_gesture()

        label = str(getattr(decision, "label", "Unknown"))
        confidence = float(getattr(decision, "confidence", 0.0))
        accepted = bool(getattr(decision, "accepted", False))
        reason = str(getattr(decision, "reason", "no reason"))

        # Trích xuất Top K ứng viên có xác suất cao nhất để chẩn đoán
        top_candidates: list[dict[str, Any]] = []
        probabilities = getattr(decision, "probabilities", None)
        if probabilities is not None and labels is not None and len(probabilities) == len(labels):
            probs = np.asarray(probabilities, dtype=np.float32)
            top_k_indices = np.argsort(probs)[::-1][:5]
            for idx in top_k_indices:
                top_candidates.append(
                    {
                        "label": labels[idx],
                        "confidence": float(probs[idx]),
                    }
                )
        else:
            top_candidates.append({"label": label, "confidence": confidence})

        calc_duration = duration
        if calc_duration <= 0.0 and len(frames_to_save) >= 2:
            calc_duration = max(0.0, frames_to_save[-1].timestamp - frames_to_save[0].timestamp)

        clean_signer = (signer_name or "").strip() or "guest"

        task = SaveTask(
            frames=frames_to_save,
            label=label,
            confidence=confidence,
            accepted=accepted,
            reason=reason,
            top_candidates=top_candidates,
            duration_seconds=calc_duration,
            output_dir=self.output_dir,
            fps=self.fps,
            record_mode=self.record_mode,
            signer_name=clean_signer,
            target_gesture=target_gesture,
        )
        self._task_queue.put(task)

    def close(self) -> None:
        """Đóng recorder và đợi worker hoàn thành."""
        if not self.enabled:
            return
        self._task_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

    def _draw_landmarks_and_hud(
        self,
        frame: np.ndarray,
        item: RecordedFrame,
        frame_idx: int,
        total_frames: int,
        label: str,
        confidence: float,
        accepted: bool,
    ) -> np.ndarray:
        """Vẽ khung xương MediaPipe và HUD phân tích lên frame."""
        annotated = frame.copy()
        results = item.results

        # 1. Vẽ khung xương từ MediaPipe (nếu có)
        if results is not None:
            if hasattr(results, "pose_landmarks") and results.pose_landmarks:
                draw_body_pose_landmarks(
                    annotated,
                    results.pose_landmarks,
                    point_color=(60, 180, 75),
                    line_color=(255, 225, 25),
                    thickness=2,
                    circle_radius=2,
                )
            if hasattr(results, "left_hand_landmarks") and results.left_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    results.left_hand_landmarks,
                    self._mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self._mp_drawing.DrawingSpec(color=(230, 25, 75), thickness=2, circle_radius=3),
                    connection_drawing_spec=self._mp_drawing.DrawingSpec(color=(245, 130, 48), thickness=2, circle_radius=2),
                )
            if hasattr(results, "right_hand_landmarks") and results.right_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    results.right_hand_landmarks,
                    self._mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self._mp_drawing.DrawingSpec(color=(0, 130, 200), thickness=2, circle_radius=3),
                    connection_drawing_spec=self._mp_drawing.DrawingSpec(color=(70, 240, 240), thickness=2, circle_radius=2),
                )

        # 2. Vẽ thanh trạng thái Header (HUD)
        h, w = annotated.shape[:2]
        header_height = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_height), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Biểu tượng và thông tin recording
        cv2.circle(annotated, (18, 21), 7, (0, 0, 255), -1)
        cv2.putText(annotated, "REC", (32, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(
            annotated,
            f"Frame {frame_idx + 1}/{total_frames}",
            (85, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Trạng thái theo dõi tay trái / phải
        left_color = (0, 255, 0) if item.left_present else (0, 0, 220)
        right_color = (0, 255, 0) if item.right_present else (0, 0, 220)
        left_str = "L: OK" if item.left_present else "L: LOST"
        right_str = "R: OK" if item.right_present else "R: LOST"

        cv2.putText(annotated, left_str, (w - 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
        cv2.putText(annotated, right_str, (w - 95, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

        # 3. Vẽ Banner kết quả ở chân video
        footer_height = 36
        footer_overlay = annotated.copy()
        cv2.rectangle(footer_overlay, (0, h - footer_height), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(footer_overlay, 0.8, annotated, 0.2, 0, annotated)

        status_tag = "[ACCEPTED]" if accepted else "[REJECTED]"
        tag_color = (50, 205, 50) if accepted else (30, 30, 220)
        cv2.putText(annotated, status_tag, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_color, 2)

        # Vẽ nhãn nhận diện tiếng Việt
        res_text = f"{label} ({confidence:.1%})"
        annotated = draw_unicode_text(
            annotated,
            res_text,
            (140, h - 30),
            font_size=18,
            color_bgr=(255, 255, 255),
        )

        return annotated

    def _render_diagnostic_card(
        self,
        base_frame: np.ndarray,
        task: SaveTask,
        left_ratio: float,
        right_ratio: float,
    ) -> np.ndarray:
        """Tạo màn hình tổng kết chẩn đoán sai số hiển thị ở cuối video."""
        card = base_frame.copy()
        h, w = card.shape[:2]

        # Phủ bóng mờ toàn màn hình
        dark = np.zeros_like(card)
        cv2.addWeighted(card, 0.25, dark, 0.75, 0, card)

        # Khung bảng chẩn đoán ở giữa
        box_w, box_h = min(w - 60, 580), min(h - 60, 380)
        bx1 = (w - box_w) // 2
        by1 = (h - box_h) // 2
        bx2 = bx1 + box_w
        by2 = by1 + box_h

        # Khung viền và nền card
        card_overlay = card.copy()
        cv2.rectangle(card_overlay, (bx1, by1), (bx2, by2), (32, 35, 42), -1)
        cv2.addWeighted(card_overlay, 0.9, card, 0.1, 0, card)

        border_color = (46, 204, 113) if task.accepted else (70, 70, 230)
        cv2.rectangle(card, (bx1, by1), (bx2, by2), border_color, 2)

        # Tiêu đề Card
        status_text = "NHẬN DIỆN THÀNH CÔNG (ACCEPTED)" if task.accepted else "TỪ CHỐI / CHƯA ĐẠT (REJECTED)"
        status_color = (46, 204, 113) if task.accepted else (80, 80, 240)
        card = draw_unicode_text(
            card,
            status_text,
            (bx1 + 25, by1 + 20),
            font_size=20,
            color_bgr=status_color,
        )

        # Nhãn dự đoán chính
        pred_line = f"Cử chỉ: {task.label}"
        card = draw_unicode_text(
            card,
            pred_line,
            (bx1 + 25, by1 + 55),
            font_size=24,
            color_bgr=(255, 255, 255),
        )

        # Độ tin cậy & Thời lượng
        metrics_line = (
            f"Độ tin cậy: {task.confidence:.1%}  |  "
            f"Thời lượng: {task.duration_seconds:.2f}s ({len(task.frames)} frames)"
        )
        card = draw_unicode_text(
            card,
            metrics_line,
            (bx1 + 25, by1 + 95),
            font_size=16,
            color_bgr=(200, 200, 200),
        )

        # Lý do (đặc biệt quan trọng nếu REJECTED)
        reason_line = f"Lý do phân loại: {task.reason}"
        reason_color = (180, 220, 180) if task.accepted else (100, 150, 255)
        card = draw_unicode_text(
            card,
            reason_line,
            (bx1 + 25, by1 + 125),
            font_size=15,
            color_bgr=reason_color,
        )

        # Đường kẻ phân cách
        cv2.line(card, (bx1 + 20, by1 + 155), (bx2 - 20, by1 + 155), (80, 85, 95), 1)

        # Danh sách Top Candidates (Phân tích xem mô hình nhầm với nhãn nào)
        card = draw_unicode_text(
            card,
            "Top dự đoán khả dĩ nhất (Phân tích sai số):",
            (bx1 + 25, by1 + 165),
            font_size=16,
            color_bgr=(240, 210, 100),
        )

        cand_y = by1 + 195
        for i, cand in enumerate(task.top_candidates[:3]):
            prefix = f"{i + 1}. {cand['label']}:"
            score_str = f"{cand['confidence']:.1%}"
            bar_text = f"{prefix:<25} {score_str}"
            card = draw_unicode_text(
                card,
                bar_text,
                (bx1 + 35, cand_y),
                font_size=15,
                color_bgr=(230, 230, 230),
            )
            cand_y += 26

        # Thống kê bắt nét bàn tay của MediaPipe
        cv2.line(card, (bx1 + 20, cand_y + 6), (bx2 - 20, cand_y + 6), (80, 85, 95), 1)
        tracking_info = f"MediaPipe bắt nét: Tay trái {left_ratio:.0%}  |  Tay phải {right_ratio:.0%}"
        card = draw_unicode_text(
            card,
            tracking_info,
            (bx1 + 25, cand_y + 16),
            font_size=14,
            color_bgr=(180, 180, 180),
        )

        if left_ratio < 0.5 and right_ratio < 0.5:
            card = draw_unicode_text(
                card,
                "Cảnh báo: Cả hai tay bị mất nét nhiều frame! Hãy đưa tay rõ trước camera.",
                (bx1 + 25, cand_y + 40),
                font_size=13,
                color_bgr=(50, 50, 255),
            )

        return card

    def _worker_loop(self) -> None:
        """Background thread thực hiện xuất file video và metadata JSON."""
        while True:
            try:
                task = self._task_queue.get()
                if task is None:
                    break
                self._process_save_task(task)
            except Exception as exc:
                print(f"[RECORDER ERROR] Lỗi khi xử lý lưu video: {exc}")
            finally:
                self._task_queue.task_done()

    def _process_save_task(self, task: SaveTask) -> None:
        if not task.frames:
            return

        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        status_tag = "ACCEPTED" if task.accepted else "REJECTED"
        label_slug = slugify(task.label)
        conf_tag = f"{int(round(task.confidence * 100))}pct"
        signer_slug = slugify(task.signer_name) if task.signer_name and task.signer_name.lower() != "guest" else ""

        # Xác định thư mục lưu trữ theo từng cử chỉ luyện tập (nếu có)
        target_gesture_slug = slugify(task.target_gesture) if task.target_gesture and str(task.target_gesture).strip() else ""
        if target_gesture_slug:
            save_dir = task.output_dir / target_gesture_slug
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = task.output_dir

        if signer_slug:
            base_name = f"{timestamp_str}_{signer_slug}_{status_tag}_{label_slug}_{conf_tag}"
        else:
            base_name = f"{timestamp_str}_{status_tag}_{label_slug}_{conf_tag}"
        meta_path = save_dir / f"{base_name}.json"

        # Tính toán tỷ lệ phát hiện tay để phân tích
        total_frames = len(task.frames)
        left_frames = sum(1 for f in task.frames if f.left_present)
        right_frames = sum(1 for f in task.frames if f.right_present)
        left_ratio = left_frames / total_frames if total_frames > 0 else 0.0
        right_ratio = right_frames / total_frames if total_frames > 0 else 0.0

        # Tính toán FPS thực tế dựa trên timestamps để tốc độ video chuẩn xác 1:1 với ngoài đời
        if total_frames > 1:
            time_span = max(0.05, task.frames[-1].timestamp - task.frames[0].timestamp)
            effective_fps = (total_frames - 1) / time_span
        elif task.duration_seconds > 0.05:
            effective_fps = total_frames / task.duration_seconds
        else:
            effective_fps = task.fps

        actual_fps = max(5.0, min(60.0, float(effective_fps)))

        first_frame = task.frames[0].frame
        h, w = first_frame.shape[:2]

        save_skeleton = task.record_mode in ("both", "skeleton")
        save_raw = task.record_mode in ("both", "raw")

        skeleton_path = save_dir / f"{base_name}_skeleton.mp4"
        raw_path = save_dir / f"{base_name}_raw.mp4"

        def create_writer(path: Path) -> tuple[cv2.VideoWriter | None, Path]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w_obj = cv2.VideoWriter(str(path), fourcc, actual_fps, (w, h))
            if not w_obj.isOpened():
                path = path.with_suffix(".avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                w_obj = cv2.VideoWriter(str(path), fourcc, actual_fps, (w, h))
            return (w_obj if w_obj.isOpened() else None), path

        skeleton_writer = None
        raw_writer = None
        saved_video_names: list[str] = []

        if save_skeleton:
            skeleton_writer, skeleton_path = create_writer(skeleton_path)
            if skeleton_writer is not None:
                saved_video_names.append(skeleton_path.name)
            else:
                print(f"[RECORDER ERROR] Không thể mở VideoWriter cho file: {skeleton_path}", file=sys.stderr)

        if save_raw:
            raw_writer, raw_path = create_writer(raw_path)
            if raw_writer is not None:
                saved_video_names.append(raw_path.name)
            else:
                print(f"[RECORDER ERROR] Không thể mở VideoWriter cho file: {raw_path}", file=sys.stderr)

        if not saved_video_names:
            return

        last_rendered_skeleton = first_frame
        last_rendered_raw = first_frame

        try:
            # 1. Ghi các frame cử chỉ (bản skeleton có HUD & khớp xương, bản raw giữ nguyên bản gốc)
            for idx, item in enumerate(task.frames):
                if skeleton_writer is not None:
                    annotated = self._draw_landmarks_and_hud(
                        item.frame,
                        item,
                        idx,
                        total_frames,
                        task.label,
                        task.confidence,
                        task.accepted,
                    )
                    skeleton_writer.write(annotated)
                    last_rendered_skeleton = annotated

                if raw_writer is not None:
                    raw_frame = item.frame.copy()
                    raw_writer.write(raw_frame)
                    last_rendered_raw = raw_frame

            # 2. Ghi freeze-frame màn hình chẩn đoán sai số ở cuối video (~1.5s)
            freeze_frame_count = max(10, int(actual_fps * 1.5))

            if skeleton_writer is not None:
                diag_skeleton = self._render_diagnostic_card(
                    last_rendered_skeleton,
                    task,
                    left_ratio,
                    right_ratio,
                )
                for _ in range(freeze_frame_count):
                    skeleton_writer.write(diag_skeleton)

            if raw_writer is not None:
                diag_raw = self._render_diagnostic_card(
                    last_rendered_raw,
                    task,
                    left_ratio,
                    right_ratio,
                )
                for _ in range(freeze_frame_count):
                    raw_writer.write(diag_raw)

        finally:
            if skeleton_writer is not None:
                skeleton_writer.release()
            if raw_writer is not None:
                raw_writer.release()

        # 3. Lưu file metadata JSON cùng tên
        metadata = {
            "timestamp": now_dt.isoformat(),
            "signer": task.signer_name,
            "target_gesture": task.target_gesture,
            "videos": saved_video_names,
            "status": status_tag,
            "label": task.label,
            "confidence": task.confidence,
            "accepted": task.accepted,
            "reason": task.reason,
            "duration_seconds": round(task.duration_seconds, 3),
            "total_frames": total_frames,
            "fps": round(actual_fps, 2),
            "record_mode": task.record_mode,
            "hand_tracking": {
                "left_hand_ratio": round(left_ratio, 4),
                "right_hand_ratio": round(right_ratio, 4),
                "left_frames": left_frames,
                "right_frames": right_frames,
            },
            "top_candidates": task.top_candidates,
        }

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[RECORDER WARNING] Không thể lưu file metadata JSON: {exc}", file=sys.stderr)

        video_desc = " & ".join(saved_video_names)
        folder_info = f" -> videos/{target_gesture_slug}/" if target_gesture_slug else ""
        print(
            f"[RECORDER] Đã lưu video cử chỉ ({task.signer_name}){folder_info}: {video_desc} "
            f"({total_frames} frames @ {actual_fps:.1f} FPS chuẩn thực tế, {task.duration_seconds:.2f}s) -> {status_tag} {task.label} ({task.confidence:.1%})"
        )

        # 4. Tự động đẩy ngầm lên Hugging Face Dataset API (nếu có cấu hình HF_TOKEN)
        hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()
        hf_dataset = (os.environ.get("HF_DATASET_REPO") or "ntbii305/vslr-remote").strip()
        if hf_token and hf_dataset:
            folder_slug = f"{target_gesture_slug}/{signer_slug or 'guest'}" if target_gesture_slug else (signer_slug or "guest")
            files_to_upload = [meta_path] + [save_dir / name for name in saved_video_names]
            self._upload_to_huggingface_async(
                token=hf_token,
                repo_id=hf_dataset,
                folder_slug=folder_slug,
                file_paths=files_to_upload,
                commit_title=f"[{status_tag}] {task.signer_name}: {task.label} ({conf_tag})",
            )

    def _upload_to_huggingface_async(
        self,
        token: str,
        repo_id: str,
        folder_slug: str,
        file_paths: list[Path],
        commit_title: str,
    ) -> None:
        """Đẩy batch video và JSON lên Hugging Face Dataset ở luồng ngầm (0ms ảnh hưởng camera)."""
        def _upload_worker() -> None:
            try:
                from huggingface_hub import CommitOperationAdd, HfApi

                api = HfApi(token=token)
                operations = []
                for fp in file_paths:
                    if fp.exists():
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"data/{folder_slug}/{fp.name}",
                                path_or_fileobj=str(fp),
                            )
                        )
                if operations:
                    api.create_commit(
                        repo_id=repo_id,
                        repo_type="dataset",
                        operations=operations,
                        commit_message=commit_title,
                    )
                    print(f"[HF DATASET] Đã đồng bộ {len(operations)} file lên Hugging Face Dataset '{repo_id}' -> data/{folder_slug}/")
            except Exception as exc:
                print(f"[HF DATASET WARNING] Không thể đồng bộ lên Hugging Face ({repo_id}): {exc}", file=sys.stderr)

        threading.Thread(target=_upload_worker, daemon=True, name="HFDatasetUploader").start()
