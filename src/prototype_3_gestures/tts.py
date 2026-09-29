"""Vietnamese Text-to-Speech integration for VSLR using VieNeu-TTS and fallbacks."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import unicodedata
from typing import Optional

import numpy as np

VIENEU_PRESET_VOICES = [
    "Trúc Ly",
    "Mai Anh",
    "Minh Quân",
    "Thùy Dung",
    "Anh Khôi",
    "Ngọc Huyền",
    "Quang Sơn",
    "Ngọc Trân",
    "Minh Đức",
    "Phạm Tuyên",
    "Thái Sơn",
    "Xuân Vĩnh",
    "Thanh Bình",
    "Ngọc Linh",
    "Đoan Trang",
    "Thục Đoan",
    "Minh Triết",
    "Mỹ Duyên",
    "Quỳnh Anh",
    "Đức Trí",
    "Kim Thanh",
    "Mạnh Dũng",
    "Adam",
    "Adam bựa",
    "Thiền Tâm Đức",
]


def _strip_diacritics(text: str) -> str:
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D").lower().strip()


def normalize_speech_text(text: str) -> str:
    """Chuẩn hóa văn bản và thêm dấu câu/ngữ điệu tự nhiên cho bộ phát âm."""
    s = text.strip()
    if not s:
        return ""
    if s.endswith((".", "!", "?", "…")):
        return s

    lower = s.lower()
    # Các từ / cụm từ hỏi thường gặp trong ngôn ngữ ký hiệu tiếng Việt
    question_endings = (
        "không", "k", "ko", "khong",
        "gì", "gi",
        "đâu", "dau",
        "nào", "nao",
        "sao",
        "thế nào", "the nao",
        "như thế nào", "nhu the nao",
        "mấy", "may",
        "bao nhiêu", "bao nhieu",
        "được không", "duoc khong",
        "sao thế", "sao the",
        "chuyện gì", "chuyen gi",
        "vấn đề gì", "van de gi",
    )
    is_question = any(lower.endswith(w) for w in question_endings) or any(
        lower.startswith(w) for w in ("bạn có", "có phải", "tại sao", "làm sao", "chuyện gì", "đi đâu", "mấy tuổi")
    )

    # Các từ chào hỏi, cảm ơn, biểu cảm
    exclamation_phrases = (
        "xin chào", "chào bạn",
        "cảm ơn", "cảm ơn bạn",
        "tạm biệt",
        "hẹn gặp lại",
        "rất vui được gặp bạn",
        "rất vui",
        "về nhà cẩn thận",
        "chúc mừng",
        "cứu tôi", "cứu với",
        "gọi xe cứu thương",
        "xin lỗi",
        "lâu rồi không gặp",
    )
    is_exclamation = any(lower == p or lower.startswith(p) for p in exclamation_phrases)

    if is_question:
        return s + "?"
    elif is_exclamation:
        return s + "!"
    return s + "."


def resolve_vieneu_voice(requested_voice: str) -> str:
    """Map user-provided voice string to exact VieNeu preset name.

    Supports exact match, case-insensitive, and unaccented matching
    (e.g., 'truc ly' or 'Truc Ly' -> 'Trúc Ly', 'mai anh' -> 'Mai Anh').
    """
    if not requested_voice or not str(requested_voice).strip():
        return "Trúc Ly"

    req_clean = str(requested_voice).strip()
    # 1. Exact match
    for v in VIENEU_PRESET_VOICES:
        if v == req_clean:
            return v

    # 2. Case-insensitive match
    req_lower = req_clean.lower()
    for v in VIENEU_PRESET_VOICES:
        if v.lower() == req_lower:
            return v

    # 3. Unaccented match
    req_unaccented = _strip_diacritics(req_clean)
    for v in VIENEU_PRESET_VOICES:
        if _strip_diacritics(v) == req_unaccented:
            return v

    return req_clean


class TTSManager:
    """Manages Vietnamese speech synthesis and asynchronous playback."""

    def __init__(
        self,
        engine: str = "vieneu",
        voice: str = "Trúc Ly",
        enabled: bool = True,
        temperature: float = 0.65,
        top_p: float = 0.92,
        top_k: int = 20,
        repetition_penalty: float = 1.15,
    ) -> None:
        self.engine = engine.lower()
        self.voice = resolve_vieneu_voice(voice) if self.engine == "vieneu" else voice
        self.enabled = enabled
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._vieneu_instance = None
        self._lock = threading.Lock()

        if self.enabled and self.engine != "none":
            self._start_worker()
            if self.engine == "vieneu":
                threading.Thread(target=self._preload_vieneu, daemon=True).start()

    def _start_worker(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    @staticmethod
    def _fix_hf_onnx_symlinks() -> None:
        """Chuyển symlink trong cache Hugging Face thành hardlink trên Linux để tránh lỗi
        ONNX Runtime: External data path validation failed (escapes model directory).
        """
        try:
            from pathlib import Path
            import shutil

            try:
                import huggingface_hub.file_download as hf_fd
                def _hardlink_or_copy(src: str, dst: str, new_blob: bool = False) -> None:
                    if os.path.exists(dst) or os.path.islink(dst):
                        os.remove(dst)
                    try:
                        os.link(src, dst)
                    except Exception:
                        shutil.copy2(src, dst)
                hf_fd._create_symlink = _hardlink_or_copy
            except Exception:
                pass

            hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface" / "hub"))
            if not hf_home.is_dir():
                return
            for link_path in hf_home.rglob("*"):
                try:
                    if link_path.is_symlink():
                        target = link_path.resolve()
                        if target.is_file():
                            link_path.unlink()
                            try:
                                os.link(target, link_path)
                            except Exception:
                                shutil.copy2(target, link_path)
                except Exception:
                    continue
        except Exception:
            pass

    def _create_vieneu_instance(self):
        self._fix_hf_onnx_symlinks()
        from vieneu import Vieneu

        is_cloud = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACES_ZERO_GPU"))
        if is_cloud:
            # Trên Cloud / Hugging Face ZeroGPU: ép chạy ONNX CPU để tránh lỗi "No CUDA GPUs are available"
            try:
                return Vieneu(device="cpu", backend="onnx")
            except Exception:
                self._fix_hf_onnx_symlinks()
                return Vieneu(device="cpu", backend="onnx")
        try:
            return Vieneu()
        except Exception:
            # Fallback an toàn sang ONNX CPU nếu CUDA không khả dụng hoặc dính symlink
            self._fix_hf_onnx_symlinks()
            return Vieneu(device="cpu", backend="onnx")

    def _preload_vieneu(self) -> None:
        with self._lock:
            if self._vieneu_instance is None:
                try:
                    print(f"[TTS] Đang nạp model giọng đọc VieNeu-TTS ({self.voice})...")
                    self._vieneu_instance = self._create_vieneu_instance()
                    print(f"[TTS] Model giọng đọc VieNeu-TTS đã sẵn sàng!")
                except Exception as exc:
                    print(f"[TTS] Không thể khởi tạo VieNeu-TTS ({exc}), sẽ dùng bộ phát âm dự phòng.", file=sys.stderr)
                    self._vieneu_instance = False

    def _get_vieneu(self):
        with self._lock:
            if self._vieneu_instance is None:
                try:
                    print(f"[TTS] Đang nạp model giọng đọc VieNeu-TTS ({self.voice})...")
                    self._vieneu_instance = self._create_vieneu_instance()
                    print(f"[TTS] Model giọng đọc VieNeu-TTS đã sẵn sàng!")
                except Exception as exc:
                    print(f"[TTS] Không thể khởi tạo VieNeu-TTS ({exc}), sẽ dùng bộ phát âm dự phòng.", file=sys.stderr)
                    self._vieneu_instance = False
            return self._vieneu_instance if self._vieneu_instance is not False else None

    def _worker_loop(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            try:
                self._synthesize_and_play(text)
            except Exception as exc:
                print(f"[TTS Error] {exc}", file=sys.stderr)
            finally:
                self._queue.task_done()

    def _synthesize_and_play(self, text: str) -> None:
        normalized_text = normalize_speech_text(text)
        if not normalized_text:
            return

        print(f"TTS> Đang đọc: {normalized_text}")
        if self.engine == "vieneu":
            try:
                vieneu_engine = self._get_vieneu()
                if vieneu_engine is not None:
                    import sounddevice as sd

                    # Sử dụng tham số sampling tối ưu để giọng mượt mà, triệt tiêu rè kim loại
                    audio = vieneu_engine.infer(
                        normalized_text,
                        voice=self.voice,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                    )
                    sr = getattr(vieneu_engine, "sample_rate", 48000)
                    audio_arr = np.asarray(audio, dtype=np.float32)

                    # Làm mượt âm đuôi (20ms fade-out) và đệm 0.3s khoảng lặng
                    # để âm thanh cuối cùng phát trọn vẹn, không bị ngắt cụt bởi hardware buffer
                    fade_len = min(len(audio_arr), int(sr * 0.02))
                    if fade_len > 0:
                        audio_arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

                    pad_samples = int(sr * 0.30)
                    padded_audio = np.pad(audio_arr, (0, pad_samples), mode="constant")

                    sd.play(padded_audio, samplerate=sr)
                    sd.wait()
                    time.sleep(0.10)  # Chờ buffer phần cứng xả hết ra loa hoàn toàn
                    print(f"TTS> Hoàn tất đọc: {normalized_text}")
                    return
            except Exception as exc:
                print(f"[TTS] VieNeu-TTS phát âm lỗi ({exc}), chuyển sang engine dự phòng...", file=sys.stderr)

        # Fallback 1: pyttsx3
        try:
            import pyttsx3  # type: ignore

            py_engine = pyttsx3.init()
            py_engine.say(text)
            py_engine.runAndWait()
            return
        except Exception:
            pass

        # Fallback 2: Windows System.Speech
        if os.name == "nt":
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Speak([Console]::In.ReadToEnd())"
            )
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    input=text,
                    text=True,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except OSError:
                pass

        print("TTS backend unavailable; sentence was printed instead.", file=sys.stderr)

    def speak(self, text: str) -> None:
        """Enqueue sentence to speak asynchronously without freezing video preview."""
        if not self.enabled or self.engine == "none" or not text.strip():
            return
        self._queue.put(text.strip())

    def stop(self) -> None:
        """Stop worker thread and wait for remaining items."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put(None)
            self._worker_thread.join(timeout=1.0)


# Global default manager
_GLOBAL_TTS: Optional[TTSManager] = None


def get_default_tts(
    engine: str = "vieneu",
    voice: str = "Trúc Ly",
    enabled: bool = True,
    temperature: float = 0.65,
    top_p: float = 0.92,
    top_k: int = 20,
    repetition_penalty: float = 1.15,
) -> TTSManager:
    global _GLOBAL_TTS
    resolved = resolve_vieneu_voice(voice) if engine == "vieneu" else voice
    if _GLOBAL_TTS is None or _GLOBAL_TTS.voice != resolved or _GLOBAL_TTS.engine != engine:
        if _GLOBAL_TTS is not None:
            _GLOBAL_TTS.stop()
        _GLOBAL_TTS = TTSManager(
            engine=engine,
            voice=voice,
            enabled=enabled,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    return _GLOBAL_TTS


def speak_text(text: str, voice: str = "Trúc Ly", engine: str = "vieneu") -> None:
    """Backward-compatible helper function to speak text."""
    manager = get_default_tts(engine=engine, voice=voice)
    manager.speak(text)
