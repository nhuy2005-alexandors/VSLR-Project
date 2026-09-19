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
    ) -> None:
        self.engine = engine.lower()
        self.voice = resolve_vieneu_voice(voice) if self.engine == "vieneu" else voice
        self.enabled = enabled
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

    def _preload_vieneu(self) -> None:
        with self._lock:
            if self._vieneu_instance is None:
                try:
                    print(f"[TTS] Đang nạp model giọng đọc VieNeu-TTS ({self.voice})...")
                    from vieneu import Vieneu

                    self._vieneu_instance = Vieneu()
                    print(f"[TTS] Model giọng đọc VieNeu-TTS đã sẵn sàng!")
                except Exception as exc:
                    print(f"[TTS] Không thể khởi tạo VieNeu-TTS ({exc}), sẽ dùng bộ phát âm dự phòng.", file=sys.stderr)
                    self._vieneu_instance = False

    def _get_vieneu(self):
        with self._lock:
            if self._vieneu_instance is None:
                try:
                    print(f"[TTS] Đang nạp model giọng đọc VieNeu-TTS ({self.voice})...")
                    from vieneu import Vieneu

                    self._vieneu_instance = Vieneu()
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
        normalized_text = text.strip()
        if normalized_text and not normalized_text.endswith((".", "!", "?", "…")):
            normalized_text += "."

        print(f"TTS> Đang đọc: {normalized_text}")
        if self.engine == "vieneu":
            try:
                vieneu_engine = self._get_vieneu()
                if vieneu_engine is not None:
                    import sounddevice as sd

                    audio = vieneu_engine.infer(normalized_text, voice=self.voice)
                    sr = getattr(vieneu_engine, "sample_rate", 48000)
                    audio_arr = np.asarray(audio, dtype=np.float32)

                    # Làm mượt âm đuôi (20ms fade-out) và đệm 0.4s khoảng lặng
                    # để âm thanh cuối cùng phát trọn vẹn, không bị ngắt cụt bởi hardware buffer
                    fade_len = min(len(audio_arr), int(sr * 0.02))
                    if fade_len > 0:
                        audio_arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

                    pad_samples = int(sr * 0.40)
                    padded_audio = np.pad(audio_arr, (0, pad_samples), mode="constant")

                    sd.play(padded_audio, samplerate=sr)
                    sd.wait()
                    time.sleep(0.15)  # Chờ buffer phần cứng xả hết ra loa hoàn toàn
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


def get_default_tts(engine: str = "vieneu", voice: str = "Trúc Ly", enabled: bool = True) -> TTSManager:
    global _GLOBAL_TTS
    resolved = resolve_vieneu_voice(voice) if engine == "vieneu" else voice
    if _GLOBAL_TTS is None or _GLOBAL_TTS.voice != resolved or _GLOBAL_TTS.engine != engine:
        if _GLOBAL_TTS is not None:
            _GLOBAL_TTS.stop()
        _GLOBAL_TTS = TTSManager(engine=engine, voice=voice, enabled=enabled)
    return _GLOBAL_TTS


def speak_text(text: str, voice: str = "Trúc Ly", engine: str = "vieneu") -> None:
    """Backward-compatible helper function to speak text."""
    manager = get_default_tts(engine=engine, voice=voice)
    manager.speak(text)
