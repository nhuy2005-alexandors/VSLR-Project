#!/usr/bin/env python3
"""VSLR Realtime 24/7 Daemon on AWS EC2 Singapore.

Chạy nền liên tục 24/7:
- Tự động duy trì FastAPI Backend (Uvicorn).
- Tự động duy trì Cloudflare Tunnel (tự kết nối lại nếu rớt mạng).
- Tự động phát sóng link mới lên Auto-Discovery cho Vercel.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

# Thêm src vào PYTHONPATH
_here = Path(__file__).resolve().parent
root_dir = _here.parent if _here.name == "scripts" else _here
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Giả lập module sounddevice cho Linux Headless Cloud
if "sounddevice" not in sys.modules:
    mock_sd = types.ModuleType("sounddevice")
    mock_sd.default = types.SimpleNamespace(samplerate=48000, channels=1, device=None)
    mock_sd.play = lambda *args, **kwargs: None
    mock_sd.stop = lambda *args, **kwargs: None
    mock_sd.wait = lambda *args, **kwargs: None
    mock_sd.query_devices = lambda *args, **kwargs: []
    mock_sd.InputStream = object
    mock_sd.OutputStream = object
    mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
    sys.modules["sounddevice"] = mock_sd


def run_uvicorn() -> None:
    import uvicorn
    from prototype_3_gestures.web_server import RealtimeVSLRPipeline, create_app

    model_path = root_dir / "models" / "gesture_lstm.pt"
    if not model_path.is_file():
        model_path = root_dir / "artifacts" / "v3-realtime-test-candidate" / "gesture_lstm.pt"

    pipeline = RealtimeVSLRPipeline(str(model_path), confidence_threshold=0.52)
    app = create_app(pipeline, web_dir=root_dir / "vslr-web")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


def sync_registry_loop(tunnel_url: str) -> None:
    """Phát sóng và giữ nhịp heartbeat (15 phút/lần) lên Auto-Discovery cho tên miền Vercel."""
    import urllib.request

    reg_url = "https://ntfy.sh/vslr_ctu_aws_active_backend_prod_v3"
    while True:
        try:
            req = urllib.request.Request(reg_url, data=tunnel_url.encode("utf-8"), method="POST")
            urllib.request.urlopen(req, timeout=5.0)
        except Exception:
            pass
        time.sleep(900)


def main() -> None:
    # 1. Khởi động AI Uvicorn Server trên luồng riêng
    server_thread = threading.Thread(target=run_uvicorn, daemon=True, name="UvicornServer")
    server_thread.start()
    time.sleep(3.0)

    cloudflared_bin = root_dir / "cloudflared"
    if not cloudflared_bin.is_file():
        cloudflared_bin = Path("cloudflared")

    url_re = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # 2. Vòng lặp giám sát Cloudflare Tunnel (Tự phục hồi 24/7 nếu rớt mạng)
    while True:
        print("[VSLR Daemon] Đang khởi tạo Cloudflare Tunnel...")
        try:
            proc = subprocess.Popen(
                [str(cloudflared_bin), "tunnel", "--url", "http://127.0.0.1:8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            found_url = False
            assert proc.stdout is not None
            for line in proc.stdout:
                if not found_url:
                    m = url_re.search(line)
                    if m:
                        found_url = True
                        url = m.group(0)
                        # Ghi URL ra tệp để setup script đọc
                        url_file = root_dir / "active_tunnel_url.txt"
                        url_file.write_text(url + "\n", encoding="utf-8")

                        # Bật luồng đồng bộ ngầm lên Auto-Discovery
                        threading.Thread(target=sync_registry_loop, args=(url,), daemon=True).start()
                        print(f"🎉 [VSLR Daemon 24/7] Active URL: {url}")

            proc.wait()
        except Exception as exc:
            print(f"[VSLR Daemon Warning] Tunnel gián đoạn ({exc}), sẽ tự động kết nối lại sau 3s...")
        time.sleep(3.0)


if __name__ == "__main__":
    main()
