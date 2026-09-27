"""Script khởi động đồng thời Backend VSLR (30 FPS) + Cloudflare Tunnel (HTTPS) + Đồng bộ Hugging Face Dataset."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Thêm src vào PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from prototype_3_gestures.vsl3.console import configure_utf8_stdio

configure_utf8_stdio()


def ensure_cloudflared(root: Path) -> str:
    """Kiểm tra hoặc tự động tải cloudflared.exe nếu chưa có."""
    local_bin = root / "cloudflared.exe"
    if local_bin.is_file():
        return str(local_bin)

    # Kiểm tra trong PATH hệ thống
    try:
        subprocess.run(["cloudflared", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "cloudflared"
    except Exception:
        pass

    print("[Cloudflare] Đang tự động tải công cụ cloudflared.exe (chỉ tải 1 lần đầu)...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    urllib.request.urlretrieve(url, str(local_bin))
    print("[Cloudflare] ✓ Đã tải xong cloudflared.exe!")
    return str(local_bin)


def copy_to_clipboard(text: str) -> None:
    """Tự động copy đường link vào Clipboard Windows để dán ngay."""
    try:
        subprocess.run(["clip"], input=text.strip().encode("utf-8"), check=False)
    except Exception:
        pass


def start_vslr_server(port: int = 8000) -> None:
    """Khởi động FastAPI VSLR Server trên luồng nền."""
    import uvicorn
    from prototype_3_gestures.web_server import RealtimeVSLRPipeline, create_app

    model_candidates = [
        root_dir / "artifacts" / "v3-realtime-test-candidate" / "gesture_lstm.pt",
        root_dir / "models" / "gesture_lstm.pt",
    ]
    model_path = next((p for p in model_candidates if p.is_file()), model_candidates[0])

    pipeline = RealtimeVSLRPipeline(
        model_path=model_path,
        camera_id=0,
        confidence_threshold=0.62,
        cooldown=1.0,
        word_gap=0.35,
        sentence_gap=1.8,
        tts_voice="Trúc Ly",
        tts_engine="vieneu",
        record_dir=root_dir / "videos",
        allow_uncalibrated=True,
    )
    app = create_app(pipeline, web_dir=root_dir / "vslr-web")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def load_local_env(root: Path) -> None:
    """Đọc cấu hình bảo mật từ file vslr.env cục bộ (nếu có)."""
    env_file = root / "vslr.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    # Cấu hình tự động đồng bộ video lên Hugging Face Dataset
    load_local_env(root_dir)
    os.environ.setdefault("HF_DATASET_REPO", "ntbii305/vslr-remote")

    cf_bin = ensure_cloudflared(root_dir)
    port = 8000

    print("\n[1/2] Đang khởi động AI VSLR Server (30 FPS + VieNeu-TTS Trúc Ly) tại http://localhost:8000 ...")
    server_thread = threading.Thread(target=start_vslr_server, args=(port,), daemon=True)
    server_thread.start()
    time.sleep(3.5)

    print("[2/2] Đang mở đường hầm Cloudflare Tunnel tốc độ cao (Ping nội địa 5-15ms)...")
    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    public_url = None

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if public_url is None:
                match = url_pattern.search(line)
                if match:
                    public_url = match.group(0)
                    copy_to_clipboard(public_url)
                    vercel_link = f"https://vslr-project-v3.vercel.app/?backend={public_url}"
                    print("\n" + "=" * 76)
                    print("🎉 HỆ THỐNG VSLR ĐÃ PUBLIC THÀNH CÔNG (TỐC ĐỘ 30 FPS NHƯ LOCALHOST!)")
                    print("=" * 76)
                    print(f"👉 1. LINK GỬI BẠN BÈ (Đã tự động Copy vào Clipboard - Chỉ việc Ctrl+V):")
                    print(f"      {public_url}")
                    print(f"\n👉 2. HOẶC DÙNG QUA TÊN MIỀN VERCEL CỦA BẠN:")
                    print(f"      {vercel_link}")
                    print(f"\n👉 3. LINK TRÊN MÁY CỦA BẠN:")
                    print(f"      http://localhost:{port}")
                    print("-" * 76)
                    print("💾 Nơi lưu Video tự động:")
                    print(f"   • Trên ổ cứng máy bạn: {root_dir / 'videos'}")
                    print(f"   • Trên Hugging Face:   https://huggingface.co/datasets/ntbii305/vslr-remote")
                    print("=" * 76)
                    print("\n(Giữ nguyên cửa sổ này khi bạn bè đang test. Nhấn Ctrl+C để tắt Server)\n")
                    webbrowser.open(f"http://localhost:{port}")
    except KeyboardInterrupt:
        print("\nĐang đóng đường hầm Cloudflare và dừng Server...")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
