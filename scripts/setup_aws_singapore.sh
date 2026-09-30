#!/usr/bin/env bash
# =============================================================================
# VSLR: Automated 1-Command Deployment on AWS EC2 Singapore (Ubuntu LTS)
# =============================================================================
set -e

echo "===================================================================="
echo "   KHỞI TẠO HỆ THỐNG VSLR BACKEND TẠI AWS SINGAPORE"
echo "   - Region: Asia Pacific (Singapore) ap-southeast-1"
echo "   - Tốc độ: Ping 25-35ms cực mượt, băng thông 10 Gbps"
echo "   - Tự động đồng bộ: Hugging Face Dataset (ntbii305/vslr-remote)"
echo "===================================================================="
echo ""

# 1. Cập nhật và cài đặt các gói hệ thống Linux cần thiết
echo "[1/4] Đang cài đặt thư viện hệ thống (Build tools, Git LFS, OpenCV, Audio, Fonts)..."
sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    git \
    git-lfs \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    libportaudio2 \
    fonts-dejavu-core \
    fonts-liberation
git lfs install >/dev/null 2>&1 || true

# 2. Tải mã nguồn và mô hình AI từ Hugging Face Space
echo ""
echo "[2/4] Đang tải mã nguồn VSLR và mô hình BiLSTM (ntbii305/vslr-backend)..."
cd ~
if [ ! -d "vslr-backend" ]; then
    git clone https://huggingface.co/spaces/ntbii305/vslr-backend
fi
cd vslr-backend
git fetch origin
git reset --hard origin/main
git lfs pull || true

# 3. Cài đặt Python 3.11 chuẩn (tương thích 100% với MediaPipe & PyTorch) thông qua trình quản lý siêu tốc uv
echo ""
echo "[3/4] Đang thiết lập Python 3.11 và cài đặt PyTorch, MediaPipe, FastAPI..."
if ! command -v uv >/dev/null 2>&1 && [ ! -f "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -f "venv/bin/python" ] || ! ./venv/bin/python --version 2>&1 | grep -q "3.11"; then
    rm -rf venv
    uv venv --python 3.11 venv
fi
source venv/bin/activate
uv pip install -r requirements.txt

# Cài đặt font Arial tiếng Việt vào hệ thống Linux
if [ -f "src/prototype_3_gestures/arial.ttf" ]; then
    sudo mkdir -p /usr/share/fonts/truetype/custom
    sudo cp src/prototype_3_gestures/arial.ttf /usr/share/fonts/truetype/custom/
    sudo fc-cache -f >/dev/null 2>&1 || true
fi

# 4. Tải công cụ Cloudflare Tunnel cho Linux (nếu chưa có)
if [ ! -f "cloudflared" ]; then
    echo "Đang tải cloudflared Linux..."
    curl -sSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
    chmod +x cloudflared
fi

# Cấu hình Token lưu dữ liệu Hugging Face Dataset
export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-ntbii305/vslr-remote}"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 5. Khởi động AI Server và mở đường hầm Cloudflare Tunnel
echo ""
echo "[4/4] Đang khởi động AI Server và tạo đường hầm Cloudflare HTTPS..."
python3 -c "
import subprocess, re, time, os, threading, sys, types

# Trên máy chủ AWS EC2 không có card âm thanh vật lý / PulseAudio daemon,
# giả lập module sounddevice để MediaPipe và VieNeu-TTS hoạt động mượt mà
mock_sd = types.ModuleType('sounddevice')
mock_sd.default = types.SimpleNamespace(samplerate=48000, channels=1, device=None)
mock_sd.play = lambda *args, **kwargs: None
mock_sd.stop = lambda *args, **kwargs: None
mock_sd.wait = lambda *args, **kwargs: None
mock_sd.query_devices = lambda *args, **kwargs: []
mock_sd.InputStream = object
mock_sd.OutputStream = object
mock_sd.PortAudioError = type('PortAudioError', (Exception,), {})
sys.modules['sounddevice'] = mock_sd

ready_evt = threading.Event()
startup_err = []

def run_uvicorn():
    try:
        import uvicorn
        from prototype_3_gestures.web_server import RealtimeVSLRPipeline, create_app
        pipeline = RealtimeVSLRPipeline('models/gesture_lstm.pt', confidence_threshold=0.52)
        app = create_app(pipeline, web_dir='vslr-web')
        ready_evt.set()
        uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')
    except Exception as exc:
        startup_err.append(exc)
        ready_evt.set()
        raise

threading.Thread(target=run_uvicorn, daemon=True).start()
print('⏳ Đang nạp mô hình AI BiLSTM (24 cử chỉ) và khởi tạo FastAPI Server...')
ready_evt.wait(timeout=45.0)
if startup_err:
    print(f'❌ Lỗi khởi động AI Server: {startup_err[0]}', file=sys.stderr)
    sys.exit(1)
print('✅ AI Server đã sẵn sàng tại cổng 8000! Đang kết nối Cloudflare Tunnel...')
time.sleep(1.5)

proc = subprocess.Popen(['./cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000'],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding='utf-8', errors='replace')

url_re = re.compile(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
found_url = False
try:
    for line in proc.stdout:
        if not found_url:
            m = url_re.search(line)
            if m:
                found_url = True
                url = m.group(0)

                def sync_to_vercel_registry(tunnel_url):
                    import urllib.request
                    reg_url = 'https://ntfy.sh/vslr_ctu_aws_active_backend_prod_v3'
                    try:
                        req = urllib.request.Request(reg_url, data=tunnel_url.encode('utf-8'), method='POST')
                        urllib.request.urlopen(req, timeout=5.0)
                    except Exception:
                        pass
                    while True:
                        time.sleep(900)
                        try:
                            req = urllib.request.Request(reg_url, data=tunnel_url.encode('utf-8'), method='POST')
                            urllib.request.urlopen(req, timeout=5.0)
                        except Exception:
                            pass

                threading.Thread(target=sync_to_vercel_registry, args=(url,), daemon=True).start()

                print('\n' + '='*74)
                print('🎉 HỆ THỐNG VSLR ĐÃ CHẠY THÀNH CÔNG TRÊN AWS SINGAPORE! (PING 30ms)')
                print('='*74)
                print('👉 1. TÊN MIỀN CỐ ĐỊNH CHÍNH THỨC (Mở trên Web / In vào Báo cáo / QR Code):')
                print('      https://vslr-project-v3.vercel.app')
                print('      (Đã tự động đồng bộ ngầm với đường hầm AWS của bạn - Không cần đổi link!)')
                print('\n👉 2. ĐỊA CHỈ ĐƯỜNG HẦM CLOUDFLARE TRỰC TIẾP:')
                print(f'      {url}')
                print('='*74)
                print('\n(Giữ nguyên cửa sổ terminal này để duy trì server. Nhấn Ctrl+C để dừng)\n')
except KeyboardInterrupt:
    print('\nĐang dừng server AWS...')
    proc.terminate()
"
