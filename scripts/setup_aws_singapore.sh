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
echo "[1/4] Đang cài đặt thư viện hệ thống (Python 3, Git LFS, OpenCV, MediaPipe, Fonts)..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    git-lfs \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
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

# 3. Tạo môi trường ảo và cài đặt thư viện Python
echo ""
echo "[3/4] Đang tạo môi trường ảo Python và cài đặt PyTorch, MediaPipe, FastAPI..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

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
import subprocess, re, time, os, threading

def run_uvicorn():
    import uvicorn
    from prototype_3_gestures.web_server import RealtimeVSLRPipeline, create_app
    pipeline = RealtimeVSLRPipeline('models/gesture_lstm.pt', confidence_threshold=0.62)
    app = create_app(pipeline, web_dir='vslr-web')
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')

threading.Thread(target=run_uvicorn, daemon=True).start()
time.sleep(3.5)

proc = subprocess.Popen(['./cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000'],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding='utf-8', errors='replace')

url_re = re.compile(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
for line in proc.stdout:
    m = url_re.search(line)
    if m:
        url = m.group(0)
        vercel_link = f'https://vslr-project-v3.vercel.app/?backend={url}'
        print('\n' + '='*74)
        print('🎉 HỆ THỐNG VSLR ĐÃ CHẠY THÀNH CÔNG TRÊN AWS SINGAPORE! (PING 30ms)')
        print('='*74)
        print('👉 1. LINK TRỰC TIẾP CLOUDFLARE (Gửi bạn bè):')
        print(f'      {url}')
        print('\n👉 2. LINK QUA TÊN MIỀN VERCEL:')
        print(f'      {vercel_link}')
        print('='*74)
        print('\n(Giữ nguyên cửa sổ terminal này để duy trì server. Nhấn Ctrl+C để dừng)\n')
        break

try:
    proc.wait()
except KeyboardInterrupt:
    print('\nĐang dừng server AWS...')
    proc.terminate()
"
