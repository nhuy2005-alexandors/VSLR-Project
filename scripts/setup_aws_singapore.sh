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

# Cấu hình Token lưu dữ liệu Hugging Face Dataset & Google Drive Webhook
export HF_TOKEN="${HF_TOKEN:-}"
export HF_DATASET_REPO="${HF_DATASET_REPO:-ntbii305/vslr-remote}"
export GDRIVE_WEBHOOK_URL="${GDRIVE_WEBHOOK_URL:-}"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 5. Thiết lập dịch vụ chạy nền liên tục 24/7 (Systemd Service)
echo ""
echo "[4/4] Đang kích hoạt dịch vụ chạy ngầm 24/7 (Tự khởi động & Tự phục hồi kết nối)..."

pkill -f "cloudflared tunnel" >/dev/null 2>&1 || true
pkill -f "run_aws_daemon.py" >/dev/null 2>&1 || true
rm -f active_tunnel_url.txt

sudo tee /etc/systemd/system/vslr.service >/dev/null <<EOF
[Unit]
Description=VSLR AI Server & Cloudflare Tunnel 24/7 Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PWD
Environment="PATH=$PWD/venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$PWD/src"
Environment="HF_TOKEN=${HF_TOKEN:-}"
Environment="HF_DATASET_REPO=${HF_DATASET_REPO:-ntbii305/vslr-remote}"
Environment="GDRIVE_WEBHOOK_URL=${GDRIVE_WEBHOOK_URL:-}"
ExecStart=$PWD/venv/bin/python $PWD/scripts/run_aws_daemon.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable vslr.service >/dev/null 2>&1 || true
sudo systemctl restart vslr.service

echo "⏳ Đang nạp mô hình AI BiLSTM (24 cử chỉ) và mở đường hầm Cloudflare..."
for i in {1..30}; do
    if [ -s "active_tunnel_url.txt" ]; then
        break
    fi
    sleep 1
done

TUNNEL_URL=$(cat active_tunnel_url.txt 2>/dev/null || echo "Đang khởi tạo...")

echo ""
echo "=========================================================================="
echo "🎉 HỆ THỐNG VSLR ĐÃ KÍCH HOẠT CHẾ ĐỘ CHẠY LIÊN TỤC 24/7 TRÊN AWS!"
echo "=========================================================================="
echo "👉 1. TÊN MIỀN CỐ ĐỊNH CHÍNH THỨC (Mở trên Web / In vào Báo cáo / QR Code):"
echo "      https://vslr-project-v3.vercel.app"
echo "      (Tự động đồng bộ ngầm với đường hầm AWS - Không bao giờ phải đổi link!)"
echo ""
echo "👉 2. ĐỊA CHỈ ĐƯỜNG HẦM CLOUDFLARE HIỆN TẠI:"
echo "      ${TUNNEL_URL}"
echo "=========================================================================="
echo "✅ TRẠNG THÁI CHẠY NGẦM 24/7 ĐÃ BẬT:"
echo "   • Bạn có thể ĐÓNG CỬA SỔ TERMINAL NÀY và TẮT MÁY TÍNH ngay bây giờ!"
echo "   • Server sẽ tiếp tục chạy liên tục suốt ngày đêm và tự động phục hồi nếu rớt mạng."
echo "   • Xem log trực tiếp bất kỳ lúc nào: sudo journalctl -u vslr -f"
echo "   • Lệnh tạm dừng server:              sudo systemctl stop vslr"
echo "=========================================================================="
