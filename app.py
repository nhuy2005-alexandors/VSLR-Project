"""VSLR Realtime Sign Language Recognition - Hugging Face Spaces Backend.

Chạy hệ thống AI Backend trên Hugging Face Spaces (ZeroGPU Free).
Cung cấp API cho Frontend Vercel và tự động đồng bộ video vào Dataset ntbii305/vslr-remote.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Tắt chế độ Node SSR của Gradio 6 để tránh chiếm cổng 7860 của Uvicorn
os.environ["GRADIO_SSR_MODE"] = "False"

# Thêm thư mục src vào PYTHONPATH
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import gradio as gr
try:
    import spaces
except ImportError:
    spaces = None

from prototype_3_gestures.web_server import RealtimeVSLRPipeline, create_app


def _zerogpu_decorator(fn):
    if spaces is not None and hasattr(spaces, "GPU"):
        return spaces.GPU(fn)
    return fn


@_zerogpu_decorator
def verify_zerogpu_engine() -> str:
    return "🟢 ZeroGPU & VSLR BiLSTM Engine Active!"

# 1. Định vị checkpoint mô hình BiLSTM V3
model_candidates = [
    root_dir / "artifacts" / "v3-realtime-test-candidate" / "gesture_lstm.pt",
    root_dir / "models" / "gesture_lstm.pt",
    root_dir / "gesture_lstm.pt",
]
model_path = None
for candidate in model_candidates:
    if candidate.is_file():
        model_path = candidate
        break

if model_path is None:
    raise FileNotFoundError("Không tìm thấy checkpoint mô hình gesture_lstm.pt!")

# 2. Khởi tạo Pipeline VSLR Realtime
record_dir = root_dir / "videos"
web_dir = root_dir / "vslr-web"

pipeline = RealtimeVSLRPipeline(
    model_path=model_path,
    camera_id=0,
    confidence_threshold=0.70,
    cooldown=1.0,
    word_gap=0.28,
    sentence_gap=1.8,
    tts_voice="Trúc Ly",
    tts_engine="vieneu",
    record_dir=record_dir,
    allow_uncalibrated=True,
)

# 3. Khởi tạo FastAPI App (mount static files tại /web để root / phục vụ Gradio Dashboard)
api_app = create_app(pipeline, web_dir=web_dir, static_mount_path="/web" if web_dir.is_dir() else "")

# 4. Tạo giao diện Dashboard trên Gradio để Hugging Face Spaces nhận diện trạng thái Healthy
dataset_repo = os.environ.get("HF_DATASET_REPO", "ntbii305/vslr-remote")
has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
token_status = "✅ Đã cấu hình (Tự động đồng bộ video)" if has_token else "⚠️ Chưa đặt HF_TOKEN (Chỉ lưu tạm trên container)"

with gr.Blocks(title="VSLR Realtime Sign Language API") as demo:
    gr.Markdown(
        f"""
        # 🚀 VSLR Vietnamese Sign Language Recognition API
        ### Hệ thống Nhận diện Cử chỉ Ngôn ngữ Ký hiệu Việt Nam Thời gian thực
        **Đề tài Nghiên cứu Khoa học — Trường CNTT&TT, Đại học Cần Thơ (CTU)**

        ---
        - **Trạng thái:** 🟢 **API Server Online** (Hugging Face Spaces ZeroGPU)
        - **Kho lưu trữ Video Test:** [`{dataset_repo}`](https://huggingface.co/datasets/{dataset_repo})
        - **Bảo mật Hugging Face Token:** {token_status}
        - **Mô hình AI:** BiLSTM v3 (24 nhãn chuẩn VSLR) + MediaPipe Holistic C++

        ### 📡 Các cổng API phục vụ Frontend (Vercel):
        - `GET /api/status`: Trạng thái FPS, camera, số tay và người thử nghiệm
        - `GET /api/events`: Luồng Server-Sent Events (SSE) trả kết quả nhận diện và câu thoại
        - `POST /api/client_frame`: Tiếp nhận khung hình từ Webcam trình duyệt từ xa (0-lag)
        - `WebSocket /api/ws/client_feed`: Kết nối stream webcam tốc độ cao
        - `POST /api/signer?name=...`: Cập nhật tên người thử nghiệm
        - `GET /api/tts?text=...&play_server=false`: Sinh dữ liệu âm thanh giọng đọc tiếng Việt 48kHz (Trúc Ly)
        - `GET /web/`: Giao diện Web Studio chạy trực tiếp trên Space
        """
    )
    with gr.Row():
        status_btn = gr.Button("Kiểm tra kết nối ZeroGPU Engine", variant="primary")
        status_out = gr.Textbox(label="Trạng thái phần cứng", value="🟢 Sẵn sàng", interactive=False)
        status_btn.click(fn=verify_zerogpu_engine, inputs=[], outputs=[status_out])

from fastapi.middleware.cors import CORSMiddleware

# 5. Tích hợp seluruh Router của FastAPI vào Gradio App để tương thích 100% với ZeroGPU demo.launch()
custom_app = gr.routes.App()
custom_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
custom_app.include_router(api_app.router)

if __name__ == "__main__":
    pipeline.start()
    port = int(os.environ.get("PORT", 7860))
    print(f"[VSLR Space] Khởi động Gradio + FastAPI trên 0.0.0.0:{port}...")
    demo.launch(
        _app=custom_app,
        app_kwargs={"lifespan": api_app.router.lifespan_context},
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False,
        strict_cors=False,
    )
