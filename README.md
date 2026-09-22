# HỆ THỐNG NHẬN DIỆN CỬ CHỈ NGÔN NGỮ KÝ HIỆU TIẾNG VIỆT THỜI GIAN THỰC (VSLR)
**Đề tài Nghiên cứu Khoa học — Vietnamese Sign Language Recognition (VSLR)**

---

## 1. Giới thiệu Đề tài
Hệ thống nghiên cứu và xây dựng giải pháp hỗ trợ giao tiếp cho người khiếm thính tại Việt Nam thông qua việc nhận diện cử chỉ ngôn ngữ ký hiệu tiếng Việt thời gian thực (Real-time VSLR).

### Kiến trúc Kỹ thuật Lõi:
1. **Trích xuất đặc trưng không gian - thời gian:** Sử dụng **MediaPipe Holistic** (trích xuất đồng thời tư thế thân người Pose và 2 bàn tay Hands) với vector đặc trưng chuẩn hóa **203 chiều** (`holistic-landmarks-v3`).
2. **Mô hình học sâu (Deep Learning):** Mạng **BiLSTM (Bidirectional LSTM)** 2 lớp kết hợp cơ chế xử lý phân đoạn cử chỉ động (Adaptive Dynamic Segmentation Tracker).
3. **Bộ phát âm tiếng Việt (TTS):** Tự động chuyển đổi câu ký hiệu hoàn chỉnh thành giọng nói tự nhiên (VieNeu-TTS / Windows Speech Synthesis).
4. **Giao diện điều khiển (Web Studio):** Xây dựng trên nền tảng **FastAPI + SSE (Server-Sent Events)** hiển thị luồng video 30 FPS không độ trễ, đồ thị xác suất Top-3 và tự động lưu video mẫu.

---

## 2. Yêu cầu Hệ thống & Tính Tương thích
- **Hệ điều hành:** Windows 10 / Windows 11 (64-bit).
- **Python:** Phiên bản **Python 3.10** hoặc **3.11** (Lưu ý tích chọn *"Add Python to PATH"* khi cài đặt).
- **Phần cứng:** 
  - Tối ưu hóa chạy mượt mà trực tiếp trên **CPU** của laptop văn phòng thông thường (~25–30 FPS).
  - Tự động kích hoạt tăng tốc **NVIDIA GPU (CUDA)** nếu máy có trang bị card đồ họa rời.
  - Thiết bị đầu vào: **Webcam** tích hợp sẵn trên laptop và **Loa/Tai nghe**.

---

## 3. Hướng dẫn Khởi động Nhanh (1-Click Setup)

### Bước 1: Cài đặt môi trường tự động (Chỉ cần chạy 1 lần đầu)
- Nhấp đúp chuột vào file **`CAI_DAT_MOI_TRUONG.bat`**.
- Hệ thống sẽ tự động tạo môi trường ảo độc lập (`venv`) và cài đặt các thư viện cần thiết. Quá trình mất khoảng 2–4 phút.

### Bước 2: Khởi chạy và trải nghiệm hệ thống
Quý Thầy/Cô có thể lựa chọn một trong hai phương thức trải nghiệm sau:

| Phương thức | Tệp khởi chạy | Mô tả chi tiết |
|---|---|---|
| **Giao diện Web (Khuyến nghị)** | **`CHAY_WEB.bat`** | Hệ thống tự động mở trình duyệt tại `http://localhost:8000`. Cung cấp giao diện đồ họa trực quan, hiển thị khung xương thời gian thực, bảng xác suất nhận diện, nút phát âm lại và quản lý video đã ghi. |
| **Cửa sổ Camera trực tiếp** | **`CHAY_WEBCAM.bat`** | Mở trực tiếp cửa sổ luồng camera OpenCV trên màn hình. Nhẹ, khởi động tức thì. |

> **Các phím tắt tiện ích khi sử dụng:**
> - `[S]`: Buộc đọc phát âm ngay câu hiện tại (Speak Now).
> - `[C]`: Xóa câu hiện tại và làm mới bộ nhớ đệm (Clear Sentence).
> - `[SPACE]`: Chốt mốc phân đoạn cử chỉ thủ công (Force Boundary).
> - `[Q]` hoặc `[ESC]`: Thoát ứng dụng.

---

## 4. Danh mục 24 Cử chỉ Ngôn ngữ Ký hiệu Hệ thống Nhận diện

Hệ thống được huấn luyện và đánh giá trên bộ từ vựng 24 câu giao tiếp thiết yếu thường ngày:

| STT | Nhãn Cử chỉ | Ý nghĩa ngữ cảnh |
|:---:|---|---|
| 1 | **Xin chào** | Chào hỏi ban đầu |
| 2 | **Tạm biệt** | Chào tạm biệt |
| 3 | **Cảm ơn** | Bày tỏ lòng biết ơn |
| 4 | **Xin lỗi** | Bày tỏ sự xin lỗi |
| 5 | **Bạn tên gì** | Hỏi thông tin cá nhân |
| 6 | **Bạn quê ở đâu** | Hỏi quê quán |
| 7 | **Mấy tuổi** | Hỏi tuổi tác |
| 8 | **Bạn đang làm gì** | Hỏi hoạt động hiện tại |
| 9 | **Hôm nay bạn khỏe không** | Hỏi thăm sức khỏe |
| 10 | **Tôi khỏe** | Trả lời sức khỏe tốt |
| 11 | **Tôi bình thường** | Trả lời sức khỏe ổn định |
| 12 | **Tôi không khỏe** | Báo tình trạng không khỏe |
| 13 | **Bạn có cần giúp đỡ không** | Đề nghị hỗ trợ |
| 14 | **Bạn có vấn đề gì không** | Thăm dò khó khăn |
| 15 | **Gọi xe cứu thương** | Tình huống khẩn cấp y tế |
| 16 | **Rất vui được gặp bạn** | Giao tiếp xã giao |
| 17 | **Lâu rồi không gặp** | Gặp lại người quen |
| 18 | **Về nhà cẩn thận** | Lời dặn dò an toàn |
| 19 | **Đi đâu** | Hỏi phương hướng/địa điểm |
| 20 | **Siêu thị** | Địa điểm mua sắm |
| 21 | **Sao thế** | Hỏi han tình hình |
| 22 | **Chuyện gì** | Thắc mắc sự việc |
| 23 | **Như thế nào** | Hỏi cách thức |
| 24 | **Được không** | Xin ý kiến / xác nhận |

---

## 5. Cấu trúc Thư mục Dự án

```text
VSLR-Project/
├── artifacts/                      # Trọng số mô hình Candidate V3 24 lớp tốt nhất
│   └── v3-realtime-test-candidate/
│       ├── gesture_lstm.pt         # Trọng số mô hình BiLSTM tối ưu
│       └── labels_v2_24.txt        # Danh sách 24 nhãn cử chỉ chuẩn Unicode NFC
├── models/                         # Hồ sơ siêu tham số và báo cáo đánh giá khoa học
│   ├── labels.json                 # Định dạng JSON 24 nhãn
│   └── loso_report.json            # Kết quả kiểm nghiệm Leave-One-Signer-Out
├── src/                            # Mã nguồn kiến trúc hệ thống
│   └── prototype_3_gestures/
│       ├── vsl3/                   # Trích xuất đặc trưng 203 chiều & mô hình BiLSTM
│       ├── realtime.py             # Pipeline nhận diện camera và suy luận thời gian thực
│       ├── web_server.py           # Backend FastAPI phục vụ giao diện Web Studio
│       ├── tts.py                  # Module phát âm giọng đọc tiếng Việt
│       └── recorder.py             # Bộ lưu trữ và quản lý clip cử chỉ
├── vslr-web/                       # Giao diện người dùng Web Studio (HTML/CSS/JavaScript)
├── tests/                          # 182 bài kiểm thử đơn vị tự động (Unit Tests)
├── docs/                           # Tài liệu kỹ thuật và báo cáo nghiệm thu
├── CAI_DAT_MOI_TRUONG.bat          # Kịch bản cài đặt tự động 1 chạm
├── CHAY_WEB.bat                    # Kịch bản khởi chạy giao diện Web Studio
├── CHAY_WEBCAM.bat                 # Kịch bản khởi chạy camera trực tiếp
├── requirements.txt                # Danh mục các thư viện phụ thuộc
└── pyproject.toml                  # Cấu hình gói và metadata chuẩn PEP 517/621
```

---

## 6. Đánh giá Thực nghiệm (LOSO & Unseen Signer Evaluation)
Hệ thống được kiểm định độ tin cậy thông qua giao thức đánh giá nghiêm ngặt **Leave-One-Signer-Out (LOSO)** trên 4 người ký (`P01`, `P02`, `P03`, `P04`) trên 768 clips và kiểm định ngoại cảnh độc lập trên người ký thứ năm (**P05**):
- **Độ chính xác LOSO (4 Folds gộp)**: **98.57%** (757 / 768 clips đúng).
- **Top-3 Accuracy**: **99.35%** (763 / 768 clips).
- **Đánh giá ngoài Unseen Signer P05**: **100.00%** (48 / 48 clips đúng).

![Biểu đồ đánh giá LOSO và P05](docs/reports/figures/01_loso_performance_by_fold.png)

![Tiến trình cải tiến V1 sang V3](docs/reports/figures/02_evolution_v1_v2_v3.png)

- Chi tiết báo cáo ma trận nhầm lẫn, biểu đồ hội tụ và độ chính xác từng lớp được ghi nhận đầy đủ tại [docs/reports/V3_TECHNICAL_EVALUATION_REPORT.md](docs/reports/V3_TECHNICAL_EVALUATION_REPORT.md).
