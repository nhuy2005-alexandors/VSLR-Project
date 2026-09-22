# BÁO CÁO KIỂM TOÁN MÁY TỰ ĐỘNG (V3 MACHINE AUDIT REPORT)
## ĐỢT HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH VSLR V3 (4 NGƯỜI KÝ P01–P04 & KIỂM ĐỊNH P05) — 24 CỬ CHỈ

* **Dự án**: Nghiên cứu Khoa học — Nhận diện Cử chỉ Ngôn ngữ Ký hiệu Tiếng Việt (VSLR)
* **Tính chất**: Kiểm toán máy tự động (Machine Audit) xác minh tính toàn vẹn số liệu, mã băm SHA-256 và chữ ký số 3 chiều (Three-Way Binding).
* **Phiên bản mô hình**: VSLR Candidate V3 13-epoch (`runs/v3-ship-4signers-24-8clips-13ep-20260916-020145`)
* **Thời gian kiểm toán**: Tháng 09/2026

---

## 1. TỔNG HỢP KẾT QUẢ KIỂM TOÁN TỰ ĐỘNG

| Hạng mục kiểm tra | Tiêu chuẩn đặt ra | Kết quả thực tế V3 | Kết luận |
|---|---|:---:|:---:|
| **Khóa Dataset (Phase 1)** | 768 video P01–P04, 768 SHA-256 phân biệt 100%, $\le 10$s, Full HD | 768 file path & 768 hash độc nhất trong manifest V3 | **PASS** |
| **Cổng nghiệm thu dữ liệu** | `vslr-check` exit code 0 | 768/768 clips đạt chuẩn, đủ độ phủ 4 người × 24 nhãn × 8 clips | **PASS** |
| **Trích xuất đặc trưng sạch** | Extractor 203D (`presence-aware group-local-z`), sequence 60 | Khóa phiên bản `features_version = 3`, 0 rò rỉ landmark | **PASS** |
| **Đánh giá LOSO 4-Folds** | 4 folds độc lập, 13 epochs, GPU/CPU, kiểm định unaugmented test | 768 lượt dự đoán, 0 rò rỉ giữa train và val fold | **PASS** |
| **Pooled Accuracy (LOSO)** | Tỉ lệ đoán đúng trên 768 clips test thực tế | **98.57%** (**757 / 768 clips**) | **PASS** |
| **Top-3 Accuracy (LOSO)** | Tỉ lệ nhãn đúng nằm trong Top 3 dự đoán | **99.35%** (**763 / 768 clips**) | **PASS** |
| **Fold thấp nhất (Worst Fold)** | Ghi nhận trung thực theo số liệu | Fold **P03: 97.40%** (187 / 192 clips) | **PASS** |
| **Fold cao nhất (Best Fold)** | Ghi nhận trung thực theo số liệu | Fold **P01: 100.00%** (192 / 192 clips) | **PASS** |
| **Huấn luyện Ship Candidate** | Huấn luyện toàn bộ 768 clips với tham số đóng băng | Checkpoint `gesture_lstm.pt` xuất thành công | **PASS** |
| **Khóa Ba Chiều (Binding)** | `training_signature` khớp giữa LOSO report, metrics và weights | Chữ ký `918588dedeb...` khớp 100% 3 bên, SHA-256 khớp | **PAIR OK** |
| **Đánh giá Ngoài Unseen P05** | 48 clips độc lập từ người ký mới P05, zero leakage | **100.00%** Top-1 Accuracy (48/48), 0 lỗi | **PASS** |
| **Bộ kiểm thử phần mềm** | `pytest` kiểm thử toàn bộ module chức năng | **181 passed**, 1 skipped (100% core pass) | **PASS** |

---

## 2. KIỂM ĐỊNH TÍNH TOÀN VẸN VÀ CHỐNG RÒ RỈ DỮ LIỆU (ZERO DATA LEAKAGE)

### 2.1. Căn cứ Khóa Ba Chiều (Three-Way Binding Invariant)
Mô hình V3 áp dụng cơ chế khóa liên kết 3 bên để ngăn chặn việc tráo đổi trọng số hoặc công bố sai lệch chỉ số:
1. **Trọng số Checkpoint (`artifacts/v3-realtime-test-candidate/gesture_lstm.pt`)**:
   - SHA-256: `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`
2. **Chữ ký huấn luyện (Training Signature)**:
   - Signature: `918588dedeb2e9bdb2dabf8a2f8ac843aa6cf66dc68d61c2e8acb1e4b07a65cf`
   - Hash bao hàm toàn bộ siêu tham số (`epochs=13`, `lr=0.001`, `augment=120`, `hidden_size=96`), danh sách 24 nhãn NFC và toàn bộ 768 mã băm SHA-256 của tập dữ liệu huấn luyện.
3. **Run Manifest (`RUN_MANIFEST.json`)**:
   - Đóng dấu thời gian, git commit, branch và đối soát chữ ký số khớp 100% với file weights.

### 2.2. Kiểm định Độc lập trên Người ký Ngoại cảnh P05
* **Quy tắc tiêu thụ (`P05_CONSUMED = true`)**:
  - Tập dữ liệu P05 đã tiêu thụ hoàn toàn cho một lần đánh giá độc lập duy nhất.
  - Vĩnh viễn không được đưa vào tập huấn luyện hay tái sử dụng cho bất kỳ vòng lựa chọn mô hình nào.
* **Đối chiếu mã băm (Zero Leakage Check)**:
  - Toàn bộ 48 video của P05 được kiểm tra chéo từng byte với 768 video của P01–P04. Kết quả: **0% trùng lặp**, xác nhận điều kiện thử nghiệm mù hoàn toàn độc lập.
* **Kết quả đo lường ngoại cảnh**:
  - 48 / 48 clips nhận diện chính xác ở vị trí Top-1 (100.00%).
  - Biên độ cách biệt xác suất (Confidence Margin) giữa Top-1 và Top-2 đạt trung bình 0.9615, chứng minh mô hình tự tin và phân biệt rõ ràng giữa các cử chỉ.

---

## 3. THÔNG SỐ VẬN HÀNH VÀ TRIỂN KHAI HỆ THỐNG

* **Model File**: `artifacts/v3-realtime-test-candidate/gesture_lstm.pt` (và đồng bộ tại `models/gesture_lstm.pt`).
* **Kích thước file trọng số**: **987,668 bytes** (~964 KB).
* **Thời gian nạp mô hình vào RAM**: $< 0.2$ giây.
* **Bộ phát âm tiếng Việt (TTS Integration)**:
  - Tự động gọi VieNeu-TTS (mô hình neural voice Trúc Ly 48kHz).
  - Tự động chuyển đổi mượt mà sang `pyttsx3` (Microsoft SAPI5 offline) nếu máy không có kết nối internet tải mô hình.
* **Giao diện Web Studio**:
  - Cổng phục vụ: `http://localhost:8000` (FastAPI + HTML5 + CSS3 + Vanilla JS).
  - Băng thông video stream MJPEG: ổn định 25–30 FPS, độ trễ $< 50$ms trên mạng nội bộ localhost.
