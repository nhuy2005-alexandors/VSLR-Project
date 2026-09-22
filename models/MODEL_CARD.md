# Model Card — VSLR 24 Cử chỉ (BiLSTM Candidate V3)

## 1. Thông tin mô hình (Model Details)

- **Tên mô hình**: VSLR 24-gesture BiLSTM (Candidate V3 13-epoch)
- **Phiên bản**: `v3-ship-4signers-24-8clips-13ep-20260916`
- **Kiến trúc mạng & Trích xuất đặc trưng**:
  - Feature Extractor: MediaPipe Holistic v3 (`features_version = 3`, `feature_contract = holistic-landmarks-v3-presence-aware-group-local-z`)
  - Kích thước vector đặc trưng: `feature_dim = 203` (67 landmarks × 3 tọa độ không gian + 2 kênh hiện diện bàn tay L/R)
  - Độ dài chuỗi chuẩn hóa: `sequence_length = 60`
  - Mô hình học sâu: PyTorch `BiLSTM` (`hidden_size = 96`, `num_layers = 1`, `bidirectional = True`)
  - Cơ chế Pooling: `fwd-last + bwd-first` (`pool_sequence` 192D)
  - Đầu ra phân loại: 24 lớp (Linear classifier head 192 -> 24)
- **Tập nhãn**: 24 cử chỉ tiếng Việt chuẩn Unicode NFC (khai báo tại `dataset/labels_v2_24.txt`).
- **Trọng số triển khai**:
  - `artifacts/v3-realtime-test-candidate/gesture_lstm.pt` (và `models/gesture_lstm.pt`)
  - Kích thước: ~987 KB (987,668 bytes)
- **Mã băm Checkpoint (SHA-256)**:
  `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`
- **Chữ ký huấn luyện (Training Signature)**:
  `918588dedeb2e9bdb2dabf8a2f8ac843aa6cf66dc68d61c2e8acb1e4b07a65cf`
- **Khóa ba chiều (Three-Way Binding)**: Signature khớp 100% giữa weights, `RUN_MANIFEST.json` và log huấn luyện.

---

## 2. Dữ liệu huấn luyện & Đánh giá thực nghiệm

- **Tập dữ liệu huấn luyện**: 768 clips (4 người ký P01, P02, P03, P04 × 24 cử chỉ × 8 clips/cặp). Toàn bộ 768 clips có mã băm SHA-256 độc nhất.
- **Đánh giá kiểm định chéo LOSO (Leave-One-Signer-Out 4 Folds)**:
  - **Macro Mean Accuracy**: **98.57%**
  - **Pooled Accuracy**: **98.57%** (757 / 768 clips đoán đúng trên tập test unaugmented)
  - **Top-3 Accuracy**: **99.35%** (763 / 768 clips)
  - Fold P01: 192 / 192 (**100.00%**), Val Loss: 0.0261
  - Fold P02: 189 / 192 (**98.44%**), Val Loss: 0.1544
  - Fold P03: 187 / 192 (**97.40%**), Val Loss: 0.1433 (worst fold)
  - Fold P04: 189 / 192 (**98.44%**), Val Loss: 0.1127
- **Đánh giá ngoài độc lập trên người ký mới (Unseen Signer P05 External Evaluation)**:
  - Quy mô: 48 clips (24 nhãn × 2 clips từ người ký P05 độc lập hoàn toàn với tập train).
  - **Raw Top-1 Accuracy**: **100.00%** (48 / 48 clips nhận diện đúng).
  - **Top-3 Accuracy**: **100.00%** (48 / 48 clips).
  - **Rejection Rate**: 0.00% (ngưỡng tin cậy 0.50).
  - **Accepted Accuracy**: **100.00%**.
  - Mean Confidence: 0.9669.

---

## 3. Khả năng vận hành thời gian thực (Real-time Deployment)

1. **Hiệu năng CPU**:
   - Kiến trúc nhẹ (~200k tham số), MediaPipe Holistic C++ Backend đạt **25–30 FPS** ổn định trên CPU máy tính văn phòng thông thường.
   - Độ trễ suy luận mô hình: $< 1.5$ ms / frame.
2. **Tích hợp Web Studio & TTS**:
   - Phục vụ qua FastAPI Web Server tại `http://localhost:8000`.
   - Tự động chuyển đổi văn bản thành giọng nói tiếng Việt (VieNeu-TTS / Microsoft SAPI5).
