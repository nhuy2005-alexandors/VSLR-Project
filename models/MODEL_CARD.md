# Model Card — VSLR 24 Cử chỉ (BiLSTM Candidate)

## 1. Thông tin mô hình (Model Details)

- **Tên mô hình**: VSLR 24-gesture BiLSTM
- **Phiên bản**: `candidate-v2-4signers-24-20260907`
- **Kiến trúc**:
  - Feature Extractor: MediaPipe Holistic v3 (`features_version = 3`, `feature_contract = holistic-landmarks-v3-presence-aware-group-local-z`)
  - Kích thước vector đặc trưng: `feature_dim = 203` (67 landmarks × 3 + 2 presence channels L/R)
  - Độ dài chuỗi chuẩn hóa: `sequence_length = 60`
  - Mô hình học sâu: PyTorch `BiLSTM` (`hidden_size = 96`, `num_layers = 1`, `bidirectional = True`)
  - Pooling: `fwd-last + bwd-first` (`pool_sequence`)
  - Đầu ra phân loại: 24 lớp (Linear classifier head 192 -> 24)
- **Tập nhãn**: 24 cử chỉ tiếng Việt chuẩn Unicode NFC (khai báo tại `dataset/labels_v2_24.txt`, nhãn `Hẹn gặp lại` đã bị loại bỏ để triệt tiêu data leakage cross-signer).
- **Trọng số được triển khai (Ship Checkpoint)**: `models/gesture_lstm.pt` (kích thước ~987 KB).
- **Mã băm Checkpoint (SHA-256)**:
  `5e202eb9108c06e446da5ae1d27aaebcec7e14cb2e72a9c233c4ac99da245b83`
- **Chữ ký huấn luyện (Training Signature)**:
  `43316055f78bbf2e83c806396d5791d06faeb57b62bef13be17d77095a9c8675`
- **Khóa ba chiều (Three-Way Binding)**: Signature khớp 100% giữa `models/gesture_lstm.pt`, `models/metrics.json` và `models/loso_report.json`.

---

## 2. Dữ liệu huấn luyện & Kỷ luật số liệu (Training Data & Evaluation Disciplines)

- **Tập dữ liệu huấn luyện**: 576 clips (4 người ký P01, P02, P03, P04 × 24 cử chỉ × 6 clips/cặp). Toàn bộ 576 clips có mã băm SHA-256 độc nhất.
- **Lưu trữ video**: Dataset video gốc (867 MB) **không** được lưu trên Git repository để bảo vệ dung lượng repo. Thành viên trong nhóm cần tải dataset từ Google Drive nội bộ theo hướng dẫn của dự án.
- **Đánh giá LOSO (Leave-One-Signer-Out 4 folds)**:
  - **Macro Mean Accuracy**: **90.97%**
  - **Pooled Accuracy**: **90.97%** (524 / 576 clips đúng trên tập test unaugmented)
  - Fold P01: 140 / 144 (97.22%), Val Loss: 0.1274
  - Fold P02: 134 / 144 (93.06%), Val Loss: 0.3197
  - Fold P03: 138 / 144 (95.83%), Val Loss: 0.2642
  - Fold P04: 112 / 144 (77.78%), Val Loss: 0.9171 (fold thấp nhất)
- **Kỷ luật số liệu quan trọng**:
  - Con số **90.97%** là kết quả đánh giá kỹ thuật chéo (paired LOSO) của quy trình huấn luyện (training recipe), **không phải là test accuracy nội tại của checkpoint ship** `models/gesture_lstm.pt`.
  - Checkpoint ship được huấn luyện trên toàn bộ 576 clips của P01–P04 để phục vụ demo. **Tuyệt đối không được nạp lại 576 clips này để đo và công bố làm "test accuracy"** (vi phạm quy tắc leakage trong `docs/GOTCHAS.md`). Muốn báo cáo accuracy độc lập, bắt buộc phải quay dữ liệu kiểm thử từ người ký mới chưa từng tham gia huấn luyện.

---

## 3. Giới hạn & Cảnh báo an toàn (Limitations & Operational Warnings)

1. **Reject Policy chưa được Calibration**:
   - Hiện tại mô hình chưa có tập dữ liệu calibration riêng biệt cho các trường hợp cử chỉ nghỉ (`idle_stationary`) hoặc cử chỉ ngoài tập từ vựng (`oov_motion`).
   - Mặc định, `vslr-camera` và realtime engine sẽ từ chối chạy (fail-closed) và không phát âm TTS nếu thiếu calibration artifact.
   - **Chỉ sử dụng cờ `--allow-uncalibrated` cho mục đích chẩn đoán và chạy thử nghiệm (demo) tạm thời**, không sử dụng trong môi trường thực tế.
2. **Khả năng khái quát hóa**:
   - Mô hình **chưa được chứng minh** độ tin cậy đối với:
     - Người ký mới (người ngoài P01–P04).
     - Góc quay camera mới (khác góc thẳng trực diện).
     - Bối cảnh/ánh sáng phức tạp hoặc nền chuyển động có thể làm rung landmark của MediaPipe.
3. **Phân tích lỗi người ký P04**:
   - Người ký P04 đạt độ chính xác thấp nhất (77.78%), trong đó 2 nhãn `Bạn tên gì` và `Sao thế` bị nhầm 6/6 do P04 có hình thái ký lệch (thêm động tác vẫy tay mở đầu ở `Bạn tên gì` và tạo hình chữ OK đẩy tới ở `Sao thế`). Chi tiết phân tích tại `runs/v2-4signers-24-20260907-005546/visual_audit/REPORT.md`.

---

## 4. Liên kết Artifacts & Báo cáo kỹ thuật (Provenance)

- Run gốc đã niêm phong: `runs/v2-4signers-24-20260907-005546/`
- Báo cáo kỹ thuật chi tiết: `runs/v2-4signers-24-20260907-005546/REPORT_FOR_AGENT.md`
- Manifest chạy máy (Run Manifest): `runs/v2-4signers-24-20260907-005546/RUN_MANIFEST.json`
- Báo cáo kết quả đánh giá tự động: `runs/v2-4signers-24-20260907-005546/evaluation_report.md`
- Báo cáo audit trực quan: `runs/v2-4signers-24-20260907-005546/visual_audit/REPORT.md`
