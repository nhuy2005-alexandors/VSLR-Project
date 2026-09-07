# Spec — Đánh giá Tập Kiểm thử Bên ngoài (External Evaluation)

_Phiên bản: 1.0. Ngày: 2026-09-07. Trạng thái: Bản thảo kỹ thuật (Draft / Ready for TDD)._

## 1. Mục tiêu (Goals)

Xây dựng công cụ đánh giá độc lập (`vslr-eval`) để đo lường hiệu năng của mô hình nhận diện cử chỉ trên tập dữ liệu kiểm thử hoàn toàn mới (ví dụ: người ký độc lập P05, video ghi hình thực tế hoặc trích xuất từ camera), tuân thủ các nguyên tắc cốt lõi:
1. **Chỉ suy diễn (Inference Only)**: Tuyệt đối không huấn luyện lại, không cập nhật trọng số (`requires_grad = False`), không thay đổi mô hình checkpoint.
2. **Triệt tiêu Rò rỉ Dữ liệu (Zero Data Leakage)**: Kiểm tra mã băm SHA-256 của từng video kiểm thử đối chiếu với manifest huấn luyện (`--training-manifest`). Nếu phát hiện trùng lặp byte với bất kỳ clip huấn luyện nào, hệ thống lập tức dừng (fail-closed) và báo lỗi.
3. **Đồng nhất đường ống đặc trưng**: Dùng chung pipeline trích xuất (`HolisticExtractor`), chuẩn hóa toạ độ, xử lý presence và nội suy thời gian (`resample_sequence`) đồng nhất 100% với môi trường camera realtime và offline.
4. **Kỷ luật số liệu**:
   - Tách biệt rõ ràng giữa độ chính xác thô (top-1 raw accuracy) và độ chính xác sau bộ lọc tin cậy (accepted accuracy).
   - Đo lường tỉ lệ từ chối (rejection rate) và độ phủ (coverage).
   - Không coi ngưỡng thử nghiệm `confidence 0.50` là ngưỡng production.
   - Không gắn độ chính xác của tập test vào checkpoint như một thuộc tính nội tại của mô hình.

---

## 2. Định nghĩa Cây Dữ liệu Kiểm thử (`dataset/external_eval_v3/`)

Cấu trúc thư mục kiểm thử bên ngoài độc lập:

```text
dataset/external_eval_v3/
  P05/
    Bạn có cần giúp đỡ không/
      001.mov
      002.mov
    Bạn có vấn đề gì không/
      001.mov
      002.mov
    ...
    Xin lỗi/
      001.mov
      002.mov
```

### Quy mô dự kiến ban đầu (P05 Baseline)
- **Đối tượng**: 1 người mới độc lập (`P05`).
- **Số lượng nhãn**: Đủ 24 cử chỉ tương ứng với manifest `labels_v2_24.txt` và checkpoint `models/gesture_lstm.pt`.
- **Số clip mỗi cử chỉ**: 2 clip mỗi cử chỉ (đảm bảo tính lặp lại tối thiểu).
- **Tổng số clips**: 48 clips (1 người × 24 nhãn × 2 clips).
- **Ràng buộc toàn vẹn**: 100% 48 clips chưa từng xuất hiện trong tập huấn luyện của bất kỳ fold nào (P01–P04).

---

## 3. Giao diện Dòng lệnh (CLI Interface: `vslr-eval`)

### 3.1. Chế độ Thư mục (Directory Mode)

```powershell
vslr-eval `
  --data-dir dataset/external_eval_v3 `
  --model models/gesture_lstm.pt `
  --training-manifest runs/v2-4signers-24-20260907-005546/dataset_files_sha256.csv `
  --allow-uncalibrated `
  --confidence 0.50 `
  --output-dir evaluation/p05_baseline
```

### 3.2. Chế độ Một Clip đơn lẻ (Single Clip Mode)

Hỗ trợ kiểm tra nhanh một clip đã quay từ webcam:

```powershell
vslr-eval `
  --video "Xin chào=path/to/new_clip.mov" `
  --model models/gesture_lstm.pt `
  --training-manifest runs/v2-4signers-24-20260907-005546/dataset_files_sha256.csv `
  --allow-uncalibrated `
  --confidence 0.50
```

---

## 4. Đặc tả Kỹ thuật của `vslr-eval`

### 4.1. Cổng Kiểm soát Toàn vẹn (Integrity Gates)
1. **Model Integrity Gate**:
   - Nạp checkpoint bằng `load_checkpoint()`, kiểm tra `features_version == 3`, `feature_dim == 203`, `sequence_length == 60`.
   - Kiểm tra class order của mô hình: 24 nhãn chuẩn Unicode NFC.
   - Ghi nhận `checkpoint_sha256_before` và xác nhận lại `checkpoint_sha256_after == checkpoint_sha256_before` sau khi kết thúc đánh giá.
2. **Leakage Gate**:
   - Đọc danh sách SHA-256 từ tệp `--training-manifest` (nếu cung cấp).
   - Tính toán SHA-256 của từng clip kiểm thử. Nếu tồn tại bất kỳ clip nào trùng mã băm SHA-256 với tập train, dừng chương trình ngay lập tức với mã lỗi khác 0 và chỉ rõ file trùng lặp.
3. **Label Compatibility Gate**:
   - Tên thư mục cử chỉ trong `--data-dir` (hoặc nhãn trong `--video`) phải thuộc tập 24 nhãn của checkpoint sau khi chuẩn hoá Unicode NFC (`normalise_label`). Bất kỳ nhãn lạ nào đều kích hoạt fail-closed.
   - Hỗ trợ đầy đủ đường dẫn có ký tự tiếng Việt có dấu trên môi trường Windows.

### 4.2. Suy diễn & Đo lường Chỉ số (Metrics & Scoring)
1. **Suy diễn Top-K**:
   - Trả về nhãn dự đoán Top-1 và xác suất tin cậy (Softmax confidence).
   - Ghi nhận Top-3 dự đoán (danh sách gồm nhãn và xác suất) để phân tích lỗi tiệm cận.
2. **Chính sách chấp nhận/từ chối (Accept / Reject)**:
   - Nếu `confidence < threshold` (mặc định 0.50 cho demo/chẩn đoán): clip bị đánh dấu `accepted = False` (`status = REJECTED`).
   - Nếu `confidence >= threshold`: clip được chấp nhận `accepted = True` (`status = ACCEPTED`).
3. **Bộ chỉ số phân tách**:
   - **Top-1 Raw Accuracy**: Tỉ lệ đoán đúng trên toàn bộ tập clip được đánh giá: $\frac{N_{\text{correct}}}{N_{\text{total}}}$.
   - **Coverage**: Tỉ lệ clip được mô hình chấp nhận dự đoán: $\frac{N_{\text{accepted}}}{N_{\text{total}}}$.
   - **Rejection Rate**: Tỉ lệ clip bị từ chối do không đủ độ tin cậy: $1 - \text{Coverage}$.
   - **Accepted Accuracy**: Độ chính xác tính riêng trên các clip được chấp nhận: $\frac{N_{\text{correct\_and\_accepted}}}{N_{\text{accepted}}}$ (nếu $N_{\text{accepted}} = 0$, trả về 0.0).

### 4.3. Báo cáo Xuất ra (`--output-dir`)
1. `predictions.csv`:
   - Các cột: `video_path`, `signer`, `ground_truth`, `predicted_top1`, `confidence`, `top2_label`, `top2_conf`, `top3_label`, `top3_conf`, `accepted`, `correct`.
2. `metrics.json`:
   - Ghi nhận toàn bộ thông tin môi trường, model SHA-256, số liệu tổng thể, per-label accuracy, số clip rejected, tham số đánh giá.
3. `confusion_matrix.png`:
   - Ma trận nhầm lẫn kích thước 24 × 24 vẽ bằng matplotlib.
4. `REPORT.md`:
   - Báo cáo tổng kết dễ đọc, bảng per-class accuracy, danh sách các clip bị từ chối hoặc dự đoán sai.

---

## 5. Kế hoạch TDD (Test-Driven Development)

Tạo `tests/test_eval.py` bao phủ các trường hợp kiểm thử trước khi viết code triển khai:
1. `test_detects_hash_overlap_with_training_manifest`: Dừng ngay khi clip test trùng SHA-256 với train.
2. `test_rejects_unknown_label_not_in_checkpoint`: Từ chối nhãn không nằm trong danh sách 24 nhãn.
3. `test_model_hash_unchanged_after_evaluation`: Trọng số checkpoint trên đĩa không bị biến đổi 1 bit nào trước và sau khi chạy eval.
4. `test_metrics_calculation_top1_accepted_rejection_coverage`: Xác minh công thức toán học tính các chỉ số top-1, accepted accuracy, coverage và rejection rate.
5. `test_output_artifacts_generated_correctly`: Xác minh sinh đủ `predictions.csv`, `metrics.json`, `REPORT.md`.
6. `test_windows_unicode_paths`: Chạy mượt mà với đường dẫn thư mục tiếng Việt có dấu.
7. `test_single_video_mode`: Chạy đúng với cú pháp `--video "Nhãn=path"`.
