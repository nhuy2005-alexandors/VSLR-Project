# Spec — Đánh giá Tập Kiểm thử Bên ngoài (External Evaluation)

_Phiên bản: 2.0. Ngày: 2026-09-08. Trạng thái: Hoàn thành và Kiểm định (Implemented & Verified)._

## 1. Mục tiêu (Goals)

Xây dựng công cụ đánh giá độc lập (`vslr-eval`) để đo lường hiệu năng của mô hình nhận diện cử chỉ trên tập dữ liệu kiểm thử hoàn toàn mới (ví dụ: người ký độc lập P05, video ghi hình thực tế hoặc trích xuất từ camera), tuân thủ các nguyên tắc cốt lõi:
1. **Chỉ suy diễn (Inference Only)**: Tuyệt đối không huấn luyện lại, không cập nhật trọng số (`requires_grad = False`), sử dụng `torch.inference_mode()`, không thay đổi mô hình checkpoint.
2. **Triệt tiêu Rò rỉ Dữ liệu (Zero Data Leakage)**: Kiểm tra mã băm SHA-256 của từng video kiểm thử đối chiếu với manifest huấn luyện (`--training-manifest`). Chỉ được khẳng định Zero Leakage khi toàn bộ clip kiểm thử đã vượt qua cổng đối chiếu này. Nếu phát hiện trùng lặp byte với bất kỳ clip huấn luyện nào, hệ thống lập tức dừng (fail-closed) và báo lỗi.
3. **Đồng nhất đường ống đặc trưng**: Dùng chung pipeline trích xuất (`HolisticExtractor`), chuẩn hóa toạ độ, xử lý presence và nội suy thời gian (`resample_sequence`) đồng nhất 100% với môi trường camera realtime và offline.
4. **Kỷ luật số liệu**:
   - Tách biệt rõ ràng giữa độ chính xác thô (top-1 raw accuracy) và độ chính xác sau bộ lọc tin cậy (accepted accuracy).
   - Đo lường tỉ lệ từ chối (rejection rate) và độ phủ (coverage).
   - Ngưỡng thử nghiệm `confidence 0.50` chỉ là ngưỡng chẩn đoán tạm thời, **tuyệt đối không phải ngưỡng production** (ngưỡng khuyến nghị khi có policy là $\ge 0.72$).
   - Không gắn độ chính xác của tập test vào checkpoint như một thuộc tính nội tại của mô hình.
   - Hiện tại P05 chưa quay video thật, do đó **chưa có số liệu accuracy kiểm thử thực tế của P05**.

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
- **Số lượng nhãn**: Đủ đúng 24 cử chỉ tương ứng với manifest `labels_v2_24.txt` và checkpoint `models/gesture_lstm.pt` sau khi chuẩn hóa Unicode NFC.
- **Số clip mỗi cử chỉ**: Đúng 2 clip mỗi cử chỉ (`--clips-per-label 2`).
- **Tổng số clips**: 48 clips (1 người × 24 nhãn × 2 clips).
- **Ràng buộc toàn vẹn**: 100% 48 clips có SHA-256 độc nhất và chưa từng xuất hiện trong tập huấn luyện của bất kỳ fold nào (P01–P04).

---

## 3. Giao diện Dòng lệnh (CLI Interface: `vslr-eval`)

### 3.1. Chế độ Thư mục (Directory Mode)

```powershell
vslr-eval `
  --data-dir dataset/external_eval_v3 `
  --model models/gesture_lstm.pt `
  --training-manifest runs/v2-4signers-24-20260907-005546/dataset_files_sha256.csv `
  --run-manifest runs/v2-4signers-24-20260907-005546/RUN_MANIFEST.json `
  --expected-signer P05 `
  --clips-per-label 2 `
  --allow-uncalibrated `
  --confidence 0.50 `
  --output-dir evaluation/p05_baseline
```

### 3.2. Chế độ Một Clip đơn lẻ (Single Clip Mode)

Hỗ trợ kiểm tra nhanh một clip đã quay từ webcam:

```powershell
vslr-eval `
  --video "Xin chào=path/to/webcam_clip.mov" `
  --model models/gesture_lstm.pt `
  --training-manifest runs/v2-4signers-24-20260907-005546/dataset_files_sha256.csv `
  --run-manifest runs/v2-4signers-24-20260907-005546/RUN_MANIFEST.json `
  --allow-uncalibrated `
  --confidence 0.50
```

*Lưu ý*: Chế độ clip đơn lẻ không áp dụng cổng kiểm tra 48 clips của thư mục, nhưng vẫn bắt buộc kiểm tra `--training-manifest` và `--run-manifest` để chống rò rỉ dữ liệu và khoá chặt mô hình.

---

## 4. Các Cổng Kiểm soát Toàn vẹn (Integrity Gates)

1. **Run Manifest 3-Way Binding Gate (Bắt buộc)**:
   - Cả hai chế độ đều yêu cầu `--run-manifest` (fail-closed nếu thiếu).
   - Trước khi nạp MediaPipe hoặc suy diễn, đối soát chính xác 3 điều kiện:
     - `SHA-256(--training-manifest) == RUN_MANIFEST.json.dataset_manifest_sha256`
     - `SHA-256(model) == RUN_MANIFEST.json.checkpoint_sha256`
     - `checkpoint.training_signature == RUN_MANIFEST.json.training_signature`
   - Sai bất kỳ khoá nào: lập tức dừng (fail-closed), in `error: ...`, exit 2, không trích xuất đặc trưng, không inference, không ghi artifact.
2. **Output Safety Gate**:
   - Chặn `--output-dir` trùng hoặc nằm bên dưới `models/` (ngăn ô nhiễm checkpoint).
   - Chặn `--output-dir` trùng thư mục cha của checkpoint hoặc thư mục dữ liệu (`--data-dir`).
   - Nếu thư mục kết quả đã chứa artifacts cũ (`predictions.csv`, `metrics.json`), dừng chương trình trừ khi có cờ `--overwrite`.
3. **Directory Completeness Gate** (Chế độ Thư mục):
   - Bắt buộc khai báo ít nhất một người ký qua `--expected-signer` (ví dụ `--expected-signer P05`).
   - Tham số `--clips-per-label` phải là số nguyên dương $\ge 1$ (mặc định: 2).
   - Kiểm tra mô hình checkpoint có đúng 24 nhãn duy nhất sau chuẩn hóa Unicode NFC.
   - Mỗi người ký phải có đủ chính xác 24 thư mục cử chỉ, không thiếu và không thừa nhãn.
   - Mỗi thư mục cử chỉ phải chứa đúng số clip quy định (`--clips-per-label`).
   - Chạy toàn bộ trước khi nạp MediaPipe để tiết kiệm thời gian.
4. **Training Manifest Gate (Bắt buộc)**:
   - Cả hai chế độ đều yêu cầu `--training-manifest` (fail-closed nếu thiếu).
   - Mọi mã băm SHA-256 trong manifest phải hợp lệ: đúng 64 ký tự hex, lowercase, không rỗng, không trùng lặp.
   - Nếu bất kỳ clip kiểm thử nào trùng mã băm với manifest, dừng ngay lập tức: `DATA LEAKAGE DETECTED`.
5. **Test-Set Duplicate Gate**:
   - Tính mã băm SHA-256 của toàn bộ clip kiểm thử trước inference. Nếu phát hiện 2 clip kiểm thử trùng byte nhau, từ chối đánh giá.
6. **Unicode Normalization Gate**:
   - Chuẩn hóa toàn bộ nhãn checkpoint và thư mục sang Unicode NFC trước khi so sánh, bảo đảm tương thích NFC/NFD.
7. **Model Invariant Gate**:
   - `model.requires_grad_(False)`, chạy dưới `torch.inference_mode()`.
   - Kiểm tra mã băm SHA-256 của checkpoint trước và sau khi suy diễn trong khối `finally` (bảo đảm tính bất biến ngay cả khi có ngoại lệ).
8. **Evaluation Provenance Gate**:
   - `predictions.csv`: ghi nhận `video_path`, `video_sha256`, `model_sha256`, `signer`, `ground_truth`, `predicted_top1`, `confidence`, `top2`, `top3`, `accepted`, `correct`.
   - `metrics.json`: ghi nhận đầy đủ provenance gồm `num_signers`, `signers`, `label_count`, `clips_per_label`, `total_clips`, `test_set_fingerprint`, `training_manifest_path`, `training_manifest_sha256`, `run_manifest_path`, `run_manifest_sha256`, `model_sha256`, `training_signature`, `confidence_threshold`, `reject_policy_calibrated`.
   - `REPORT.md`: báo cáo tổng quan và chi tiết từng cử chỉ kèm provenance run manifest.
   - `confusion_matrix.png`: ma trận nhầm lẫn kích thước 24 × 24.
9. **CLI Error Handling Gate**:
   - Mọi vi phạm cổng hoặc lỗi đầu vào dự kiến được in ngắn gọn dạng `error: <thông điệp>` ra `stderr`, không kèm traceback, thoát với mã lỗi exit 2.
   - Các lỗi lập trình bất ngờ (`TypeError`, `AttributeError`, `KeyError`,...) không bị nuốt và hiển thị đầy đủ traceback để gỡ lỗi.
