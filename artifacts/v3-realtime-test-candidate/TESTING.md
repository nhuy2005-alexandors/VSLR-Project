# Hướng dẫn Kiểm thử Thời gian thực Ngoài Thực địa — V3 Candidate (TESTING.md)

## 1. Trạng thái Bản phát hành (Release Status)

- **Trạng thái**: **EXTERNAL TEST CANDIDATE (Ứng viên Kiểm thử Bên ngoài)**
- **Cảnh báo**: **NOT PRODUCTION READY (Chưa sẵn sàng cho môi trường sản xuất)**
- **Phiên bản mô hình**: **V3 13-epoch frozen candidate**
- **Nguồn gốc Checkpoint**: `runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/gesture_lstm.pt`
- **Mã băm SHA-256**: `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`
- **Ký hiệu huấn luyện (training_signature)**: `918588dedeb2e9bdb2dabf8a2f8ac843aa6cf66dc68d61c2e8acb1e4b07a65cf`
- **Số lớp**: 24 cử chỉ tiếng Việt chuẩn NFC ([labels_v2_24.txt](labels_v2_24.txt))

---

## 2. Kỷ luật Dữ liệu & Quy tắc Kiểm thử Bắt buộc

1. **P05 Đã Tiêu thụ Hoàn toàn (`P05_CONSUMED = true`)**:
   - Tập dữ liệu người ký P05 đã được sử dụng độc quyền cho phiên đánh giá ngoài offline (kết quả: 48/48 Top-1 đúng, 100%).
   - P05 **tuyệt đối không được coi là unseen** hay sử dụng lại để hiệu chuẩn ngưỡng từ chối hoặc lựa chọn mô hình.
2. **Quy ước Người ký Mới (New Testers)**:
   - Toàn bộ người tham gia thử nghiệm realtime phải được gán mã định danh ẩn danh: `R01`, `R02`, `R03`, ...
3. **Cấm Huấn luyện Lại (No Retraining)**:
   - Các mẫu video hoặc dữ liệu thu thập từ người thử nghiệm (`R01`, `R02`, ...) **tuyệt đối KHÔNG được bổ sung vào tập huấn luyện** trong suốt chu kỳ nghiệm thu này.
4. **Nghiệm thu Realtime & OOD Chưa Hoàn tất (Pending Acceptance)**:
   - Việc thử nghiệm thời gian thực, đánh giá trên phân phối ngoài từ điển (OOD/Neutral/Nhiễu) và hiệu chuẩn chính sách từ chối (`RejectPolicy`) vẫn đang ở trạng thái **`PENDING / HOLD`**.
   - Model sản xuất chính thức tại `models/gesture_lstm.pt` (SHA-256 `5e202eb9108c...`) giữ nguyên tuyệt đối và chưa bị thay thế.

---

## 3. Lệnh Khởi chạy Realtime Chính thức

Mô hình V3 sử dụng bộ trích xuất đặc trưng 203 chiều (`holistic-landmarks-v3-presence-aware-group-local-z`) và chính sách từ chối hiện ở trạng thái chưa hiệu chuẩn trên OOD (`uncalibrated`). Do đó, lệnh chạy thử nghiệm yêu cầu cờ `--allow-uncalibrated` và khuyến nghị ngưỡng chẩn đoán `--confidence 0.72`.

### 3.1. Chạy với TTS Phát âm Tiếng Việt (Mặc định)

```powershell
vslr-camera `
  --model artifacts/v3-realtime-test-candidate/gesture_lstm.pt `
  --allow-uncalibrated `
  --confidence 0.72 `
  --cooldown 1.5
```

Hoặc gọi trực tiếp qua Python module:

```powershell
python -m prototype_3_gestures.realtime `
  --model artifacts/v3-realtime-test-candidate/gesture_lstm.pt `
  --allow-uncalibrated `
  --confidence 0.72 `
  --cooldown 1.5
```

### 3.2. Chạy ở Chế độ Không dùng Loa (No TTS — Chỉ hiển thị văn bản)

```powershell
vslr-camera `
  --model artifacts/v3-realtime-test-candidate/gesture_lstm.pt `
  --allow-uncalibrated `
  --confidence 0.72 `
  --cooldown 1.5 `
  --no-tts
```

---

## 4. Phím Tắt Điều Khiển trên Giao diện Webcam

- **`Q`** hoặc **`ESC`**: Thoát chương trình camera.
- **`C`**: Xóa câu đang tích lũy (`sentence`) và xóa bộ đệm cử chỉ lặp.
- **`S`**: Buộc phát âm ngay câu hiện tại (Speak Now) mà không cần chờ hết khoảng lặng `sentence-gap`.
- **`SPACE`**: Buộc đóng phân đoạn cử chỉ hiện tại (Force Gesture Boundary).

---

## 5. Danh sách Tệp trong Gói Phân phối

1. `gesture_lstm.pt`: Trọng số mô hình BiLSTM Candidate V3 13-epoch (SHA-256: `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`).
2. `labels_v2_24.txt`: Danh sách 24 nhãn cử chỉ chuẩn Unicode NFC.
3. `RUN_MANIFEST.json`: Hồ sơ nguồn gốc huấn luyện, siêu tham số và chữ ký mô hình.
4. `TESTING.md`: Tài liệu hướng dẫn và ràng buộc kiểm thử.
5. `SHA256SUMS.txt`: Bảng mã băm SHA-256 niêm phong toàn bộ tệp trong gói.
