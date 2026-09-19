# BÁO CÁO KIỂM TOÁN MÁY TỰ ĐỘNG (MACHINE AUDIT REPORT)
## ĐỢT HUẤN LUYỆN VÀ ĐÁNH GIÁ THỬ NGHIỆM SẠCH 3 NGƯỜI KÝ (P01, P02, P03) — 25 CỬ CHỈ

* **Dự án**: Nghiên cứu Khoa học — Nhận diện Cử chỉ Ngôn ngữ Ký hiệu Tiếng Việt (VSLR)
* **Tính chất**: Kiểm toán máy tự động (Machine Audit) kiểm tra tính toàn vẹn số liệu và artifact. Đây không phải báo cáo đánh giá của reviewer độc lập.
* **Trạng thái External Review**: **PENDING** (Chờ chuyên gia / reviewer độc lập bên ngoài đánh giá).
* **Trạng thái mô hình**: Kết quả LOSO thử nghiệm trên 3 người ký P01–P03 (Interim Experimental Benchmark). Chưa phải mô hình phát hành chính thức.
* **Thời gian thực hiện**: 2026-09-03
* **Thư mục Run lưu trữ**: `runs/p123-clean-20260903-092539/`

---

## 1. TỔNG HỢP KIỂM TOÁN TỰ ĐỘNG (MACHINE AUDIT SUMMARY)

| Hạng mục kiểm tra | Tiêu chuẩn đặt ra | Kết quả thực tế | Kết luận |
|---|---|:---:|:---:|
| **Đóng băng Code (Phase 1)** | Commit xác định, tracked tree sạch 100%, 114 test pass | Commit `2ec3dddc1286596c57eaa605101a6b06281e9fce`, 114 tests pass | **PASS** |
| **Khóa Dataset (Phase 2)** | 450 video P01–P03, 450 SHA-256 duy nhất, FHD 60fps, $\le 10$s | 450 file path & 450 hash duy nhất trong `dataset_files_sha256.csv` | **PASS** |
| **Cổng nghiệm thu dữ liệu** | `vslr-check` exit code 0 | 450/450 clip đạt chuẩn, 0 lỗi, đủ độ phủ 3 người x 25 nhãn | **PASS** |
| **Trích xuất sạch (Phase 3)** | Không dùng cache cũ; 100% cache miss ở lượt LOSO | 450 cache misses, 0 cache hit, `failed_clips = []` | **PASS** |
| **Đánh giá LOSO (Phase 5)** | 3 folds độc lập, 40 epochs, GPU CUDA, không P04 | Exit code 0, không có P04 trong bất kỳ train/test clip nào | **PASS** |
| **Pooled Accuracy** | Tỉ lệ đoán đúng trên 450 clips test thực tế | **88.44%** (**398 / 450 clips**) | **PASS** |
| **Macro Mean Accuracy** | Trung bình cộng 3 folds | **88.44%** | **PASS** |
| **Fold thấp nhất (Worst Fold)** | Ghi nhận trung thực theo số liệu | Fold **P01: 84.00%** (126 / 150 clips) | **PASS** |
| **Fold cao nhất (Best Fold)** | Ghi nhận trung thực theo số liệu | Fold **P03: 95.33%** (143 / 150 clips) | **PASS** |
| **Train Ship Model (Phase 6)** | Train toàn bộ 450 clips, không tự gán accuracy | Checkpoint `gesture_lstm.pt` xuất thành công, exit code 0 | **PASS** |
| **Three-Way Binding (Phase 7)** | `training_signature` khớp giữa LOSO report, metrics và weights | Chữ ký `564a3a0b32f06ac7...` khớp 3 bên, SHA-256 checkpoint khớp | **PAIR OK** |
| **Machine Audit Script** | Script kiểm toán đối soát độc lập mọi chỉ số | `audit_run_artifacts.py` xác nhận đạt toàn bộ tiêu chí kỹ thuật | **PASS** |

---

## 2. GIẢI TRÌNH DỮ LIỆU & NGUYÊN TẮC TOÀN VẸN (DATA INTEGRITY AUDIT)

### 2.1. Căn cứ loại trừ P04 khỏi đợt thử nghiệm này
1. **Bằng chứng số liệu (Ground-truth SHA-256 Collision)**:  
   Tại thư mục cử chỉ `Hẹn gặp lại`, lệnh băm SHA-256 xác nhận 2 file video của P04 có nội dung byte trùng lặp 100% với P02:
   * `recordings_v1/P04/Hẹn gặp lại/004.mov` $\leftrightarrow$ `recordings_v1/P02/Hẹn gặp lại/001.MOV`  
     SHA-256: `D3C85A772E40A72A190AC8C0D47EC5BAD1F097C52D645133F8F853768117AA3C`
   * `recordings_v1/P04/Hẹn gặp lại/003.mov` $\leftrightarrow$ `recordings_v1/P02/Hẹn gặp lại/002.MOV`  
     SHA-256: `7477E1CD487C00441C087D3A0D91B7CEB9F61B8EF6B50EDAF3A6EAAD38D27DE5`
2. **Cổng kỹ thuật tự động (Structural Fail-Closed Gate)**:  
   Hàm `validate_recording_tree` (`prepare_train.py:186-190`) phát hiện duplicate bytes và từ chối chạy ngay lập tức (`exit code 1`).
3. **Nguyên tắc phương pháp luận khoa học**:  
   Trong đánh giá **Leave-One-Signer-Out (LOSO)**, nếu đưa P04 vào thì khi test fold P04, mô hình đã học chính clip đó từ P02 lúc train $\rightarrow$ vi phạm điều kiện độc lập (Data Leakage).  
   *(Lưu ý: Thông tin "P02 quay thay do P04 không làm được" là giải trình giao tiếp ngoại tuyến; về mặt dữ liệu, hệ thống chỉ căn cứ vào tính trùng lặp byte-for-byte).*

### 2.2. Xử lý Container Gate (Clip quá hạn của P02)
* Clip `recordings_v1/P02/Hôm nay bạn khỏe không/002.MOV` ban đầu dài `10.0567s` (vượt ngưỡng 10.0s của container gate).
* Đã dùng FFmpeg cắt bớt xuống còn `9.5095s` (570 frames @ 59.94 FPS).
* *(Metadata xác nhận clip giảm từ khoảng 10.055s xuống 9.5095s; artifact không chứng minh phần bị cắt là tĩnh).*
* Bản gốc được lưu an toàn tại `002.MOV.bak` ngoài dataset. File `.bak` tuyệt đối không được đưa vào train.

### 2.3. Hợp đồng 25 Cử chỉ
* Manifest được nhóm chốt gồm 25 nhãn.
* File `dataset/labels.txt` đã được chốt cố định 25 dòng theo chuẩn Unicode NFC. Thứ tự dòng chính là Class Index (0 đến 24) của mô hình BiLSTM.

---

## 3. THÔNG SỐ KIẾN TRÚC & MÔI TRƯỜNG THỰC TẾ

*(Trích xuất nguyên trạng từ `metrics.json` và checkpoint `gesture_lstm.pt`)*

* **Feature Extractor**: MediaPipe Holistic (Pose, Face, Left Hand, Right Hand).
  * `features_version = 3`
  * `feature_dim = 203`
  * `sequence_length = 60`
* **Mô hình Mạng (Neural Architecture)**: BiLSTM
  * `hidden_size = 96`
  * `num_layers = 1`
  * `bidirectional = true`
  * `pooling = fwd_last_bwd_first`
* **Tối ưu hóa (Optimization & Loss)**:
  * Optimizer: `AdamW` (`learning_rate = 0.001`, `weight_decay = 0.0001`)
  * Loss Function: `CrossEntropyLoss` (`train_label_smoothing = 0.03`)
* **Huấn luyện & Tăng cường (Training & Augmentation)**:
  * Epochs: `40` (cố định, không early stopping trên test set)
  * Batch Size: `32`
  * Augmentation on-the-fly (chỉ tập train): `120` samples/clip (jitter, scale, time-warp)
  * Random Seed: `42`
  * Workers: `4`
* **Phần cứng & Runtime**:
  * GPU: `NVIDIA GeForce RTX 4050 Laptop GPU`
  * NVIDIA Driver: `591.66` | CUDA Driver: `13.1` | PyTorch CUDA Build: `12.6`
  * Device: `cuda`
  * PyTorch: `2.12.1+cu126` | NumPy: `1.26.4` | Python: `3.11.9`

---

## 4. BẢNG SỐ LIỆU ĐÁNH GIÁ CHI TIẾT (LOSO 3-FOLDS)

### 4.1. Kết quả theo từng Fold

| Fold (Held-out Signer) | Train Clips | Test Clips | Clips đúng | Accuracy | Test Loss (CE) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold P01** | 300 | 150 | 126 / 150 | **84.00%** | 0.7177 |
| **Fold P02** | 300 | 150 | 129 / 150 | **86.00%** | 0.5314 |
| **Fold P03** | 300 | 150 | 143 / 150 | **95.33%** | 0.2587 |
| **TOÀN BỘ (Pooled)** | **—** | **450** | **398 / 450** | **88.44%** | **—** |

### 4.2. Tiến trình Hội tụ qua các Epochs (Trích xuất từ `loso_report.json`)
* **Fold P01**:
  * Epoch 001: Train Acc 73.5% / Loss 1.0259 | Held-out P01 Acc: 84.0% / Loss 0.7657
  * Epoch 010: Train Acc 100.0% / Loss 0.2411 | Held-out P01 Acc: 90.0% / Loss 0.4393
  * Epoch 020: Train Acc 100.0% / Loss 0.2373 | Held-out P01 Acc: 84.7% / Loss 0.7371
  * Epoch 030: Train Acc 100.0% / Loss 0.2360 | Held-out P01 Acc: 85.3% / Loss 0.5327
  * Epoch 040: Train Acc 100.0% / Loss 0.2350 | Held-out P01 Acc: 84.0% / Loss 0.7177
* **Fold P02**:
  * Epoch 001: Train Acc 75.4% / Loss 0.9948 | Held-out P02 Acc: 80.0% / Loss 0.6422
  * Epoch 010: Train Acc 100.0% / Loss 0.2400 | Held-out P02 Acc: 83.3% / Loss 0.6752
  * Epoch 020: Train Acc 100.0% / Loss 0.2389 | Held-out P02 Acc: 86.7% / Loss 0.5812
  * Epoch 030: Train Acc 100.0% / Loss 0.2361 | Held-out P02 Acc: 86.0% / Loss 0.4872
  * Epoch 040: Train Acc 100.0% / Loss 0.2350 | Held-out P02 Acc: 86.0% / Loss 0.5314
* **Fold P03**:
  * Epoch 001: Train Acc 80.7% / Loss 0.8524 | Held-out P03 Acc: 84.7% / Loss 0.4539
  * Epoch 010: Train Acc 100.0% / Loss 0.2386 | Held-out P03 Acc: 88.0% / Loss 0.4609
  * Epoch 020: Train Acc 100.0% / Loss 0.2369 | Held-out P03 Acc: 87.3% / Loss 0.5137
  * Epoch 030: Train Acc 100.0% / Loss 0.2355 | Held-out P03 Acc: 88.7% / Loss 0.4255
  * Epoch 040: Train Acc 100.0% / Loss 0.2344 | Held-out P03 Acc: 95.3% / Loss 0.2587

### 4.3. Phân bố Độ chính xác Chi tiết trên 25 Cử chỉ (Mỗi nhãn test 18 clips)
* **100.0% (18 / 18 clips)**: `Bạn có cần giúp đỡ không`, `Bạn có vấn đề gì không`, `Bạn quê ở đâu`, `Chuyện gì`, `Lâu rồi không gặp`, `Mấy tuổi`, `Tôi bình thường`, `Tôi không khỏe`, `Tạm biệt`, `Xin chào` (10 cử chỉ).
* **94.4% (17 / 18 clips)**: `Bạn tên gì`, `Cảm ơn`, `Gọi xe cứu thương`, `Hôm nay bạn khỏe không`, `Hẹn gặp lại`, `Siêu thị`, `Về nhà cẩn thận` (7 cử chỉ).
* **88.9% (16 / 18 clips)**: `Được không`, `Sao thế` (2 cử chỉ).
* **77.8% (14 / 18 clips)**: `Tôi khỏe` (1 cử chỉ).
* **66.7% (12 / 18 clips)**: `Bạn đang làm gì`, `Rất vui được gặp bạn`, `Xin lỗi` (3 cử chỉ).
* **50.0% (9 / 18 clips)**: `Như thế nào` (1 cử chỉ).
* **44.4% (8 / 18 clips)**: `Đi đâu` (1 cử chỉ).
*(Tổng đối soát: $10 \times 18 + 7 \times 17 + 2 \times 16 + 1 \times 14 + 3 \times 12 + 1 \times 9 + 1 \times 8 = \mathbf{398 / 450}$ clips đúng).*

### 4.4. Toàn bộ 23 Cặp Nhầm lẫn (Kê khai đầy đủ 52 clips sai)
Trong 52 clips đoán sai (11.56%), phân bố cụ thể như sau:
1. `Như thế nào` $\rightarrow$ `Xin chào`: **7 clips**
2. `Bạn đang làm gì` $\rightarrow$ `Bạn tên gì`: **6 clips**
3. `Đi đâu` $\rightarrow$ `Được không`: **6 clips**
4. `Rất vui được gặp bạn` $\rightarrow$ `Xin chào`: **4 clips**
5. `Tôi khỏe` $\rightarrow$ `Lâu rồi không gặp`: **4 clips**
6. `Xin lỗi` $\rightarrow$ `Gọi xe cứu thương`: **4 clips**
7. `Đi đâu` $\rightarrow$ `Sao thế`: **2 clips**
8. `Đi đâu` $\rightarrow$ `Tạm biệt`: **2 clips**
9. `Được không` $\rightarrow$ `Đi đâu`: **2 clips**
10. `Xin lỗi` $\rightarrow$ `Về nhà cẩn thận`: **2 clips**
11. `Bạn tên gì` $\rightarrow$ `Chuyện gì`: **1 clip**
12. `Cảm ơn` $\rightarrow$ `Như thế nào`: **1 clip**
13. `Gọi xe cứu thương` $\rightarrow$ `Xin lỗi`: **1 clip**
14. `Hẹn gặp lại` $\rightarrow$ `Lâu rồi không gặp`: **1 clip**
15. `Hôm nay bạn khỏe không` $\rightarrow$ `Như thế nào`: **1 clip**
16. `Như thế nào` $\rightarrow$ `Cảm ơn`: **1 clip**
17. `Như thế nào` $\rightarrow$ `Về nhà cẩn thận`: **1 clip**
18. `Rất vui được gặp bạn` $\rightarrow$ `Cảm ơn`: **1 clip**
19. `Rất vui được gặp bạn` $\rightarrow$ `Siêu thị`: **1 clip**
20. `Sao thế` $\rightarrow$ `Được không`: **1 clip**
21. `Sao thế` $\rightarrow$ `Tôi bình thường`: **1 clip**
22. `Siêu thị` $\rightarrow$ `Chuyện gì`: **1 clip**
23. `Về nhà cẩn thận` $\rightarrow$ `Như thế nào`: **1 clip**

*(Tổng số clips nhầm lẫn qua đúng 23 cặp: **52 / 52** clips, đối soát khớp 100%).*

---

## 5. ĐỐI SOÁT TOÀN VẸN BA BÊN (THREE-WAY BINDING AUDIT)

Hệ thống đã thực hiện kiểm định ràng buộc 3 bên giữa file báo cáo LOSO, file metadata và checkpoint weights:

```text
[loso_report.json] ── training_signature ──> 564a3a0b32f06ac795ffa78b8d88083fa24c9a879ed55b6ac1ba2607bfbfcb2b
[metrics.json]     ── training_signature ──> 564a3a0b32f06ac795ffa78b8d88083fa24c9a879ed55b6ac1ba2607bfbfcb2b
[gesture_lstm.pt]  ── config.signature   ──> 564a3a0b32f06ac795ffa78b8d88083fa24c9a879ed55b6ac1ba2607bfbfcb2b
                                              └─► Kết quả: KHỚP 100% (TRIPLE BINDING OK)

[metrics.json.checkpoint_sha256]           ──> 277a8def21179bbcf96285ab9f7ef3ff22aafbfc8b61208e9b82ffcbb5236ac4
[SHA-256 thực tế của gesture_lstm.pt]      ──> 277a8def21179bbcf96285ab9f7ef3ff22aafbfc8b61208e9b82ffcbb5236ac4
                                              └─► Kết quả: KHỚP 100% (WEIGHT HASH OK)
```

---

## 6. KẾT LUẬN VÀ GIỚI HẠN PROVENANCE

1. **Kết quả kỹ thuật**: Các chỉ số cốt lõi (450 clips, không P04, 398/450 = 88.44%, triple signature binding, SHA-256 checkpoint) đã được kiểm chứng tính toán tự động qua script `audit_run_artifacts.py`.
2. **Hạn chế Provenance**: Script tạo dataset manifest ban đầu (`scripts/build_dataset_manifest.py`) được chạy trước khi được commit vào Git repo.
3. **Trạng thái Review Độc lập**: **PENDING** — Chờ một reviewer độc lập (con người hoặc hệ thống đánh giá ngoài) kiểm tra và đối soát chéo độc lập.
