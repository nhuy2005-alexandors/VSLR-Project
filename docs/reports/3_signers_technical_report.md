# BÁO CÁO KỸ THUẬT: ĐÁNH GIÁ THỬ NGHIỆM LOSO 3 NGƯỜI KÝ (P01, P02, P03)

* **Trạng thái**: Kết quả thử nghiệm kỹ thuật nội bộ (Interim / Experimental). Chưa phải mô hình phát hành chính thức (Ship Model V1).
* **Kiến trúc đối chiếu từ artifact (`metrics.json`)**:
  - Extractor: MediaPipe Holistic, `features_version = 3`, `feature_dim = 203`, `sequence_length = 60`.
  - Model: BiLSTM (`hidden_size = 96`, `num_layers = 1`, `bidirectional = true`).
  - Optimizer: AdamW (`lr = 0.001`, `weight_decay = 0.0001`).
  - Loss: CrossEntropyLoss (`train_label_smoothing = 0.03`).
  - Runtime / Device: PyTorch 2.12.1+cu126, CUDA (`NVIDIA GeForce RTX 4050 Laptop GPU`).
  - Training signature: `564a3a0b32f06ac795ffa78b8d88083fa24c9a879ed55b6ac1ba2607bfbfcb2b`.
  - Checkpoint SHA-256: `08d164a02948558df0831a80a4ac0da48d86da0ef739dac0b01fb31b89f893b2`.
* **Thư mục Artifacts**: `runs/v1-3signers-20260903-004907/`

---

## PHẦN I: GIẢI TRÌNH LÝ DO CHỈ ĐO ĐẠC TRÊN 3 NGƯỜI KÝ (P01–P03)

Pipeline VSLR áp dụng các cổng kiểm định **fail-closed** (chặn cứng trước khi cấp phát bộ nhớ train để ngăn rò rỉ dữ liệu). Đợt kiểm tra đầu vào ngày 2026-09-03 phát hiện 2 vi phạm dữ liệu tại nguồn `recordings_v1`:

### 1. Vi phạm toàn vẹn dữ liệu tại P04 (Duplicate File Hash)
* **Bằng chứng số liệu (Ground-truth Hash)**: Tại thư mục `Hẹn gặp lại`, lệnh băm SHA-256 xác nhận 2 file của P04 trùng 100% từng byte với file của P02:
  - `recordings_v1/P04/Hẹn gặp lại/004.mov` $\leftrightarrow$ `recordings_v1/P02/Hẹn gặp lại/001.MOV`  
    SHA-256: `D3C85A772E40A72A190AC8C0D47EC5BAD1F097C52D645133F8F853768117AA3C`
  - `recordings_v1/P04/Hẹn gặp lại/003.mov` $\leftrightarrow$ `recordings_v1/P02/Hẹn gặp lại/002.MOV`  
    SHA-256: `7477E1CD487C00441C087D3A0D91B7CEB9F61B8EF6B50EDAF3A6EAAD38D27DE5`
* **Về mặt kỹ thuật**: Hàm `validate_recording_tree` (`prepare_train.py:186-190`) phát hiện duplicate bytes và trả về lỗi:
  ```text
  ERROR: duplicate video content/hash detected; every path must be a distinct recording
  ```
  Lệnh `vslr-train` tự động từ chối chạy (exit code 1) theo đúng thiết kế an toàn.
* **Về mặt phương pháp luận NCKH**:
  - Trong phương pháp **Leave-One-Signer-Out (LOSO)**, tập test của người ký kiểm thử phải hoàn toàn độc lập với tập train.
  - Nếu đưa P04 vào: Khi fold P04 được test, mô hình đã học chính clip đó từ P02 lúc train $\rightarrow$ vi phạm điều kiện độc lập (Data Leakage), làm sai lệch chỉ số đánh giá.
  - *(Lưu ý: Thông tin "P02 quay thay do P04 không làm được" là giải trình bên lề từ con người; về mặt dữ liệu, hệ thống chỉ ghi nhận và xử lý sự trùng lặp byte-for-byte giữa 2 tệp).*

### 2. Chuẩn hóa Container & Hợp đồng Dữ liệu
* **Cắt ngắn clip quá hạn**: Clip `recordings_v1/P02/Hôm nay bạn khỏe không/002.MOV` trước đó dài `10.055s` (vượt ngưỡng 10s của container gate). Đã được dùng FFmpeg cắt bớt 0.5s tĩnh ở cuối xuống `9.5095s` (570 frames @ 59.94 FPS), lưu bản gốc dự phòng `002.MOV.bak`.
* **Thu gọn tập 25 nhãn**: Google Drive gốc của nhóm thực tế chứa 25 thư mục cử chỉ (thiếu 6 nhãn so với bản đề xuất 30 nhãn ban đầu và có thêm nhãn `Tôi khỏe`). File `dataset/labels.txt` đã được cập nhật chính xác 25 nhãn theo chuẩn Unicode NFC.
* **Cô lập bộ 3 người ký**: Để kiểm thử kỹ thuật pipeline mà không thỏa hiệp kỷ luật số liệu, 450 clips sạch của P01, P02, P03 được đưa vào kế hoạch `dataset/recording_plan_p123.json` (`dataset_version: recordings_v1_p123`). 100% 450 clips đều có hash duy nhất, đạt chuẩn FHD 1080p và vượt qua mọi gate.

---

## PHẦN II: BÁO CÁO KẾT QUẢ HUẤN LUYỆN VÀ ĐÁNH GIÁ (LOSO 3-FOLDS)

*(Toàn bộ số liệu dưới đây được đối chiếu trực tiếp từ `runs/v1-3signers-20260903-004907/loso_report.json` và `metrics.json`)*

### 1. Tổng quan Đánh giá
* **Tổng số clips kiểm thử thực tế**: 450 clips (3 người ký $\times$ 25 nhãn $\times$ 6 clips, không augmentation trong tập test).
* **Macro Mean Accuracy**: **88.44%** (Trung bình unweighted của 3 folds).
* **Pooled Accuracy**: **88.44%** (**398 / 450 clips** đoán đúng).
* **Tổng số clip sai**: **52 / 450 clips** (tỉ lệ lỗi: **11.6%**).

| Fold (Held-out Signer) | Train Clips | Test Clips | Val Accuracy (Exact) | Val Loss (CE) |
|:---:|:---:|:---:|:---:|:---:|
| **P01** | 300 | 150 | **84.00%** (126 / 150) | 0.7177 |
| **P02** | 300 | 150 | **86.00%** (129 / 150) | 0.5314 |
| **P03** | 300 | 150 | **95.33%** (143 / 150) | 0.2587 |

---

### 2. Tiến trình Huấn luyện Thực tế từng Fold (Trích xuất từ `loso_report.json`)

* **Fold P01** (Held-out: P01 | Train: P02, P03):
  - Epoch 001: Train Acc 73.5% / Loss 1.0259 | Val Acc 84.0% / Val Loss 0.7657
  - Epoch 010: Train Acc 100.0% / Loss 0.2411 | Val Acc 90.0% / Val Loss 0.4393
  - Epoch 020: Train Acc 100.0% / Loss 0.2373 | Val Acc 84.7% / Val Loss 0.7371
  - Epoch 030: Train Acc 100.0% / Loss 0.2360 | Val Acc 85.3% / Val Loss 0.5327
  - Epoch 040: Train Acc 100.0% / Loss 0.2350 | Val Acc 84.0% / Val Loss 0.7177

* **Fold P02** (Held-out: P02 | Train: P01, P03):
  - Epoch 001: Train Acc 75.4% / Loss 0.9948 | Val Acc 80.0% / Val Loss 0.6422
  - Epoch 010: Train Acc 100.0% / Loss 0.2400 | Val Acc 83.3% / Val Loss 0.6752
  - Epoch 020: Train Acc 100.0% / Loss 0.2389 | Val Acc 86.7% / Val Loss 0.5812
  - Epoch 030: Train Acc 100.0% / Loss 0.2361 | Val Acc 86.0% / Val Loss 0.4872
  - Epoch 040: Train Acc 100.0% / Loss 0.2350 | Val Acc 86.0% / Val Loss 0.5314

* **Fold P03** (Held-out: P03 | Train: P01, P02):
  - Epoch 001: Train Acc 80.7% / Loss 0.8524 | Val Acc 84.7% / Val Loss 0.4539
  - Epoch 010: Train Acc 100.0% / Loss 0.2386 | Val Acc 88.0% / Val Loss 0.4609
  - Epoch 020: Train Acc 100.0% / Loss 0.2369 | Val Acc 87.3% / Val Loss 0.5137
  - Epoch 030: Train Acc 100.0% / Loss 0.2355 | Val Acc 88.7% / Val Loss 0.4255
  - Epoch 040: Train Acc 100.0% / Loss 0.2344 | Val Acc 95.3% / Val Loss 0.2587

---

### 3. Phân bố Độ chính xác Chi tiết theo Nhãn (25 Cử chỉ)

Dữ liệu tổng hợp từ toàn bộ 450 predictions trong `loso_report.json` (mỗi nhãn được test chính xác 18 lần = 3 người $\times$ 6 clips):

| Tỉ lệ đúng | Số clips đúng | Danh sách cử chỉ |
|:---:|:---:|---|
| **100.0%** | **18 / 18** | `Bạn có cần giúp đỡ không`, `Bạn có vấn đề gì không`, `Bạn quê ở đâu`, `Chuyện gì`, `Lâu rồi không gặp`, `Mấy tuổi`, `Tôi bình thường`, `Tôi không khỏe`, `Tạm biệt`, `Xin chào` (10 cử chỉ) |
| **94.4%** | **17 / 18** | `Bạn tên gì`, `Cảm ơn`, `Gọi xe cứu thương`, `Hôm nay bạn khỏe không`, `Hẹn gặp lại`, `Siêu thị`, `Về nhà cẩn thận` (7 cử chỉ) |
| **88.9%** | **16 / 18** | `Được không`, `Sao thế` (2 cử chỉ) |
| **77.8%** | **14 / 18** | `Tôi khỏe` (1 cử chỉ) |
| **66.7%** | **12 / 18** | `Bạn đang làm gì`, `Rất vui được gặp bạn`, `Xin lỗi` (3 cử chỉ) |
| **50.0%** | **9 / 18** | `Như thế nào` (1 cử chỉ) |
| **44.4%** | **8 / 18** | `Đi đâu` (1 cử chỉ) |

*(Kiểm toán tổng: $10 \times 18 + 7 \times 17 + 2 \times 16 + 1 \times 14 + 3 \times 12 + 1 \times 9 + 1 \times 8 = 180 + 119 + 32 + 14 + 36 + 9 + 8 = \mathbf{398 / 450}$, khớp 100%).*

---

### 4. Các Cặp Cử chỉ Nhầm lẫn Nhiều nhất (Ground-truth Confusion Count)

Trong 52 trường hợp đoán sai, các cặp nhầm lẫn phổ biến nhất gồm:
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
12. `Siêu thị` $\rightarrow$ `Chuyện gì`: **1 clip**

---

### 5. Tính toàn vẹn Artifact (Three-Way Binding)

* Lệnh huấn luyện mô hình tổng hợp trên toàn bộ 450 clips (40 epochs) đã xuất file weights:
  `runs/v1-3signers-20260903-004907/gesture_lstm.pt` (987 KB).
* **Kết quả đối soát ba bên**:
  - `loso_report.json` $\leftrightarrow$ `metrics.json` $\leftrightarrow$ checkpoint `gesture_lstm.pt` đều mang cùng một `training_signature = "564a3a0b32f06ac795ffa78b8d88083fa24c9a879ed55b6ac1ba2607bfbfcb2b"`.
  - Khóa `metrics.json.checkpoint_sha256` khớp chính xác mã SHA-256 của file `gesture_lstm.pt` (`08d164a029...`).
  - Trạng thái: **PAIR OK**.

---

### 6. Danh mục Tệp Artifacts Đính kèm

Tất cả artifacts của phiên chạy được lưu trữ độc lập tại:
`runs/v1-3signers-20260903-004907/`

* `training_curves.png`: Biểu đồ Loss và Accuracy qua 40 Epochs cho 3 Folds.
* `confusion_matrix.png`: Ma trận nhầm lẫn kích thước $25 \times 25$.
* `per_label_accuracy.png`: Biểu đồ cột xếp hạng độ chính xác từng cử chỉ.
* `gesture_lstm.pt`: Weights mô hình BiLSTM thử nghiệm.
* `loso_report.json`: Dữ liệu JSON 450 predictions độc lập.
* `metrics.json`: Metadata huấn luyện và SHA-256.
* `evaluation_report.md`: Báo cáo tóm tắt tự động sinh từ JSON.
* `loso.log` & `ship.log`: Toàn bộ nhật ký thực thi.

---

### 7. Kế hoạch khi có Dữ liệu Hoàn thiện của P04

1. **Về phía dữ liệu**: Quay riêng 6 clips độc lập cho P04 tại cử chỉ `Hẹn gặp lại`, bảo đảm người ký P04 thực hiện động tác và không copy chéo tệp từ người khác.
2. **Về phía thư mục**:
   - Lưu 6 clips mới vào đúng vị trí nguồn: `recordings_v1/P04/Hẹn gặp lại/` (từ `001.mov` đến `006.mov`).
   - Đồng bộ sang cây làm việc chính thức:
     ```powershell
     robocopy recordings_v1 dataset/recordings_v1 /E /COPY:DAT /DCOPY:DAT /R:1 /W:1
     ```
3. **Chạy nghiệm thu và Đo đạc chính thức**:
   - Kiểm tra toàn bộ 600 clips bằng `vslr-check`:
     ```powershell
     vslr-check --data-dir dataset/recordings_v1 --recording-plan dataset/recording_plan.json --min-hand-ratio 0.5
     ```
   - Chạy LOSO 4-folds chính thức (không holdout thủ công):
     ```powershell
     vslr-train --data-dir dataset/recordings_v1 --recording-plan dataset/recording_plan.json --loso --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 --seed 42 --num-workers 4
     ```
   - Số liệu 4-folds thu được mới là số liệu chính thức để đưa vào báo cáo đề tài NCKH.
