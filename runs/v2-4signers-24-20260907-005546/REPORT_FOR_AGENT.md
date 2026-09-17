# BÁO CÁO KỸ THUẬT: ĐÁNH GIÁ THỬ NGHIỆM LOSO 4 NGƯỜI KÝ (P01, P02, P03, P04) — 24 CỬ CHỈ

* **Trạng thái**: Kết quả thử nghiệm kỹ thuật (Experimental / Verified).
* **Kiến trúc đối chiếu từ artifact (`metrics.json`)**:
  - Extractor: MediaPipe Holistic, `features_version = 3`, `feature_dim = 203`, `sequence_length = 60`.
  - Model: BiLSTM (`hidden_size = 96`, `num_layers = 1`, `bidirectional = True`).
  - Optimizer: AdamW (`lr = 0.001`, `weight_decay = 0.0001`).
  - Loss: CrossEntropyLoss (`train_label_smoothing = 0.03`).
  - Runtime / Device: PyTorch 2.12.1+cu126, CUDA (`cuda`).
  - Training signature: `43316055f78bbf2e83c806396d5791d06faeb57b62bef13be17d77095a9c8675`.
  - Checkpoint SHA-256: `5e202eb9108c06e446da5ae1d27aaebcec7e14cb2e72a9c233c4ac99da245b83`.
* **Thư mục Artifacts**: `D:/Dev/Workspaces/VSLR-Workspace/runs/v2-4signers-24-20260907-005546`

---

## PHẦN I: BẢO ĐẢM TOÀN VẸN DỮ LIỆU VÀ CÁC CỔNG KIỂM SOÁT

Pipeline VSLR áp dụng các cổng kiểm định **fail-closed** nghiêm ngặt trước khi cấp phát bộ nhớ train để triệt tiêu mọi nguy cơ rò rỉ dữ liệu (Data Leakage):

### 1. Triệt tiêu Rò rỉ Dữ liệu (Loại bỏ `Hẹn gặp lại`)
* Trong tập 25 nhãn ban đầu, nhãn `Hẹn gặp lại` chứa 2 clip của P04 trùng byte với P02.
* Việc quyết định giữ 24 nhãn và loại bỏ duy nhất nhãn `Hẹn gặp lại` đã giải quyết 100% nguyên nhân duplicate cross-signer.
* Toàn bộ 576 clip trong `dataset/recordings_v2_4x24` đều có mã băm SHA-256 hoàn toàn duy nhất, 0 clip trùng lặp.
* Người ký P04 được tham gia đầy đủ và bình đẳng vào cả 4 folds của quy trình đánh giá LOSO.

### 2. Chuẩn hóa Container & Hợp đồng Dữ liệu
* **Tập nhãn**: 24 nhãn chuẩn Unicode NFC được khai báo tại `dataset/labels_v2_24.txt`.
* **Hợp đồng quay**: `dataset/recording_plan_v2_4x24.json` gồm 4 người ký (P01, P02, P03, P04) × 24 nhãn × 6 clips = 576 clips.
* **Landmark Cache**: Độc lập tại `dataset/processed/landmark_cache_v2_4x24`.

---

## PHẦN II: BÁO CÁO KẾT QUẢ HUẤN LUYỆN VÀ ĐÁNH GIÁ (LOSO 4-FOLDS)

*(Toàn bộ số liệu dưới đây được đối chiếu trực tiếp từ `loso_report.json` và `metrics.json`)*

### 1. Tổng quan Đánh giá
* **Tổng số clips kiểm thử thực tế**: 576 clips (4 người ký $\times$ 24 nhãn $\times$ 6 clips, không augmentation trong tập test).
* **Macro Mean Accuracy**: **90.97%** (Trung bình unweighted của 4 folds).
* **Pooled Accuracy**: **90.97%** (**524 / 576 clips** đoán đúng).
* **Tổng số clip sai**: **52 / 576 clips** (tỉ lệ lỗi: **9.03%**).

| Fold (Held-out Signer) | Train Clips | Test Clips | Val Accuracy (Exact) | Val Loss (CE) |
|:---:|:---:|:---:|:---:|:---:|
| **P01** | 432 | 144 | **97.22%** (140 / 144) | 0.1274 |
| **P02** | 432 | 144 | **93.06%** (134 / 144) | 0.3197 |
| **P03** | 432 | 144 | **95.83%** (138 / 144) | 0.2642 |
| **P04** | 432 | 144 | **77.78%** (112 / 144) | 0.9171 |

### 2. Tiến trình Huấn luyện Thực tế từng Fold (Trích xuất từ `loso_report.json`)

* **Fold P01** (Held-out: P01):
  - Epoch 001: Train Acc 81.7% / Loss 0.7779 | Val Acc 91.0% / Val Loss 0.2006
  - Epoch 010: Train Acc 99.9% / Loss 0.2424 | Val Acc 95.8% / Val Loss 0.1466
  - Epoch 020: Train Acc 100.0% / Loss 0.2343 | Val Acc 97.2% / Val Loss 0.1174
  - Epoch 030: Train Acc 100.0% / Loss 0.2330 | Val Acc 97.9% / Val Loss 0.1165
  - Epoch 040: Train Acc 100.0% / Loss 0.2324 | Val Acc 97.2% / Val Loss 0.1274
* **Fold P02** (Held-out: P02):
  - Epoch 001: Train Acc 81.4% / Loss 0.7908 | Val Acc 84.0% / Val Loss 0.5808
  - Epoch 010: Train Acc 100.0% / Loss 0.2366 | Val Acc 91.7% / Val Loss 0.3635
  - Epoch 020: Train Acc 100.0% / Loss 0.2339 | Val Acc 96.5% / Val Loss 0.2274
  - Epoch 030: Train Acc 100.0% / Loss 0.2329 | Val Acc 96.5% / Val Loss 0.1893
  - Epoch 040: Train Acc 100.0% / Loss 0.2322 | Val Acc 93.1% / Val Loss 0.3197
* **Fold P03** (Held-out: P03):
  - Epoch 001: Train Acc 83.2% / Loss 0.7495 | Val Acc 87.5% / Val Loss 0.6410
  - Epoch 010: Train Acc 100.0% / Loss 0.2370 | Val Acc 91.0% / Val Loss 0.3586
  - Epoch 020: Train Acc 100.0% / Loss 0.2361 | Val Acc 96.5% / Val Loss 0.2189
  - Epoch 030: Train Acc 100.0% / Loss 0.2337 | Val Acc 93.1% / Val Loss 0.3131
  - Epoch 040: Train Acc 100.0% / Loss 0.2328 | Val Acc 95.8% / Val Loss 0.2642
* **Fold P04** (Held-out: P04):
  - Epoch 001: Train Acc 83.4% / Loss 0.7440 | Val Acc 78.5% / Val Loss 1.1558
  - Epoch 010: Train Acc 100.0% / Loss 0.2377 | Val Acc 79.9% / Val Loss 1.1394
  - Epoch 020: Train Acc 100.0% / Loss 0.2345 | Val Acc 81.2% / Val Loss 0.8152
  - Epoch 030: Train Acc 100.0% / Loss 0.2332 | Val Acc 77.1% / Val Loss 0.9598
  - Epoch 040: Train Acc 100.0% / Loss 0.2324 | Val Acc 77.8% / Val Loss 0.9171

### 3. Phân bố Độ chính xác Chi tiết theo Nhãn (24 Cử chỉ)

| Tỉ lệ đúng | Số clips đúng | Danh sách cử chỉ |
|:---:|:---:|---|
| **100.0%** | **24 / 24** | `Bạn có vấn đề gì không`, `Chuyện gì`, `Hôm nay bạn khỏe không`, `Lâu rồi không gặp`, `Mấy tuổi`, `Như thế nào`, `Siêu thị`, `Tôi không khỏe`, `Về nhà cẩn thận`, `Xin chào` (10 cử chỉ) |
| **95.8%** | **23 / 24** | `Gọi xe cứu thương`, `Rất vui được gặp bạn`, `Tôi bình thường`, `Tạm biệt` (4 cử chỉ) |
| **87.5%** | **21 / 24** | `Đi đâu`, `Được không` (2 cử chỉ) |
| **83.3%** | **20 / 24** | `Bạn có cần giúp đỡ không`, `Bạn quê ở đâu`, `Bạn đang làm gì`, `Xin lỗi` (4 cử chỉ) |
| **79.2%** | **19 / 24** | `Cảm ơn`, `Tôi khỏe` (2 cử chỉ) |
| **70.8%** | **17 / 24** | `Sao thế` (1 cử chỉ) |
| **62.5%** | **15 / 24** | `Bạn tên gì` (1 cử chỉ) |

### 4. Toàn bộ 20 Cặp Cử chỉ Nhầm lẫn (Kê khai đầy đủ 52 clips sai)

 1. `Sao thế` $\rightarrow$ `Được không`: **7 clips**
 2. `Bạn tên gì` $\rightarrow$ `Bạn có vấn đề gì không`: **5 clips**
 3. `Tôi khỏe` $\rightarrow$ `Lâu rồi không gặp`: **5 clips**
 4. `Bạn có cần giúp đỡ không` $\rightarrow$ `Bạn có vấn đề gì không`: **4 clips**
 5. `Bạn tên gì` $\rightarrow$ `Bạn đang làm gì`: **4 clips**
 6. `Xin lỗi` $\rightarrow$ `Gọi xe cứu thương`: **4 clips**
 7. `Bạn đang làm gì` $\rightarrow$ `Được không`: **3 clips**
 8. `Cảm ơn` $\rightarrow$ `Như thế nào`: **3 clips**
 9. `Được không` $\rightarrow$ `Tạm biệt`: **3 clips**
10. `Bạn quê ở đâu` $\rightarrow$ `Được không`: **2 clips**
11. `Cảm ơn` $\rightarrow$ `Siêu thị`: **2 clips**
12. `Đi đâu` $\rightarrow$ `Được không`: **2 clips**
13. `Bạn quê ở đâu` $\rightarrow$ `Tôi khỏe`: **1 clips**
14. `Bạn quê ở đâu` $\rightarrow$ `Đi đâu`: **1 clips**
15. `Bạn đang làm gì` $\rightarrow$ `Bạn có vấn đề gì không`: **1 clips**
16. `Gọi xe cứu thương` $\rightarrow$ `Mấy tuổi`: **1 clips**
17. `Rất vui được gặp bạn` $\rightarrow$ `Siêu thị`: **1 clips**
18. `Tôi bình thường` $\rightarrow$ `Sao thế`: **1 clips**
19. `Tạm biệt` $\rightarrow$ `Đi đâu`: **1 clips**
20. `Đi đâu` $\rightarrow$ `Tạm biệt`: **1 clips**

*(Tổng số clips nhầm lẫn qua 20 cặp: **52 / 52** clips, khớp 100%).*

### 5. Tính toàn vẹn Artifact (Three-Way Binding)

* Lệnh huấn luyện mô hình tổng hợp trên toàn bộ 576 clips (40 epochs) đã xuất file weights: `gesture_lstm.pt` (965 KB).
* **Kết quả đối soát ba bên**:
  - `loso_report.json` $\leftrightarrow$ `metrics.json` $\leftrightarrow$ checkpoint `gesture_lstm.pt` đều mang cùng một `training_signature = "43316055f78bbf2e83c806396d5791d06faeb57b62bef13be17d77095a9c8675"`.
  - Khóa `metrics.json.checkpoint_sha256` khớp chính xác mã SHA-256 của file `gesture_lstm.pt` (`5e202eb910...`).
  - Trạng thái: **PAIR OK**.

### 6. Danh mục Tệp Artifacts Đính kèm

Tất cả artifacts của phiên chạy được lưu trữ độc lập tại: `D:/Dev/Workspaces/VSLR-Workspace/runs/v2-4signers-24-20260907-005546/`

| Tệp | Mô tả |
|---|---|
| `training_curves.png` | Biểu đồ Loss và Accuracy qua 40 Epochs cho 4 Folds |
| `confusion_matrix.png` | Ma trận nhầm lẫn kích thước 24 x 24 |
| `per_label_accuracy.png` | Biểu đồ cột xếp hạng độ chính xác từng cử chỉ |
| `gesture_lstm.pt` | Weights mô hình BiLSTM thử nghiệm |
| `loso_report.json` | Dữ liệu JSON 576 predictions độc lập |
| `metrics.json` | Metadata huấn luyện và SHA-256 |
| `evaluation_report.md` | Báo cáo tóm tắt tự động sinh từ JSON |
| `RUN_MANIFEST.json` | Manifest toàn diện khóa môi trường và dữ liệu |
| `SHA256SUMS.txt` | Bảng băm toàn bộ file trong run |
| `loso.log` & `ship.log` | Toàn bộ nhật ký thực thi |

### 7. Kết luận & Đánh giá Tính toàn vẹn

1. **Dữ liệu hoàn chỉnh**: Toàn bộ 4 người ký (P01, P02, P03, P04) tham gia đầy đủ, mỗi người đóng góp đúng 144 clips trên 24 cử chỉ.
2. **Không trùng lặp**: 100% 576 clips đều có SHA-256 duy nhất, loại bỏ hoàn toàn nguy cơ data leakage.
3. **Hiệu năng LOSO**: Pooled accuracy đạt 90.97%, macro mean accuracy đạt 90.97%.