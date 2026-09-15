# Checkpoint — VSLR

_Cập nhật: 2026-09-16 (Overnight Run: Nghiệm thu LOSO 13-epoch & Huấn luyện Ship Candidate V3)_

## Done

- **Nghiệm thu Độc lập & Đóng băng Lựa chọn Siêu tham số LOSO V3 13-epoch**:
  - Run nghiệm thu: `runs/v3-4signers-24-8clips-20260915-215753/`.
  - Cơ sở lựa chọn `epochs=13`: **development-selected epoch count** từ lịch sử hội tụ LOSO trên tập phát triển, chưa phải external validation optimum.
  - Kết quả tái tính toán độc lập từ raw `folds[].predictions` (768 predictions):
    - **Pooled Accuracy**: 98.5677% (757 / 768 clips đúng).
    - **Macro Mean Accuracy**: 98.5677%.
    - **Top-3 Accuracy**: 99.3490% (763 / 768 clips đúng top-3).
    - Fold P01: 192/192 (100.00%), loss 0.0261.
    - Fold P02: 189/192 (98.44%), loss 0.1544.
    - Fold P03: 187/192 (97.40%), loss 0.1433 (worst fold).
    - Fold P04: 189/192 (98.44%), loss 0.1127.
  - 6 nhãn trọng điểm lịch sử:
    - `Sao thế`: 32/32 (100.00%)
    - `Được không`: 32/32 (100.00%)
    - `Bạn tên gì`: 31/32 (96.88%), 1 clip nhầm sang `Bạn đang làm gì`
    - `Tôi khỏe`: 32/32 (100.00%)
    - `Bạn có vấn đề gì không`: 32/32 (100.00%)
    - `Bạn đang làm gì`: 32/32 (100.00%)
  - Độc lập kiểm định fail-closed: `python scripts/verify_v3_run.py` đạt **exit code 0**, 20 file artifacts đầy đủ, 19 file trong `SHA256SUMS.txt` khớp byte không BOM, 0 rò rỉ dữ liệu P05.
  - Reviewer độc lập: Đạt **PASS with should-fix** (toàn bộ khuyến nghị đã được xử lý và commit vào git).

- **Huấn luyện & Niêm phong Ship Candidate V3 (`runs/v3-ship-4signers-24-8clips-13ep-20260916-020145`)**:
  - Dữ liệu huấn luyện: Toàn bộ 768 clips P01–P04 từ `dataset/recordings_v3_4x24_8`.
  - Thuật toán & Cấu hình đóng băng: `epochs=13`, `augment=120`, `batch_size=32`, `lr=0.001`, `seed=42`, `num_workers=4`, cache `dataset/processed/landmark_cache_v3_4x24_8`.
  - Checkpoint: `runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/gesture_lstm.pt` (SHA-256: `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`).
  - Khóa ba chiều: `training_signature = 918588dedeb2e9bdb2dabf8a2f8ac843aa6cf66dc68d61c2e8acb1e4b07a65cf` khớp 100% giữa `loso_report.json`, `metrics.json`, và `gesture_lstm.pt` (`PAIR OK`).
  - Hồ sơ niêm phong: `RUN_MANIFEST.json` và `SHA256SUMS.txt` (20 files, UTF-8 no BOM).
  - Bất biến mô hình sản xuất: Checkpoint `models/gesture_lstm.pt` giữ nguyên tuyệt đối SHA-256 `5e202eb9108c06e446da5ae1d27aaebcec7e14cb2e72a9c233c4ac99da245b83`.

- **Chuẩn bị Sẵn sàng Lệnh Đánh giá Ngoài P05 (Dry Run / No Inference)**:
  - Tập kiểm thử ngoài: `dataset/external_eval_v3/P05` (48 clips = 24 nhãn x 2 clips, 100% độc nhất, 0 mã băm trùng lặp với tập train 768 clips).
  - Cổng Run Manifest Gate của `vslr-eval` đã kiểm thử logic và xác nhận khớp 3 chiều trước khi chạy thật.
  - Tuyệt đối KHÔNG thực hiện suy diễn (inference) hay tính accuracy P05 trong phiên overnight.

- **Hoàn thành Xây dựng & Siết chặt Công cụ Đánh giá Ngoài Độc lập (`vslr-eval`)**:
  - Module: `src/prototype_3_gestures/evaluate.py`.
  - CLI: `vslr-eval` (đã đăng ký trong `pyproject.toml`).
  - Spec: `docs/specs/external-evaluation.md` (phiên bản 2.0, Hoàn thành & Kiểm định).
  - Cổng Run Manifest khóa 3 chiều (Run Manifest Gate): bắt buộc có `--run-manifest` (`RUN_MANIFEST.json`). Kiểm tra trước khi nạp MediaPipe / suy diễn: SHA-256 của `--training-manifest` == `dataset_manifest_sha256`, SHA-256 model == `checkpoint_sha256`, `training_signature` trong checkpoint == `RUN_MANIFEST.json.training_signature`. Lệch bất kỳ trường nào: lập tức dừng (fail-closed), in `error: ...`, exit 2, không trích xuất đặc trưng, không suy diễn, không ghi artifact.
  - Cổng chống rò rỉ dữ liệu (Leakage Gate): bắt buộc có `--training-manifest` có mã băm SHA-256 64-hex hợp lệ, đối chiếu SHA-256 từng clip test với manifest train, lập tức dừng (fail-closed) nếu trùng lặp byte. Chỉ khẳng định Zero Leakage sau khi clip vượt qua cổng này.
  - Cổng an toàn đầu ra (Output Safety): chặn tuyệt đối `--output-dir` trùng hoặc nằm trong `models/`, checkpoint parent hoặc data dir; chặn ghi đè artifacts cũ nếu thiếu `--overwrite`.
  - Cổng toàn vẹn thư mục (Directory Completeness): bắt buộc ít nhất một `--expected-signer`, `--clips-per-label` phải là số nguyên dương $\ge 1$; kiểm tra đủ 24 nhãn Unicode NFC, chặn 2 clip test trùng byte.
  - Phân tách chỉ số: Top-1 Raw Accuracy, Accepted Accuracy, Coverage và Rejection Rate dựa trên ngưỡng `--confidence` (ngưỡng 0.50 chỉ là chẩn đoán/demo tạm thời, không phải production).
  - Xử lý lỗi CLI: In ngắn gọn `error: ...` ra `stderr`, exit 2, không traceback; không nuốt lỗi lập trình bất ngờ (`TypeError`, `AttributeError`,...).
  - Hiện tại người ký P05 chưa có video thực tế trên đĩa nên chưa có số liệu accuracy kiểm thử thực tế.
  - Hỗ trợ cả chế độ thư mục (`--data-dir`) và clip đơn lẻ (`--video "Nhãn=path"`).
  - Tự động sinh báo cáo đầy đủ provenance: `predictions.csv`, `metrics.json`, `REPORT.md`, `confusion_matrix.png` trong `--output-dir`.
  - Test suite: `tests/test_eval.py` (**31 unit tests** kiểm thử nghiêm ngặt).
  - Toàn bộ test suite: `python -m pytest -q` đạt **158 passed, 45 subtests passed**.
  - Bất biến mô hình: Checkpoint `models/gesture_lstm.pt` giữ nguyên mã băm SHA-256 `5e202eb9108c06e446da5ae1d27aaebcec7e14cb2e72a9c233c4ac99da245b83`.

- **Phát hành Candidate Model 24 cử chỉ lên GitHub (`pipeline-signer-split`)**:
  - Commit SHA: `05e6cb9049915a809db4d313f0ceb0d7b68f6cbe`.
  - Checkpoint ship mặc định: `models/gesture_lstm.pt` (24 nhãn, SHA-256 `5e202eb9108c...`).
  - Tài liệu bổ sung: `models/MODEL_CARD.md` và cập nhật `README.md`.
  - Khóa ba chiều: `training_signature = 43316055f78bbf2e83c806396d5791d06faeb57b62bef13be17d77095a9c8675`.

- **Hoàn thành Huấn luyện & Đánh giá Model 24 cử chỉ 4 người ký (P01, P02, P03, P04)**:
  - Thư mục run: `runs/v2-4signers-24-20260907-005546/`.
  - Manifest chuẩn: `dataset/labels_v2_24.txt` (loại bỏ duy nhất nhãn `Hẹn gặp lại` để triệt tiêu data leakage do file hash trùng giữa P02 và P04; giữ nguyên thứ tự 24 nhãn chuẩn NFC).
  - Plan 4x24: `dataset/recording_plan_v2_4x24.json` (4 người x 24 nhãn x 6 clip = 576 clips, 576 SHA-256 độc nhất).
  - Landmark cache độc lập: `dataset/processed/landmark_cache_v2_4x24/`.
  - Kết quả LOSO 4-fold (unaugmented test):
    - **Pooled Accuracy: 90.97% (524 / 576 clips đúng)**.
    - **Macro Mean Accuracy: 90.97%**.
    - Fold P01: 140 / 144 (97.22%), loss 0.1274.
    - Fold P02: 134 / 144 (93.06%), loss 0.3197.
    - Fold P03: 138 / 144 (95.83%), loss 0.2642.
    - Fold P04: 112 / 144 (77.78%), loss 0.9171 (worst fold).
  - Model Ship: Đã huấn luyện trên toàn bộ 576 clips (`gesture_lstm.pt`, SHA-256 `5e202eb9108c06e446da5ae1d27aaebcec7e14cb2e72a9c233c4ac99da245b83`).
  - Khóa ba chiều: `training_signature = 43316055f78bbf2e83c806396d5791d06faeb57b62bef13be17d77095a9c8675` khớp 100% giữa `loso_report.json`, `metrics.json` và `gesture_lstm.pt`.
  - Machine Audit: `python scripts/audit_run_artifacts.py` đạt **PASSED (ALL CRITERIA VERIFIED)** exit code 0.
  - Toàn bộ artifacts và visual charts (`training_curves.png`, `confusion_matrix.png`, `per_label_accuracy.png`, `evaluation_report.md`, `REPORT_FOR_AGENT.md`, `RUN_MANIFEST.json`, `SHA256SUMS.txt`) đã niêm phong độc lập trong thư mục run. Không ghi đè model gốc tại `models/`.

- **Feature extraction v3**: MediaPipe Holistic, 67 landmark × 3 + 2 presence channels trái/phải = 203 features, sequence 60; presence-aware bridge/resample/augment không tạo ghost hand; z group-local (pose theo hip midpoint, tay theo wrist). Mỗi file video có instance MediaPipe riêng; cache khóa theo SHA-256 bytes nguồn + path/mtime/dimension/`FEATURES_VERSION=3`.
- **Model**: BiLSTM pooling đúng `fwd-last + bwd-first`; checkpoint giữ metadata `pooling`/`features_version`/`model_architecture_version`. Artifact lệch feature version bị chặn mặc định; architecture version lạ luôn bị từ chối vì cùng tensor shape vẫn có thể khác semantics.
- **Train pipeline**: `Clip(label, person, path)`, signer split trước augment, augment on-the-fly deterministic; `--loso` chỉ đo, mode mặc định chỉ ship; không early stopping/refit.
- **Cổng nạp video**: `dataset/recording_plan.json` khóa P01–P04 và 6 clip/cặp; init/check/train directory mode cùng đọc plan + `dataset/labels.txt`, chặn thiếu/overfull/người ngoài và exact duplicate bytes trước MediaPipe. Class order lấy từ manifest.
- **Cache hardening**: cache đọc được nhưng sai shape/NaN/stats cũng bị bỏ và extract lại.
- **LOSO diagnostics**: report có một prediction row cho mỗi test clip; console in fold tệ, nhãn tệ/confusion và clip sai có hand ratio thấp.
- **Artifact pairing**: `training_signature` nằm trong report, metrics và chính checkpoint; bao phủ data/nhãn, hyperparameter, model/augment/training recipe, device/runtime. `metrics.json.checkpoint_sha256` buộc metadata vào đúng file weights; ship chỉ in `PAIR OK` khi cả ba signature khớp.
- **Realtime**: `SegmentTracker` dùng min-active hysteresis theo giây, chặn một cử chỉ chậm thành hai từ; no-hand tail bị loại trước model; shared preprocessing với offline; reject policy thiếu calibration thì fail-closed và không TTS.
- **Realtime activity gate**: hand presence không còn tự mở segment; cổ tay phải cao hơn đường hông
  0,5 shoulder-width, có fallback khi thiếu pose. Giữ transition context 1,0s trước/0,5s sau nhưng
  chỉ activity mới tính duration/cap. Offline smoke: P01 25/25 và QIPEDC 3/3, chưa test camera thật.
- **Calibration/OOD**: policy chỉ được fit từ split calibration riêng, có receipt identity/content/checkpoint/signature và category `idle_stationary` + `oov_motion`; chưa có motion gate hay metric OOD thật nếu thiếu dữ liệu.
- **On-site QA**: `vslr-check` bắt buộc có manifest, dùng chung cache, phân biệt lỗi quay với `LOI HE THONG`, không train hay ghi `models/`.
- **Fail-closed extraction**: train chỉ được bỏ qua `ClipExtractionError`/file biến mất; permission, MediaPipe runtime và lỗi hệ thống khác abort ngay thay vì bị tính vào quota 5% clip xấu.
- **Windows**: train/check/realtime tự cấu hình stdout/stderr UTF-8.
- **Khởi tạo dataset V1**: `vslr-init-dataset` đọc recording plan + manifest, tạo an toàn cây
  `<người>/<nhãn>/`, hỗ trợ dry-run và không tạo/di chuyển/ghi đè video. Manifest đã chốt 30
  nhãn; kế hoạch hiện tại là 4 người × 30 nhãn × 6 clip = 720 clip.
- **Verify hiện tại**: `python -m pytest -q` → **158 passed, 45 subtests passed**; py_compile package, editable install và CLI help/dry-run smoke pass. Chưa có raw video V1 và checkpoint feature v3 để đo LOSO/câu/người mới.

- **LOSO chỉ ra nhãn nào sai**: `evaluate()` (`prepare_train.py:317`) trả một dòng mỗi clip; `folds[].predictions` mang `{video, person, label, predicted, confidence, correct}`; console in 5 nhãn tệ nhất kèm cặp confusion và clip vừa-sai-vừa-quay-tệ. `confidence` là softmax-max nên cùng thang với `--confidence` của demo. `build_eval_loader()` (`:305`) dùng chung cho train và report, không augment không shuffle — đó là điều kiện để dòng i ứng clip i.
- **Khóa dẫn xuất KHÔNG lưu vào report**: `per_label_accuracy` / `worst_labels` / `suspect_clips` là hàm thuần của `predictions[]` + `extraction[]`, chỉ in console. Lưu bản sao là tạo chỗ để trôi khỏi nguồn.
- **`suspect_clips` dùng ngưỡng tuyệt đối 0,5**, không phải "dưới trung vị": một nửa tập nào cũng dưới trung vị của chính nó nên luật trung vị không bao giờ trả về danh sách rỗng.

## In progress / working tree

- **2026-09-04 — Triển khai External Stress Test V2 (`dataset/external_test_v2`)**:
  - Khảo sát và bóc tách dữ liệu từ các nguồn mở: QIPEDC (`E001`), Hải Ly VSL (`E002`), UNDP Hà Nội (`E003`).
  - Đã cắt và chuẩn hóa 12 clip bao phủ **10 / 25 cử chỉ** vào `dataset/external_test_v2/clips/`.
  - Đối soát giải phẫu / hình thái cử chỉ: `Mấy tuổi` và `Chuyện gì` của `E002` khớp 100% với cử chỉ chuẩn P01; chỉ ra rõ các biến thể ngôn ngữ ký hiệu ở `Xin chào`, `Tôi khỏe`, `Xin lỗi` của `E003`.
  - Đã chạy smoke test offline qua `vslr-sentence` với model `runs/p123-clean-20260903-092539/gesture_lstm.pt` (SHA-256 `277A8DEF...`): E001 pass 3/3 (100%), E002 và E003 được reject policy từ chối an toàn (confidence 16%–48% < 72%), không sinh phụ đề sai.
  - 15 nhãn còn lại (các câu giao tiếp tình huống) được ghi nhận trong `missing_labels.csv` và cần quay bổ sung từ người ký độc lập theo chuẩn P01-P03.

- **2026-08-28 — pipeline hardening đang triển khai** theo
  `docs/specs/pipeline-hardening.md`: recording plan/count/hash gate, presence-aware
  missing-hand preprocessing, shared offline/realtime segmentation, reject evaluation và
  `vslr-sentence`.

- Nhánh `pipeline-signer-split` có **7 commit** sau `main`, HEAD `c8ee92a`; chưa merge.
- Các sửa blocker từ reviewer cuối đang **uncommitted** trong source/tests/docs: three-way artifact binding, manifest fail-closed, typed extraction failures, exact NFC directory gate và finite realtime durations.
- Có chủ ý không đụng: `models/gesture_lstm.pt`, `models/metrics.json`, `models/backups/` và 18 clip `.mov` untracked của người dùng.
- `dataset/labels.txt` là manifest V1 gồm 30 nhãn đã chốt; thứ tự dòng là class index.

## Next

0. **Root duyệt báo cáo nghiệm thu overnight & Ship Candidate V3**:
   - Run candidate: `runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/`.
   - Trạng thái hiện tại: `READY_FOR_P05_FINAL_REVIEW`.

1. **Chạy đánh giá ngoài độc lập P05** (chỉ chạy khi Root trực tiếp phê duyệt):
   ```powershell
   vslr-eval `
     --data-dir dataset/external_eval_v3 `
     --model runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/gesture_lstm.pt `
     --training-manifest runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/dataset_files_sha256.csv `
     --run-manifest runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/RUN_MANIFEST.json `
     --expected-signer P05 `
     --clips-per-label 2 `
     --allow-uncalibrated `
     --confidence 0.50 `
     --output-dir evaluation/p05-v3-final-20260916
   ```

2. **Chỉ sau khi Root duyệt kết quả P05**: Cân nhắc sao lưu `models/gesture_lstm.pt` hiện tại rồi cập nhật model production. Chạy calibration reject policy trên dữ liệu âm bản độc lập trước khi kích hoạt TTS webcam.

3. **Chạy dry-run rồi tạo cây mới bằng `vslr-init-dataset --data-dir dataset/recordings_v1 --people-count 4`** nếu cần mở rộng tập 30 cử chỉ V1 sau này.

## Verify

```powershell
python -m pytest

# Quay/nạp xong: nghiệm thu trước, không train
vslr-check --data-dir dataset/recordings_v1 --labels-file dataset/labels.txt --clips-per-label 6

# Đo, không ghi checkpoint
vslr-train --data-dir dataset/recordings_v1 --labels-file dataset/labels.txt --loso --num-workers 4

# Ship chỉ sau khi duyệt ghi đè; phải dùng cùng config với LOSO
vslr-train --data-dir dataset/recordings_v1 --labels-file dataset/labels.txt --num-workers 4
```

`dataset/raw/` hiện vẫn phẳng (`<slug>/*.mov`) và không có signer ID, nên `--data-dir dataset/raw` từ chối đúng thiết kế. Bộ legacy chỉ train được qua `--video`; tuyệt đối không gán giả signer để lấy LOSO.

## Gotchas quan trọng

- Checkpoint tracked là model cũ, không có accuracy báo cáo được và bị chặn với extractor feature v3.
- Chỉ ghép accuracy khi report/metrics/checkpoint cùng `training_signature` và `checkpoint_sha256` khớp file weights.
- Data quality issue phải sửa file gốc + backup, không chỉ lọc in-memory.
- Chi tiết tích lũy: `docs/GOTCHAS.md`; quyết định: `docs/DECISIONS.md`; as-built mới: `docs/technical_specs/video-ingest-readiness.md`.
