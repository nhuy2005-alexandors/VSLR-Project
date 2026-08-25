# Checkpoint — VSLR

_Cập nhật: 2026-08-24_

## Done

- **Feature extraction v2**: MediaPipe Holistic, 67 landmark × 3 = 201 features, sequence 60; mỗi file video có instance MediaPipe riêng; cache khóa theo SHA-256 bytes nguồn + path/mtime/dimension/`FEATURES_VERSION=2`.
- **Model**: BiLSTM pooling đúng `fwd-last + bwd-first`; checkpoint giữ metadata `pooling`/`features_version`/`model_architecture_version`. Artifact lệch feature version bị chặn mặc định; architecture version lạ luôn bị từ chối vì cùng tensor shape vẫn có thể khác semantics.
- **Train pipeline**: `Clip(label, person, path)`, signer split trước augment, augment on-the-fly deterministic; `--loso` chỉ đo, mode mặc định chỉ ship; không early stopping/refit.
- **Cổng nạp video**: `vslr-train --data-dir` đọc manifest, chuẩn hóa NFC, chặn nhãn thiếu toàn cục, thư mục ngoài manifest, khoảng trắng sai và hai raw directory NFC/NFD trùng nhau trước MediaPipe. Class order lấy từ manifest.
- **Cache hardening**: cache đọc được nhưng sai shape/NaN/stats cũng bị bỏ và extract lại.
- **LOSO diagnostics**: report có một prediction row cho mỗi test clip; console in fold tệ, nhãn tệ/confusion và clip sai có hand ratio thấp.
- **Artifact pairing**: `training_signature` nằm trong report, metrics và chính checkpoint; bao phủ data/nhãn, hyperparameter, model/augment/training recipe, device/runtime. `metrics.json.checkpoint_sha256` buộc metadata vào đúng file weights; ship chỉ in `PAIR OK` khi cả ba signature khớp.
- **Realtime**: `SegmentTracker` chặn một cử chỉ chậm thành hai từ; ngưỡng theo giây; min-duration loại idle word-gap khỏi active span; mọi duration `nan/inf` bị từ chối trước khi load model.
- **On-site QA**: `vslr-check` bắt buộc có manifest, dùng chung cache, phân biệt lỗi quay với `LOI HE THONG`, không train hay ghi `models/`.
- **Fail-closed extraction**: train chỉ được bỏ qua `ClipExtractionError`/file biến mất; permission, MediaPipe runtime và lỗi hệ thống khác abort ngay thay vì bị tính vào quota 5% clip xấu.
- **Windows**: train/check/realtime tự cấu hình stdout/stderr UTF-8.
- **Khởi tạo dataset V1**: `vslr-init-dataset` đọc manifest, tạo an toàn cây
  `<người>/<nhãn>/`, hỗ trợ dry-run và không tạo/di chuyển/ghi đè video. Manifest đã chốt 30
  nhãn; kế hoạch hiện tại là 4 người × 30 nhãn × 6 clip = 720 clip.
- **Verify hiện tại**: `python -m pytest -q` → **78 passed, 31 subtests passed**; `git diff --check` exit 0 (chỉ cảnh báo LF→CRLF). Smoke cây tạm 2 người × 2 nhãn: check/LOSO/ship đều exit 0, 4 prediction rows, LOSO không ghi checkpoint, three-way signature + checkpoint SHA-256 khớp, strict load feature v2 / architecture v1 thành công.

- **LOSO chỉ ra nhãn nào sai**: `evaluate()` (`prepare_train.py:317`) trả một dòng mỗi clip; `folds[].predictions` mang `{video, person, label, predicted, confidence, correct}`; console in 5 nhãn tệ nhất kèm cặp confusion và clip vừa-sai-vừa-quay-tệ. `confidence` là softmax-max nên cùng thang với `--confidence` của demo. `build_eval_loader()` (`:305`) dùng chung cho train và report, không augment không shuffle — đó là điều kiện để dòng i ứng clip i.
- **Khóa dẫn xuất KHÔNG lưu vào report**: `per_label_accuracy` / `worst_labels` / `suspect_clips` là hàm thuần của `predictions[]` + `extraction[]`, chỉ in console. Lưu bản sao là tạo chỗ để trôi khỏi nguồn.
- **`suspect_clips` dùng ngưỡng tuyệt đối 0,5**, không phải "dưới trung vị": một nửa tập nào cũng dưới trung vị của chính nó nên luật trung vị không bao giờ trả về danh sách rỗng.

## In progress / working tree

- Nhánh `pipeline-signer-split` có **7 commit** sau `main`, HEAD `c8ee92a`; chưa merge.
- Các sửa blocker từ reviewer cuối đang **uncommitted** trong source/tests/docs: three-way artifact binding, manifest fail-closed, typed extraction failures, exact NFC directory gate và finite realtime durations.
- Có chủ ý không đụng: `models/gesture_lstm.pt`, `models/metrics.json`, `models/backups/` và 18 clip `.mov` untracked của người dùng.
- `dataset/labels.txt` là manifest V1 gồm 30 nhãn đã chốt; thứ tự dòng là class index.

## Next

0. **`vslr-sentence`** — đo phần ghép câu offline. Điều kiện tiên quyết (ngưỡng giây, `Segment.duration`) đã xong; còn phải kéo lọc `min_seconds` + ngưỡng confidence + ghép câu ra khỏi `realtime.main()` để hai đường dùng chung một bản. Đây là thứ duy nhất đo được mục tiêu thật của dự án.

1. Chạy dry-run rồi tạo cây mới bằng `vslr-init-dataset --data-dir dataset/recordings_v1 --people-count 4`.
2. Nạp video theo `dataset/recordings_v1/P01/<Nhãn tiếng Việt>/001.mov`, đủ 6 clip mỗi nhãn; chạy `vslr-check` ngay sau P01.
3. Sau ít nhất 2 người đủ mọi nhãn, chạy LOSO. Khi đủ người, giữ report có `training_signature` cuối.
4. Chỉ sau khi chủ dự án đồng ý ghi đè artifact: backup model hiện tại rồi chạy mode ship với **cùng** config/signature.
5. Test `vslr-camera --no-tts` thật bằng checkpoint v2 mới; checkpoint hiện tại v1 bị chặn đúng thiết kế.
6. Lỗ hổng còn lại của `record-to-test.md`: `vslr-sentence` đo câu offline từ `sentences.csv`.

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

- Checkpoint tracked là model 1 epoch, pooling/landmark v1; không có accuracy báo cáo được và không chạy mặc định với extractor v2.
- Chỉ ghép accuracy khi report/metrics/checkpoint cùng `training_signature` và `checkpoint_sha256` khớp file weights.
- Data quality issue phải sửa file gốc + backup, không chỉ lọc in-memory.
- Chi tiết tích lũy: `docs/GOTCHAS.md`; quyết định: `docs/DECISIONS.md`; as-built mới: `docs/technical_specs/video-ingest-readiness.md`.
