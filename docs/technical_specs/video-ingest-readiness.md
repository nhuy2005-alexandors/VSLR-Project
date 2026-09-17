# As-built — Cổng nạp video an toàn + chẩn đoán LOSO

_Xong: 2026-08-23. Spec liên quan: `pipeline-restructure.md`, `record-to-test.md` rev 2. Verify hiện tại: **78 passed, 31 subtests passed**._

## What was built

Pipeline `--data-dir` giờ fail-fast trước MediaPipe nếu cây video không khớp manifest nhãn; cache hit được kiểm contract; LOSO ghi prediction theo từng clip và in nhãn/clip đáng chú ý; report và checkpoint chỉ được ghép bằng một chữ ký đầy đủ. Realtime không còn chạy âm thầm checkpoint landmark sai version, và min-duration chỉ đo thời gian thực sự thấy tay.

## End-to-end flow

```text
dataset/labels.txt
        +
DIR/<person>/<nhãn>/*.mov|*.mp4
        ↓ discover + NFC + exact manifest gate
        ↓ signer/coverage validation
        ↓ cache validate hoặc MediaPipe extract
        ├─ --loso → folds[].predictions + training_signature
        └─ ship   → checkpoint feature v3 (signature bên trong) + metrics + checkpoint SHA-256
```

- `vslr-check` dùng cùng parser nhãn và cache, nên check xong train ăn cache ngay.
- `vslr-train --data-dir` lấy class-index order từ thứ tự dòng trong manifest, không từ thứ tự đi cây.
- Nhãn thiếu ở **tất cả** người, thư mục slug/khoảng trắng ngoài manifest, hoặc hai raw directory NFC/NFD của cùng người quy về một nhãn đều exit 1 trước khi tạo `model-dir`.
- Cache key chứa SHA-256 của bytes video cùng path/mtime/size/feature contract; copy đè giữ timestamp vẫn miss đúng. `_read_cache_entry()` còn yêu cầu sequence `(60, 203)`, finite, frame counts hợp lệ và hand ratio trong range; sai thì xóa và extract lại.
- `training_signature` bao phủ data fingerprint, thứ tự nhãn, epochs, augment, batch size, learning rate, seed, sequence/feature dimensions, pooling, feature/augmentation/model/training-recipe version, optimizer/loss constants, device và Torch/NumPy runtime. `num_workers` cố ý không nằm trong hash vì sample được seed theo index.
- Signature được ghi vào `loso_report.json`, `metrics.json` và config nằm **trong** `gesture_lstm.pt`; ship đọc lại checkpoint vừa ghi trước khi công bố thành công. `metrics.json.checkpoint_sha256` khóa metadata vào đúng bytes của file weights.
- LOSO evaluate lần hai bằng loader không shuffle/augment để gắn đúng prediction với đúng path. Derived worst-label/suspect lists chỉ in từ source rows, không lưu bản sao dễ drift.

## Safety behavior

- `load_checkpoint` raise khi feature version/contract/dimension lệch; không có opt-in bypass vì v1 (201 chiều) không thể nhận feature v3 (203 chiều) an toàn. `model_architecture_version` lạ luôn bị chặn.
- `Segment.duration` dùng `active_end_time`, không cộng idle `word_gap`; một detection blip không vượt default minimum.
- `vslr-check` bắt buộc đọc được recording plan + manifest trước MediaPipe; sai cwd/path không còn đường exit 0 giả. Nó có `LOI HE THONG` cho permission/cache/internal error và giữ nguyên message; không bảo người dùng quay lại clip tốt.
- Train chỉ dung sai lỗi chất lượng nguồn đã định danh bằng `ClipExtractionError` (không mở được, quá ngắn, thấy tay <10%) hoặc file biến mất. Permission/MediaPipe/runtime/programming error abort ngay.
- Train/check/realtime tự cấu hình console UTF-8 trên Windows.
- `--learning-rate` và mọi duration realtime phải finite; learning rate/duration > 0, tỷ lệ trong `(0, 1]`.

## Files changed

| File | Vai trò |
|---|---|
| `vsl3/labels.py` | parser manifest + Unicode NFC + collision gate dùng chung |
| `vsl3/console.py` | cấu hình UTF-8 dùng chung |
| `prepare_train.py` | manifest gate, cache validation, signature, LOSO predictions/diagnostics |
| `model.py` | checkpoint feature mismatch fail-closed + atomic checkpoint write |
| `realtime.py` | active duration, min-active hysteresis, shared preprocessing + reject policy |
| `check.py` | system-error status + shared labels/UTF-8 |
| `tests/test_core.py` | regression tests cho các invariant trên |

## Verification đã chạy

```text
python -m pytest -q
78 passed, 31 subtests passed in 5.87s

git diff --check
exit 0 (chỉ có cảnh báo line-ending LF→CRLF của Git trên Windows)
```

Smoke test tạm, không đụng `models/` workspace:

- cây 2 người × 2 nhãn × 1 clip, cache hợp lệ;
- `vslr-check`: exit 0, in “Tất cả clip đạt”;
- LOSO 2 fold: exit 0, 4 prediction rows, **không** ghi checkpoint;
- ship: exit 0, checkpoint load strict thành công, `features_version=2`, `model_architecture_version=1`, pooling `fwd_last_bwd_first`;
- report/metrics/checkpoint cùng signature, metrics giữ đúng checkpoint SHA-256 và ship in `PAIR OK`;
- cây thiếu toàn cục `Xin chào`: exit 1 trước MediaPipe, `model-dir` chưa được tạo;
- checkpoint workspace legacy: strict load bị chặn; không có override vì khác dimension/semantics.

## Things to watch

- `dataset/labels.txt` đã được chủ dự án chốt thành 30 nhãn V1. `vslr-init-dataset` tạo cây quay từ manifest này; không đổi thứ tự dòng sau khi đã train.
- Cache tạo trước 2026-08-23 sẽ miss một lần vì key mới thêm source SHA-256; đây là invalidation có chủ ý, không phải mất dữ liệu nguồn.
- `dataset/raw/` hiện là cây legacy phẳng, không có signer ID; không được đoán người để chạy LOSO.
- Checkpoint tracked vẫn là v1 và bị chặn mặc định; train lại sẽ ghi đè artifact nên chỉ làm sau khi chủ dự án xác nhận và backup.
- Chưa test camera vật lý hoặc dataset nhiều signer thật; smoke test dùng cache landmark tổng hợp để kiểm orchestration, không phải chất lượng model.
