# As-built — Cổng nạp video an toàn + chẩn đoán LOSO

_Xong: 2026-08-22. Spec liên quan: `pipeline-restructure.md`, `record-to-test.md` rev 2. Verify trước reviewer: **56 passed**._

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
        └─ ship   → checkpoint v2 + metrics + same signature
```

- `vslr-check` dùng cùng parser nhãn và cache, nên check xong train ăn cache ngay.
- `vslr-train --data-dir` lấy class-index order từ thứ tự dòng trong manifest, không từ thứ tự đi cây.
- Nhãn thiếu ở **tất cả** người hoặc thư mục slug ngoài manifest đều exit 1 trước khi tạo `model-dir`.
- `_read_cache_entry()` yêu cầu sequence `(60, 201)`, finite, frame counts hợp lệ và hand ratio trong range; sai thì xóa và extract lại.
- `training_signature` bao phủ data fingerprint, thứ tự nhãn, epochs, augment, batch size, learning rate, seed, pooling và feature version. `num_workers` cố ý không nằm trong hash.
- LOSO evaluate lần hai bằng loader không shuffle/augment để gắn đúng prediction với đúng path. Derived worst-label/suspect lists chỉ in từ source rows, không lưu bản sao dễ drift.

## Safety behavior

- `load_checkpoint` raise khi feature version lệch. `--allow-incompatible-model` là opt-in legacy tường minh.
- `Segment.duration` dùng `active_end_time`, không cộng idle `word_gap`; một detection blip không vượt default minimum.
- `vslr-check` có `LOI HE THONG` cho permission/cache/internal error và giữ nguyên message; không bảo người dùng quay lại clip tốt.
- Train/check/realtime tự cấu hình console UTF-8 trên Windows.
- `--learning-rate` phải finite và > 0; tỷ lệ phải trong `(0, 1]`.

## Files changed

| File | Vai trò |
|---|---|
| `vsl3/labels.py` | parser manifest + Unicode NFC dùng chung |
| `vsl3/console.py` | cấu hình UTF-8 dùng chung |
| `prepare_train.py` | manifest gate, cache validation, signature, LOSO predictions/diagnostics |
| `model.py` | checkpoint feature mismatch fail-closed |
| `realtime.py` | active duration + explicit legacy opt-in |
| `check.py` | system-error status + shared labels/UTF-8 |
| `tests/test_core.py` | regression tests cho các invariant trên |

## Verification đã chạy

```text
python -m pytest
56 passed in 5.44s

git diff --check
exit 0 (chỉ có cảnh báo line-ending LF→CRLF của Git trên Windows)
```

Smoke test tạm, không đụng `models/` workspace:

- cây 2 người × 2 nhãn × 1 clip, cache hợp lệ;
- `vslr-check`: exit 0, in “Tất cả clip đạt”;
- LOSO 2 fold: exit 0, 4 prediction rows, **không** ghi checkpoint;
- ship: exit 0, checkpoint load strict thành công, `features_version=2`, pooling `fwd_last_bwd_first`;
- report/metrics cùng signature và ship in `PAIR OK`;
- cây thiếu toàn cục `Xin chào`: exit 1 trước MediaPipe, `model-dir` chưa được tạo;
- checkpoint workspace legacy: strict load bị chặn; explicit override load `legacy_last_step` với cảnh báo.

## Things to watch

- `dataset/labels.txt` hiện là đề xuất 27 nhãn, chưa được người thạo VSL chốt. Code sẵn sàng nhận dữ liệu nhưng buổi quay chưa nên bắt đầu trước quyết định đó.
- `dataset/raw/` hiện là cây legacy phẳng, không có signer ID; không được đoán người để chạy LOSO.
- Checkpoint tracked vẫn là v1 và bị chặn mặc định; train lại sẽ ghi đè artifact nên chỉ làm sau khi chủ dự án xác nhận và backup.
- Chưa test camera vật lý hoặc dataset nhiều signer thật; smoke test dùng cache landmark tổng hợp để kiểm orchestration, không phải chất lượng model.
