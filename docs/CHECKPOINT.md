# Checkpoint — VSLR

_Cập nhật: 2026-08-22_

## Done

- **Feature extraction v2**: MediaPipe Holistic, 67 landmark × 3 = 201 features, sequence 60; mỗi file video có instance MediaPipe riêng; cache khóa theo path/mtime/dimension/`FEATURES_VERSION=2`.
- **Model**: BiLSTM pooling đúng `fwd-last + bwd-first`; checkpoint giữ metadata `pooling`/`features_version`. Artifact lệch feature version giờ bị chặn mặc định, chỉ opt-in legacy mới chạy.
- **Train pipeline**: `Clip(label, person, path)`, signer split trước augment, augment on-the-fly deterministic; `--loso` chỉ đo, mode mặc định chỉ ship; không early stopping/refit.
- **Cổng nạp video**: `vslr-train --data-dir` đọc manifest, chuẩn hóa NFC, chặn nhãn thiếu toàn cục và thư mục ngoài manifest trước MediaPipe. Class order lấy từ manifest.
- **Cache hardening**: cache đọc được nhưng sai shape/NaN/stats cũng bị bỏ và extract lại.
- **LOSO diagnostics**: report có một prediction row cho mỗi test clip; console in fold tệ, nhãn tệ/confusion và clip sai có hand ratio thấp.
- **Artifact pairing**: `training_signature` bao phủ data, thứ tự nhãn và toàn bộ hyperparameter train liên quan; ship in `PAIR OK`/`PAIR MISMATCH` với report cạnh nó.
- **Realtime**: `SegmentTracker` chặn một cử chỉ chậm thành hai từ; ngưỡng theo giây; min-duration loại idle word-gap khỏi active span.
- **On-site QA**: `vslr-check` dùng chung manifest/cache, phân biệt lỗi quay với `LOI HE THONG`, không train hay ghi `models/`.
- **Windows**: train/check/realtime tự cấu hình stdout/stderr UTF-8.
- **Verify hiện tại**: `python -m pytest` → **56 passed**; smoke test cây 2 người × 2 nhãn chạy check → LOSO → ship thành công; signature khớp; checkpoint mới load strict v2.

- **LOSO chỉ ra nhãn nào sai**: `evaluate()` (`prepare_train.py:317`) trả một dòng mỗi clip; `folds[].predictions` mang `{video, person, label, predicted, confidence, correct}`; console in 5 nhãn tệ nhất kèm cặp confusion và clip vừa-sai-vừa-quay-tệ. `confidence` là softmax-max nên cùng thang với `--confidence` của demo. `build_eval_loader()` (`:305`) dùng chung cho train và report, không augment không shuffle — đó là điều kiện để dòng i ứng clip i.
- **Khóa dẫn xuất KHÔNG lưu vào report**: `per_label_accuracy` / `worst_labels` / `suspect_clips` là hàm thuần của `predictions[]` + `extraction[]`, chỉ in console. Lưu bản sao là tạo chỗ để trôi khỏi nguồn.
- **`suspect_clips` dùng ngưỡng tuyệt đối 0,5**, không phải "dưới trung vị": một nửa tập nào cũng dưới trung vị của chính nó nên luật trung vị không bao giờ trả về danh sách rỗng.

## In progress / working tree

- Nhánh `pipeline-signer-split` có **6 commit** sau `main`: `70b9ff8`, `293dee5`, `29e5cd7`, `ad24b7a`, `097516f`, `c231e81`; chưa merge.
- Các sửa readiness/LOSO diagnostics của lượt hiện tại đang **uncommitted** trong source/tests/docs.
- Có chủ ý không đụng: `models/gesture_lstm.pt`, `models/metrics.json`, `models/backups/` và 18 clip `.mov` untracked của người dùng.
- `dataset/labels.txt` là template 27 nhãn đề xuất, **chưa chốt**; ba câu hỏi VSL nằm ngay trong file.

## Next

0. **`vslr-sentence`** — đo phần ghép câu offline. Điều kiện tiên quyết (ngưỡng giây, `Segment.duration`) đã xong; còn phải kéo lọc `min_seconds` + ngưỡng confidence + ghép câu ra khỏi `realtime.main()` để hai đường dùng chung một bản. Đây là thứ duy nhất đo được mục tiêu thật của dự án.

1. Chủ dự án + người thạo VSL chốt `dataset/labels.txt` trước buổi quay.
2. Nạp video theo `dataset/raw/P1/<Nhãn tiếng Việt>/<Nhãn> 01.mov`, đủ 4 clip mỗi nhãn; chạy `vslr-check` ngay sau P1.
3. Sau ít nhất 2 người đủ mọi nhãn, chạy LOSO. Khi đủ người, giữ report có `training_signature` cuối.
4. Chỉ sau khi chủ dự án đồng ý ghi đè artifact: backup model hiện tại rồi chạy mode ship với **cùng** config/signature.
5. Test `vslr-camera --no-tts` thật bằng checkpoint v2 mới; checkpoint hiện tại v1 bị chặn đúng thiết kế.
6. Lỗ hổng còn lại của `record-to-test.md`: `vslr-sentence` đo câu offline từ `sentences.csv`.

## Verify

```powershell
python -m pytest

# Quay/nạp xong: nghiệm thu trước, không train
vslr-check --data-dir dataset/raw --labels-file dataset/labels.txt --clips-per-label 4

# Đo, không ghi checkpoint
vslr-train --data-dir dataset/raw --labels-file dataset/labels.txt --loso --num-workers 4

# Ship chỉ sau khi duyệt ghi đè; phải dùng cùng config với LOSO
vslr-train --data-dir dataset/raw --labels-file dataset/labels.txt --num-workers 4
```

`dataset/raw/` hiện vẫn phẳng (`<slug>/*.mov`) và không có signer ID, nên `--data-dir dataset/raw` từ chối đúng thiết kế. Bộ legacy chỉ train được qua `--video`; tuyệt đối không gán giả signer để lấy LOSO.

## Gotchas quan trọng

- Checkpoint tracked là model 1 epoch, pooling/landmark v1; không có accuracy báo cáo được và không chạy mặc định với extractor v2.
- Chỉ ghép accuracy với checkpoint khi `training_signature` khớp.
- Data quality issue phải sửa file gốc + backup, không chỉ lọc in-memory.
- Chi tiết tích lũy: `docs/GOTCHAS.md`; quyết định: `docs/DECISIONS.md`; as-built mới: `docs/technical_specs/video-ingest-readiness.md`.
