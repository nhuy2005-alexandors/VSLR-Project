# Checkpoint — VSLR

_Cập nhật: 2026-08-22_

## Done

- **Pipeline feature**: `src/prototype_3_gestures/vsl3/features.py` — MediaPipe Holistic, 67 landmark (25 pose + 21×2 hand) × 3 = `FEATURE_DIM` 201, resample `SEQUENCE_LENGTH` 60, normalize theo trung điểm vai, augment không mirroring. Thêm `FEATURES_VERSION = 1` dùng làm khóa cache.
- **Model**: `src/prototype_3_gestures/vsl3/model.py` — `GestureLSTM` (BiLSTM, 242.581 tham số ở 3 nhãn). **Sửa pooling**: `out[:, -1, :]` → `cat([out[:, -1, :h], out[:, 0, h:]])`; trước đó một nửa biểu diễn chỉ là hàm của một frame. Checkpoint thiếu khóa `pooling` mặc định về `legacy_last_step` nên artifact cũ vẫn chạy đúng như lúc train.
- **Train CLI tái cấu trúc**: `prepare_train.py` — `Clip(label, person, path)`, `discover_clips` quét `DIR/<person>/<gesture>/`, cache landmark per-clip, `GestureDataset` augment on-the-fly, `signer_split` theo NGƯỜI. Hai chế độ: `--loso` chỉ đo, mặc định chỉ sản xuất. Xóa `build_dataset`, `source_holdout_split`, `best_state`/`best_epoch`/`final_fit_epochs`, khối refit.
- **Tests**: `tests/test_core.py` — **24 test**, chạy `python -m pytest` ngày 2026-08-22: **24 passed**. Kiểm bằng mutation testing: 7/7 mutation cố ý phá từng bản sửa đều bị test bắt.
- **Docs**: spec `docs/specs/pipeline-restructure.md` (rev 2 sau `spec-critic`: BLOCKED → 5 blocker đã trả lời), as-built `docs/technical_specs/pipeline-restructure.md`, ADR-003/004/005, 4 gotcha mới.
- **Vòng `reviewer` xong**: verdict BLOCKED → 2 blocker + 6 should-fix + nits, **đã xử lý hết**. Reviewer xác nhận không có đường leakage, pooling đúng khi đối chiếu `h_n`, mode tách đúng. Hai blocker đều ở nhánh xử lý lỗi: nhãn mất hết clip biến khỏi class set, và cache hỏng bị tính là clip quay tệ. Bảng đầy đủ trong as-built doc.
- **Thêm sau review**: `data_fingerprint` + `created_at` + `seed`/`lr`/`batch_size` trong cả hai report (ghép được hai file), `--num-workers` (augment ~79 s/epoch ở 540 clip), `macro_mean_accuracy` + `pooled_accuracy`, cảnh báo khi checkpoint thiếu `pooling` hoặc lệch `features_version`.
- **Spec quay rev 2**: `docs/specs/dataset-recording.md` — 27 nhãn × 5 người × 4 clip, quay tuần tự P1→P5, bắt buộc kiểm sau P1.
- **Khảo sát repo tham chiếu**: `dataset/legacy/` chính là dataset của `photienanh/Vietnamese-Sign-Language-Recognition` — 4362 video / **3315 nhãn** (2765 nhãn chỉ 1 clip; nhóm 3 clip là biến thể phương ngữ B/N/T, không phải 3 người). Model đó 2764 class, split random trên 1001 bản augment mỗi clip → leakage; notebook 0 output, không có con số accuracy nào.

## In progress

- Chưa commit gì. `src/` + `tests/` modified (+911 / −231 dòng); `docs/`, `CLAUDE.md`, `.claudeignore`, `models/backups/` untracked.

## Next

1. Chốt danh sách ~27 nhãn (chủ dự án lên sau) — chặn buổi quay, không chặn code.
2. Train lại `models/gesture_lstm.pt` bằng pipeline mới. **Cần chủ dự án đồng ý** vì ghi đè artifact đang track; backup sang `models/backups/` trước. Nhớ `--num-workers 4`.
3. Sửa `realtime.py` bug tách một cử chỉ thành hai từ (`realtime.py:139-141`) — out of scope của spec pipeline, làm riêng.
4. Thêm `time_stretch` vào `augment_sequence` — sau khi pipeline đứng.
5. Quay P1 (108 clip) rồi kiểm `hand_frame_ratio` trước khi hẹn P2.
6. Cân nhắc ghi `pooling` vào `model_state` như buffer (reviewer đề xuất) nếu sau này có nhiều checkpoint cùng lưu hành.

## Verify

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m pytest                                      # 24 passed
vslr-train --data-dir <cây P1..Pn> --loso --num-workers 4   # đo; ghi models/loso_report.json
vslr-camera --no-tts                                  # camera, chưa phát giọng
```

`dataset/raw/` hiện phẳng (`<cử chỉ>/*.mov`) nên `--data-dir dataset/raw` báo `No .mov/.mp4 clips under datasetaw` ở **cả hai** chế độ — đúng thiết kế, không đoán ID người. Bộ 27 clip hiện tại chỉ train được qua `--video` (chế độ một-người). Cần cây `DIR/<person>/<gesture>/` từ buổi quay mới.

## Gotchas hit

- Checkpoint trong `models/` vẫn là model **1 epoch, pooling cũ** — chưa train lại. Không được dùng nó để báo bất kỳ số nào.
- `dataset/raw/` không có ID người → không đoán mapping `1-3/4-6/7-9`, sẽ tạo ra LOSO giả.
- Chi tiết: `docs/GOTCHAS.md` (8 bẫy).

Xem thêm: `DECISIONS.md` (5 ADR), `GOTCHAS.md`, `specs/`, `technical_specs/`.
