# Spec — Hardening pipeline V1 trước khi quay/train

_Viết: 2026-08-28. Trạng thái: đang triển khai._

## Mục tiêu

Đóng các lỗ hổng còn lại trong luồng `recordings_v1 → landmarks → train/LOSO → realtime` cho
30 nhãn, 4 người (`P01`–`P04`), 6 clip mỗi cặp. Pipeline phải fail-closed trước MediaPipe khi
hợp đồng dataset sai, không tạo ghost landmark khi một tay bị che, và có cùng luật xử lý khi
đo clip offline với webcam.

## Hợp đồng dataset

- `dataset/labels.txt` là nguồn duy nhất của danh sách nhãn và class order.
- `dataset/recording_plan.json` chỉ chứa `schema_version`, `dataset_version`, `labels_file`,
  `people` và `clips_per_label`; không lặp lại danh sách nhãn.
- Init/check/train directory mode đọc cùng plan. Plan V1 cố định P01–P04 và 6 clip/cặp.
- Train từ chối trước MediaPipe nếu thiếu/overfull `(person, label)`, người ngoài plan, số
  người sai, clip trùng content hash giữa hai path, hoặc `clips_per_label <= 0`.
- `--video` là đường legacy một signer, không áp plan/LOSO.

## Landmark và segmentation

- Mỗi frame mang presence độc lập cho tay trái/phải. Landmark zero là missing, không phải quan
  sát hợp lệ.
- Chỉ bridge gap ngắn khi cùng bàn tay có observation hữu lệ ở hai đầu; gap dài hoặc gap một
  phía giữ missing. Resample/augment không được tạo ghost từ zero.
- Preprocessing chung nhận sequence + presence và được gọi từ extract video và realtime.
  Thay đổi representation phải bump `FEATURES_VERSION`, `FEATURE_DIM`, checkpoint/cache tests.
- Segment tracker có hysteresis/min-active evidence; detection blip hoặc đuôi no-hand không được
  mở/đưa vào model như một từ.
- Timestamp offline lấy từ FPS/video frame index, không dùng wall clock.

## Unknown và sentence evaluation

- Model vẫn closed-set 30 lớp. Không thêm `unknown` vào manifest/TTS.
- Có evaluator negative/idle/OOV và reject result; nếu chưa có calibration data thì trạng thái
  là `uncalibrated` và không tự bịa threshold. Rejected segment không TTS.
- `vslr-sentence` dùng chính `SegmentTracker`, preprocessing, classifier/reject policy; hỗ trợ
  một clip và batch CSV `person,clip,words`, kiểm nhãn theo checkpoint, exact sentence, WER,
  edit distance, count mismatch và false accept trên negative rows.

## Acceptance criteria

1. Unit tests bắt các invariant trên, gồm missing signer/empty directory, exact/overfull counts,
   duplicate bytes, short/long one-hand gaps, no ghost interpolation, shared offline/realtime
   path, idle blip/no-hand tail, checkpoint version, sentence FPS/WER/false-accept và calibration
   leakage.
2. `python -m pytest -q`, `git diff --check`, `python -m pip install -e . --no-deps` pass.
3. Init/check/train/sentence có `--help`/dry-run smoke. Synthetic end-to-end dùng TemporaryDirectory
   và không ghi `models/` hay đụng video/model người dùng.
4. Không train model thật, không claim accuracy/người mới khi chưa có đủ raw video.

## Thứ tự thực hiện

1. Contract plan/count/hash gate.
2. Presence-aware feature preprocessing.
3. Shared segment/classification path và hysteresis.
4. Reject/negative evaluator fail-closed.
5. Offline sentence CLI.
6. Checkpoint/signature/docs và full verification.
