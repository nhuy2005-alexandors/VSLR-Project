# As-built — Hardening pipeline V1

_Xong phần code: 2026-08-28. Chưa train model thật._

## Đã triển khai

- `dataset/recording_plan.json` là hợp đồng version 1, khóa `P01`–`P04` và 6 clip cho mỗi
  nhãn; không lặp danh sách nhãn. `dataset/labels.txt` vẫn là nguồn class order.
- Plan V1 kiểm tra kiểu JSON nguyên thủy, đúng người `P01`–`P04` và quota đúng số 6; không ép
  kiểu từ `6.9`, chuỗi hoặc boolean.
- Directory mode của init/check/train cùng đọc plan. Train/ship dừng trước MediaPipe nếu sai
  người, thiếu/overfull cặp, nhãn ngoài manifest hoặc exact duplicate bytes. `--video` giữ
  đường legacy một signer; quota zero bị parser từ chối.
- Feature contract v3 là 201 landmark values + 2 bit presence trái/phải = 203 chiều. Gap tay
  chỉ được bridge khi cùng tay có hai đầu quan sát và khoảng thời gian ngắn; gap dài/one-sided
  giữ zero. Resample và augmentation không nội suy qua missing. Z được làm group-local (pose
  theo hip midpoint, tay theo wrist), không coi hai nguồn z là một hệ tọa độ.
- `HolisticExtractor.extract_video`, realtime và `vslr-sentence` dùng `preprocess_sequence`.
  SegmentTracker có min-active hysteresis theo giây; `classify_segment` loại no-hand tail trước
  khi đưa sequence vào model.
- Reject policy thiếu calibration là `uncalibrated` và fail-closed; negative/idle/OOV rows
  trong sentence CSV được tính false-accept. `fit_reject_policy` từ chối rows thuộc LOSO/test,
  bắt buộc identity/content receipt và hai category âm `idle_stationary` + `oov_motion`; motion
  gate chưa được bịa khi chưa có dữ liệu.
- `vslr-sentence` hỗ trợ single clip hoặc CSV batch (`person,clip,words`, dấu `|`), in từng
  segment accepted/rejected và tính exact sentence, WER, edit distance, count mismatch,
  false accept. Timestamp offline lấy FPS video.
- Checkpoint cũ lệch version/dimension bị từ chối không có bypass; chỉ checkpoint v3 có receipt
  đầy đủ mới được dùng cho realtime/sentence.

## Chưa thể verify

- Workspace chưa có đủ 720 raw V1 clip, clip câu và negative/idle/OOV calibration; chưa có LOSO,
  sentence accuracy/WER hay claim tổng quát cho người mới.
- Checkpoint tracked được tạo theo feature contract cũ; loader v3 fail-closed. Chưa train và
  không ghi đè `models/gesture_lstm.pt`, `models/metrics.json` hoặc `models/backups/`.
- Webcam thật chưa chạy trong môi trường này; chỉ có unit/synthetic tests.

## Kiểm thử

```powershell
python -m pytest -q
git diff --check
python -m pip install -e . --no-deps
```
