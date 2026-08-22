# Decisions (ADR) — VSLR 3 cử chỉ

Quyết định kiến trúc + LÝ DO. Tích lũy, không xóa.

> Hai ADR dưới đây được suy ra từ code và metadata đang có trong repo (2026-08-21), không phải từ ghi chép gốc. Phần **Reasoning** có đánh dấu `[suy luận]` là chỗ cần chủ dự án xác nhận hoặc sửa lại.

## ADR-001: Split theo source video, không split theo sample

> **SUPERSEDED bởi ADR-003 (2026-08-22).** Cơ chế `source_holdout_split` đã bị xóa. Lý do chống-leakage-augmentation dưới đây vẫn đúng và được ADR-003 giữ, nhưng ở đơn vị **người** thay vì **clip**.

- **Date**: trước 2026-08-21 (suy ra từ `prepare_train.py:75`)
- **Context**: mỗi clip gốc sinh ra 120 bản augmentation. Nếu split random theo sample thì bản augment của cùng một clip nằm cả ở train và val → val accuracy vô nghĩa.
- **Decision**: `source_holdout_split(y, source_ids, seed)` giữ trọn **một clip nguồn mỗi nhãn** về phía validation; mọi augmentation của clip đó không bao giờ vào train. Test `test_source_holdout_split_never_leaks_a_video_between_train_and_val` khóa hành vi này.
- **Reasoning**: chống leakage augmentation — đây là dạng leakage khó thấy nhất với dataset nhỏ. Ghi rõ trong `models/metrics.json` (khóa `warning`): kết quả vẫn chỉ là "prototype estimate", không phải claim accuracy thực tế, vì chỉ có 3 clip nguồn/nhãn và khả năng cùng một người ký, cùng một setup.
- **Trade-offs**: val set chỉ còn 363 sample từ 3 clip nguồn → phương sai cao. Checkpoint cuối được refit lại trên toàn bộ clip nguồn (`final_fit_epochs`) để dùng cho camera, nên checkpoint đó **không** còn holdout tương ứng.

## ADR-002: Landmark + BiLSTM, không dùng CNN trên pixel thô

- **Date**: trước 2026-08-21 (suy ra từ `vsl3/features.py`, `vsl3/model.py`)
- **Context**: nhận cử chỉ động (chuỗi thời gian), dataset nhỏ (hiện có 27 clip, 9 clip mỗi nhãn).
- **Decision**: MediaPipe Holistic trích 67 landmark (25 pose + 21×2 hand), normalize, resample về 60 frame → `GestureLSTM` (BiLSTM) trên vector 201 chiều.
- **Reasoning**: `[suy luận]` landmark đã loại phần lớn nhiễu nền/ánh sáng/trang phục, nên số tham số cần học nhỏ hơn nhiều so với CNN pixel — phù hợp với bộ dữ liệu chỉ có 27 clip gốc.
- **Trade-offs**: phụ thuộc chất lượng MediaPipe; frame không thấy tay bị trim (`hand_frame_ratio` trong `metrics.json` chạy từ 0.59 đến 0.91). Mất thông tin biểu cảm mặt và hình dạng bàn tay chi tiết không có trong 67 điểm.

## ADR-003: Split theo NGƯỜI, không theo clip (thay ADR-001)

- **Date**: 2026-08-22
- **Context**: ADR-001 giữ ra một *clip* mỗi nhãn. Với dataset nhiều người, clip test khi đó thuộc người đã có mặt trong train → val đo "cùng người, lần quay khác", không phải "người mới". Mà "chạy được với người mới" mới là thứ project cần báo cáo. `source_ids` cũ là chỉ số clip (`enumerate` trong `build_dataset`), nên không có cách nào suy ra người.
- **Decision**: data model mang `Clip(label, person, path)`; người suy từ đường dẫn `DIR/<person>/<gesture>/`. `signer_split(clips, val_person)` giữ **toàn bộ clip của một người** về phía val. Split xảy ra trên danh sách clip **gốc, trước augment**, nên augment của người đang test không thể vào train theo cấu trúc — không chỉ theo test.
- **Reasoning**: ADR-001 đúng về nguyên tắc (không để augment của cùng nguồn nằm hai phía) nhưng sai đơn vị. Đổi đơn vị từ clip sang người vừa giữ tính chất cũ vừa cho phép leave-one-signer-out.
- **Trade-offs**: `--video LABEL=PATH` không mã hóa người nên bị gán `person="unknown"` và **không bao giờ** chạy được LOSO — 27 clip hiện có trong `dataset/raw/` nằm trong diện này. Trade-off cũ ở ADR-001 (checkpoint refit trên mọi clip để dùng cho camera) vẫn được giữ, bằng cách khác: xem ADR-004.

## ADR-004: Hai chế độ tách hẳn, không early stopping ở đâu cả

- **Date**: 2026-08-22
- **Context**: một lệnh vừa đo vừa sản xuất checkpoint chính là nguyên nhân của cả hai sự cố trong `docs/GOTCHAS.md`: lần train trên chính clip test, và lần ship checkpoint train đúng 1 epoch. `history` trong `metrics.json` cho thấy `val_loss` thấp nhất ở epoch 1 và 15 epoch sau không phá được, `val_accuracy` = 1.0 suốt → `best_epoch` chọn theo `val_loss` là chọn nhiễu.
- **Decision**: `vslr-train --loso` **chỉ đo** (ghi `models/loso_report.json`, không ghi checkpoint); không có cờ thì **chỉ sản xuất** (train trên toàn bộ người, không val, không early stopping, ship weight cuối). Xóa hẳn `best_state`, `best_epoch`, `final_fit_epochs` và khối refit-from-random-init. `--epochs` là hằng chọn trước.
- **Reasoning**: không có validation thì không tồn tại `best_epoch` để chọn sai — bẫy #3 trở thành bất khả thi về cấu trúc, không phải "đã vá". Và **`--loso` cũng không được early stop**: người giữ ra trong LOSO chính là tập test, nên dừng hay chọn epoch theo nó là tune tham số trên test. Curve val vẫn ghi vào report để xem, không dùng để chọn.
- **Trade-offs**: `--epochs` không được tune (tune tử tế cần inner split lồng trong mỗi fold — chưa làm). Checkpoint ship ra không kèm con số accuracy nào; `metrics.json` nói thẳng điều đó và trỏ sang `loso_report.json`.

## ADR-005: Pooling BiLSTM lấy fwd-cuối + bwd-đầu, giữ đường lùi cho checkpoint cũ

- **Date**: 2026-08-22
- **Context**: `GestureLSTM.forward` lấy `out[:, -1, :]`. Trên LSTM hai chiều, `out[:, -1, hidden:]` là hidden của chiều ngược tại frame cuối — chiều đó đi từ cuối về đầu nên tại đó nó mới xử lý **một** frame. Đo trực tiếp: đổi 5 trong 6 frame đầu, nửa đó không thay đổi. Tức 96 trong 192 chiều đưa vào `head` là hàm của một frame.
- **Decision**: `pool_sequence` trả `cat([out[:, -1, :hidden], out[:, 0, hidden:]])`. Thêm khóa `pooling` vào config checkpoint; `load_checkpoint` mặc định về `legacy_last_step` khi checkpoint **không có** khóa đó.
- **Reasoning**: không đổi số chiều nên `head` và config không đổi. Đường lùi là bắt buộc: `models/gesture_lstm.pt` đang được track trong git được train với pooling cũ, load nó bằng pooling mới sẽ dự đoán sai **âm thầm** — tệ hơn lỗi rõ ràng.
- **Trade-offs**: một nhánh `if` sống mãi trong `pool_sequence`. Bỏ được khi không còn checkpoint legacy nào đang dùng.

## ADR-006: Augment bằng time WARP phi tuyến, không phải time stretch

- **Date**: 2026-08-22
- **Context**: `augment_sequence` chỉ có phép hình học (xoay ±5°, scale ±6%, jitter σ=0,012, crop ±8%). Dự định ban đầu là thêm "time stretch" để phủ chuyện người múa nhanh/chậm khác nhau.
- **Decision**: thêm `time_warp_sequence(sequence, rng, max_warp=0.30)` dùng phép biến đổi trục thời gian `t → t + w·sin(πt)`, gọi trong `augment_sequence` sau bước crop. **Không** thêm time stretch toàn cục.
- **Reasoning**: time stretch toàn cục là **no-op** ở pipeline này. `extract_video` trim về vùng thấy tay rồi resample về đúng `SEQUENCE_LENGTH` 60 frame, nên thời lượng tuyệt đối đã bị chuẩn hóa mất — múa 2 giây và múa 5 giây ra cùng một chuỗi 60 frame. Thứ **thật sự** khác nhau giữa người ký là *nhịp bên trong* cử chỉ: ai dừng lâu ở đâu. Đó là cái `t → t + w·sin(πt)` bẻ: hai đầu bị ghim (t=0 → 0, t=1 → 1) nên không đổi vị trí bắt đầu/kết thúc, và ánh xạ còn đơn điệu khi `|w| < 1/π ≈ 0,318` nên không bao giờ chạy ngược thời gian. `max_warp = 0.30` giữ đạo hàm nhỏ nhất ở `1 − 0,30π ≈ 0,058 > 0`. Đo 500 seed: đơn điệu 100%, điểm giữa của một ramp thời gian chạy từ 0,211 đến 0,807 (gốc 0,508).
- **Trade-offs**: không bump `FEATURES_VERSION` vì cache chỉ giữ landmark gốc, không giữ bản augment — nhưng **phân phối augment đã đổi**, nên bất kỳ con số đo trước thay đổi này đều không so sánh trực tiếp được với con số sau. Hiện chưa có con số nào nên không mất gì.

## ADR-007: Manifest nhãn là cổng bắt buộc của `--data-dir`

- **Date**: 2026-08-22
- **Context**: suy nhãn từ cây chỉ phát hiện được nhãn đang tồn tại. Nếu cả 5 người đều quên cùng một nhãn, cây 26 lớp vẫn tự nhất quán và LOSO chạy xanh dưới tên bài toán 27 lớp.
- **Decision**: `dataset/labels.txt` là nguồn sự thật; `vslr-train --data-dir` đọc `--labels-file`, chuẩn hóa NFC và từ chối cả nhãn thiếu toàn cục lẫn thư mục ngoài manifest **trước MediaPipe**. Thứ tự dòng trong manifest là thứ tự class index. `--video` legacy giữ self-contained vì nó cố ý train một tập con một người.
- **Trade-offs**: danh sách 27 nhãn đề xuất phải được chốt trước khi train cây mới. Đây là blocker có chủ ý: fail sớm tốt hơn sinh artifact sai số lớp.

## ADR-008: Chữ ký đầy đủ mới được ghép LOSO với checkpoint ship

- **Date**: 2026-08-22
- **Context**: cùng `data_fingerprint`/epochs/seed nhưng khác augmentation, batch size hay learning rate vẫn là hai quy trình train khác nhau; gọi accuracy của một quy trình là số của quy trình kia là sai.
- **Decision**: cả hai artifact ghi `training_signature`, SHA-256 trên data fingerprint, **thứ tự nhãn**, epochs, số augment, batch size, learning rate, seed, pooling và feature version. Ship tự đọc report cạnh nó và in `PAIR OK` hoặc `PAIR MISMATCH`.
- **Trade-offs**: đổi một tham số buộc chạy lại LOSO nếu muốn gắn con số với checkpoint mới. `num_workers` không nằm trong hash vì sample được seed theo index và số worker không đổi dữ liệu/optimizer steps.

## ADR-009: Checkpoint lệch feature version fail-closed

- **Date**: 2026-08-22
- **Context**: checkpoint v1 nhận tensor đúng shape từ extractor v2 nên PyTorch không báo, nhưng phân phối landmark khác và chính loader biết predictions không đáng tin. Warning vẫn cho phép demo tiếp tục như thể hợp lệ.
- **Decision**: `load_checkpoint` raise mặc định khi `features_version` khác (thiếu khóa = v1). `vslr-camera --allow-incompatible-model` là opt-in tường minh chỉ cho demo legacy tạm thời; checkpoint mới v2 chạy không cần cờ.
- **Trade-offs**: `vslr-camera --no-tts` trên clean checkout hiện bị chặn cho tới khi chủ dự án cho phép train lại artifact. Không tự ghi đè model đang track.
