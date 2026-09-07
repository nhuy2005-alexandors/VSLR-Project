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
- **Trade-offs**: manifest phải được chốt trước khi train cây mới. Blocker này đã được giải quyết cho V1 bởi ADR-013; fail sớm vẫn tốt hơn sinh artifact sai số lớp.

## ADR-008: Chữ ký đầy đủ mới được ghép LOSO với checkpoint ship

> **ĐƯỢC SIẾT bởi ADR-010 (2026-08-23).** Bản này mới buộc hai file JSON, chưa buộc chính weights.

- **Date**: 2026-08-22
- **Context**: cùng `data_fingerprint`/epochs/seed nhưng khác augmentation, batch size hay learning rate vẫn là hai quy trình train khác nhau; gọi accuracy của một quy trình là số của quy trình kia là sai.
- **Decision**: cả hai artifact ghi `training_signature`, SHA-256 trên data fingerprint, **thứ tự nhãn**, epochs, số augment, batch size, learning rate, seed, pooling và feature version. Ship tự đọc report cạnh nó và in `PAIR OK` hoặc `PAIR MISMATCH`.
- **Trade-offs**: đổi một tham số buộc chạy lại LOSO nếu muốn gắn con số với checkpoint mới. `num_workers` không nằm trong hash vì sample được seed theo index và số worker không đổi dữ liệu/optimizer steps.

## ADR-009: Checkpoint lệch feature version fail-closed

- **Date**: 2026-08-22
- **Context**: checkpoint v1 nhận tensor đúng shape từ extractor v2 nên PyTorch không báo, nhưng phân phối landmark khác và chính loader biết predictions không đáng tin. Warning vẫn cho phép demo tiếp tục như thể hợp lệ.
- **Decision**: `load_checkpoint` raise khi `features_version`, feature contract hoặc input dimension khác (thiếu khóa = v1). Bỏ hoàn toàn bypass incompatible vì checkpoint cũ 201 chiều không thể nhận feature v3 203 chiều một cách đáng tin; realtime/sentence không quảng cáo đường fallback.
- **Trade-offs**: `vslr-camera --no-tts` trên clean checkout hiện bị chặn cho tới khi chủ dự án cho phép train lại artifact. Không tự ghi đè model đang track.

## ADR-010: Pair accuracy bằng hợp đồng ba artifact, không chỉ hai JSON

- **Date**: 2026-08-23
- **Context**: `loso_report.json` và `metrics.json` có thể cùng signature trong khi `gesture_lstm.pt` là file cũ/bị thay hoặc run bị ngắt giữa các lần ghi. Khi signature không nằm trong checkpoint, `PAIR OK` không nói gì về weights thật.
- **Decision**: ghi `training_signature` vào config bên trong checkpoint, đọc lại checkpoint sau khi save, và chỉ in `PAIR OK` khi report/metrics/checkpoint cùng signature. Metrics giữ `checkpoint_sha256`; checkpoint được ghi staging rồi atomic replace. Signature thêm augmentation/model/training-recipe version, kiến trúc, optimizer/loss constants, device và runtime.
- **Trade-offs**: đổi device/runtime hoặc recipe buộc chạy lại LOSO để ghép số; đây là fail-closed có chủ ý. `num_workers` vẫn bị loại vì không đổi sample hay optimizer step.

## ADR-011: Cổng ingest fail-closed theo loại lỗi

- **Date**: 2026-08-23
- **Context**: thiếu manifest từng vẫn exit 0; `.strip()` từng cho directory sai khoảng trắng qua cổng; hai raw directory NFC/NFD từng bị gộp; `except Exception` từng biến permission/runtime error thành một clip xấu được dung sai.
- **Decision**: `vslr-check` bắt buộc manifest trước MediaPipe; tên directory chỉ NFC, không strip, và collision raw-name của cùng signer bị từ chối. Chỉ `ClipExtractionError` (ba lỗi nguồn video đã định danh) hoặc file biến mất được đưa vào quota clip lỗi; mọi lỗi hệ thống khác abort.
- **Trade-offs**: một lỗi hạ tầng nhỏ dừng cả train và cần sửa rồi resume từ cache. Đổi lại pipeline không thể ship artifact từ dataset bị giảm âm thầm vì lỗi máy/quyền/phần mềm.

## ADR-012: Nội dung video, không phải timestamp, định danh dữ liệu

- **Date**: 2026-08-23
- **Context**: công cụ copy/restore có thể thay bytes nhưng giữ mtime; cache và `data_fingerprint` cũ chỉ dùng path+mtime nên trả landmark cũ và giữ nguyên training signature cho dữ liệu mới.
- **Decision**: cache key thêm SHA-256 bytes nguồn, size và mtime nanosecond; `data_fingerprint` dùng SHA-256 trên từng `(person, label, path, mtime_ns, size, source_sha256)`. Loader cũng kiểm `model_architecture_version`; thiếu khóa = topology gốc v1, version lạ fail-closed.
- **Trade-offs**: mỗi check/train phải đọc bytes video để hash và cache cũ miss một lần. Chi phí I/O này được chấp nhận để LOSO/ship không thể nhận nhầm landmark hoặc chữ ký của video khác.

## ADR-013: Manifest V1 có 30 nhãn; cây quay được sinh từ manifest

- **Date**: 2026-08-24
- **Context**: cây `dataset/raw/` hiện chứa bộ legacy không có signer ID. Tạo thư mục thủ công cho 4 người × 30 nhãn dễ thiếu cặp, sai Unicode hoặc trộn dữ liệu cũ. Số lớp tương lai có thể là 20, 25 hay hơn 30 nên không được viết cứng vào code.
- **Decision**: chốt thứ tự 30 nhãn V1 trong `dataset/labels.txt`. `vslr-init-dataset` đọc manifest và tạo `DIR/P01..P04/<nhãn>/`; mặc định 6 clip mỗi cặp nhưng chỉ tạo thư mục, không tạo placeholder, di chuyển hoặc ghi đè video. Bộ mới dùng `dataset/recordings_v1`; label và person được pipeline suy chính xác từ đường dẫn.
- **Trade-offs**: đổi tập nhãn phải dùng manifest/phiên bản dữ liệu mới. Bốn người đủ để chạy LOSO phục vụ đánh giá prototype và demo, nhưng chưa đủ để tuyên bố tổng quát hóa mạnh cho người ký mới.

## ADR-014: Activity của cử chỉ khác với hand presence

- **Date**: 2026-09-03
- **Context**: webcam và video từ điển vẫn thấy hai bàn tay khi người hạ tay dọc thân. Presence-only
  làm segment không đóng và thường chạm cap 5 giây.
- **Decision**: dùng cổ tay cao hơn đường hông trung bình ít nhất 0,5 shoulder-width làm activity;
  pose thiếu thì fallback presence. State machine dùng activity, model input vẫn giữ presence thật.
  Giữ rolling context tối đa 1,0 giây trước và 0,5 giây sau active.
- **Trade-offs**: camera crop mất hông không được lợi từ gate và dùng hành vi presence cũ. Ngưỡng là
  heuristic đã kiểm trên 450 sequence train, 25 clip P01 và 3 clip ngoài; vẫn cần nghiệm thu webcam
  nhiều người trước khi coi là ổn định.

## ADR-015: Khóa bộ 24 nhãn 4 người ký, loại bỏ hoàn toàn "Hẹn gặp lại" để ngăn data leakage

- **Date**: 2026-09-07
- **Context**: Tại nhãn `Hẹn gặp lại`, phát hiện 2 clip của P04 trùng 100% hash byte với clip của P02 (`001.MOV` vs `004.mov`, `002.MOV` vs `003.mov`). Trong đánh giá chéo Leave-One-Signer-Out (LOSO), nếu giữ P04 thì khi fold P04 được kiểm thử, mô hình đã học clip của P04 qua P02 lúc huấn luyện, gây rò rỉ dữ liệu (data leakage) nghiêm trọng.
- **Decision**: Loại bỏ hoàn toàn nhãn `Hẹn gặp lại` khỏi tập nhãn; bảo lưu nguyên vẹn thứ tự và mã hóa NFC của 24 nhãn còn lại (`dataset/labels_v2_24.txt`). Thiết lập cây dữ liệu dẫn xuất `dataset/recordings_v2_4x24/` (576 clip độc nhất, 576 SHA-256 khác nhau) và recording plan `dataset/recording_plan_v2_4x24.json`. Khởi tạo landmark cache riêng `dataset/processed/landmark_cache_v2_4x24/`. Các script audit và finalize được tham số hóa linh hoạt theo `expected_clips=576, labels=24, folds=4`.
- **Trade-offs**: Tập từ vựng giảm từ 25 xuống 24 cử chỉ. Đổi lại, dữ liệu 4 người ký hoàn toàn sạch, triệt tiêu leakage, cho phép đo đạc LOSO 4-fold khách quan đạt độ chính xác 90.97% mà không bị ô nhiễm chéo. Model cũ trong `models/` được giữ nguyên cho đến khi Root duyệt.

## ADR-016: Công cụ đánh giá độc lập vslr-eval với cổng chống rò rỉ dữ liệu (Zero Leakage Gate)

- **Date**: 2026-09-07
- **Context**: Đánh giá trên người mới độc lập (ví dụ P05) hoặc các clip thu thập thực tế cần đảm bảo tính khách quan tuyệt đối: không được phép huấn luyện lại, không làm thay đổi trọng số checkpoint, và phải bảo đảm 100% không có clip kiểm thử nào trùng hash byte với bất kỳ clip huấn luyện nào.
- **Decision**: Xây dựng công cụ CLI `vslr-eval` (`prototype_3_gestures.evaluate`):
  1. Chỉ suy diễn (`torch.no_grad()`, `model.eval()`), kiểm tra SHA-256 của checkpoint trước và sau khi đánh giá để bảo đảm tính bất biến (invariant).
  2. Bổ sung cổng chống rò rỉ dữ liệu (Leakage Gate): đối chiếu mã băm SHA-256 của từng clip kiểm thử với file manifest huấn luyện (`--training-manifest`). Phát hiện trùng lặp là dừng ngay lập tức (fail-closed).
  3. Phân tách rõ ràng các chỉ số: Top-1 raw accuracy, accepted accuracy, coverage và rejection rate dựa trên ngưỡng `--confidence` (mặc định 0.50 chỉ dùng cho chẩn đoán/demo, không phải ngưỡng production).
  4. Hỗ trợ cả chế độ thư mục (`--data-dir`) và kiểm tra một clip nhanh (`--video "Nhãn=path"`).
  5. Toàn bộ kết quả xuất vào `--output-dir` gồm `predictions.csv`, `metrics.json`, `REPORT.md`, `confusion_matrix.png`; không ghi đè vào `models/`.
- **Trade-offs**: Clip kiểm thử phải qua trích xuất MediaPipe và hash SHA-256 từng file trước khi suy diễn. Đổi lại, kết quả đánh giá hoàn toàn minh bạch, khách quan, không thể bị nghi ngờ rò rỉ dữ liệu hay thay đổi trọng số mô hình.


