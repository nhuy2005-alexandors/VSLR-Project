# Gotchas — VSLR 3 cử chỉ

Bẫy đã gặp. Tích lũy, không xóa. Gặp bẫy mới → append.

## Train trên cả 7 clip rồi "test độc lập" trên clip 2 và 5 → leakage

- **Symptom**: holdout val accuracy 1.0, best_holdout_val_loss 0.0177 — đẹp hơn lần train sau, nhưng con số không đáng tin.
- **Cause**: `models/backups/train7_test2_agent_mistake_20260819/metrics.json` cho thấy lần train đó dùng `source_videos: 21` (clip 1→7 của cả ba nhãn), trong đó có chính clip 2 và 5 được dùng làm "test độc lập". Clip test nằm trong train set.
- **Avoidance/fix**: đã rollback về train chỉ trên clip 1, 4, 7 (`source_videos: 9`, `models/metrics.json` hiện tại), giữ clip còn lại ra ngoài hoàn toàn. Trước khi báo bất kỳ số accuracy nào: đối chiếu danh sách `extraction[].video` trong `metrics.json` với danh sách clip test — không được giao nhau.

## README claim số liệu không reproduce được từ dataset đang có

- **Symptom**: `README.md` viết "Kiểm tra độc lập trên clip 2, 5, 9: 8/9 (88,9%)" và "`Chuyện gì 5.mov` vẫn bị nhầm thành `Cảm ơn`".
- **Cause**: `dataset/raw/` hiện chỉ còn 9 file — clip 1, 4, 7 của mỗi nhãn. Không có clip 2, 5, hay 9 nào trên đĩa (backup 2026-08-19 cho thấy trước đây có clip 1→7, chưa từng thấy clip 9). `dataset/legacy/` là bộ VSLR khác hẳn (4362 file `.mp4`, đặt tên `D0001B.mp4`).
- **Avoidance/fix**: chưa fix — cần chủ dự án xác nhận clip test nằm ở đâu, hoặc sửa lại README cho đúng artifact. Nguyên tắc: số liệu trong docs phải chỉ được về file cụ thể còn tồn tại, kèm lệnh reproduce.

### Cập nhật 2026-08-21

Đã phục hồi đủ clip 1–9 của cả ba nhãn vào `dataset/raw/` và sửa README. Theo yêu cầu hiện tại, checkpoint cuối được refit trên **toàn bộ 27 clip**, vì vậy 27/27 trên sequence gốc chỉ là kiểm tra train artifact, không phải test accuracy. Muốn đánh giá độc lập phải quay dữ liệu mới chưa từng đưa vào train.

## Checkpoint được ship ra chỉ train 1 epoch, vì val loss bão hòa ngay epoch 1

- **Symptom**: `models/metrics.json` (bản train 27 clip) có `best_epoch: 1` và `final_fit_epochs: 1`. Nhìn qua thì `holdout_val_accuracy: 1.0` nên tưởng ổn.
- **Cause**: chuỗi nhân quả trong `prepare_train.py`:
  1. `holdout_val_accuracy` đạt 1.0 ngay từ epoch 1 (bài toán 3 lớp, 3267 mẫu augment, quá dễ tách) → khác biệt `val_loss` giữa các epoch chỉ còn là nhiễu.
  2. `best_epoch` chọn theo `val_loss` thấp nhất với margin `- 1e-4` (`prepare_train.py:221`). Epoch 1 tình cờ thấp nhất (0,01600) và 15 epoch sau không phá được → early stop, `best_epoch = 1`.
  3. `final_fit_epochs = max(1, best_epoch)` = 1 (`prepare_train.py:243`).
  4. Refit tạo model **mới từ random init** và train đúng 1 epoch (`prepare_train.py:244-260`).
  5. `save_checkpoint(..., final_model, ...)` (`prepare_train.py:269`) lưu chính model 1-epoch đó. Model `best_state` đã train tử tế bị load ở dòng 239 nhưng chỉ dùng để tính metric rồi **bỏ**.
- **Hệ quả phụ**: `holdout_val_accuracy` trong `metrics.json` đo trên `model` (best_state), không phải `final_model` được lưu. Số trong metadata mô tả một model khác với model thực sự ship ra.
- **Vì sao dễ bỏ sót**: thêm dữ liệu làm bug này XUẤT HIỆN chứ không làm nó mất. Lần train 9 clip có `best_epoch: 30` nên không ai thấy gì. Càng nhiều dữ liệu, val càng bão hòa sớm, `best_epoch` càng dễ bằng 1.
- **Kiểm tra "27/27 sequence gốc" không phủ được lỗi này**: đó là 27 mẫu nằm trong train set, và là mẫu dễ nhất (chưa augment).
- **Avoidance/fix**: đừng buộc số epoch refit vào `best_epoch` khi val đã bão hòa. Chọn `best_epoch` theo (val_accuracy, rồi val_loss), hoặc đặt sàn cho số epoch refit, hoặc bỏ hẳn refit và ship `best_state`. Luôn đọc `final_fit_epochs` trong `metrics.json` trước khi tin một checkpoint.

## Landmark của một clip phụ thuộc clip nào được extract trước nó

- **Symptom**: không có symptom. Train chạy, test pass, số trông hợp lý. Nhưng chạy lại cùng một dataset có thể ra bộ landmark khác.
- **Cause**: `HolisticExtractor` dùng **một** instance MediaPipe cho mọi clip, với `static_image_mode=False, smooth_landmarks=True`. MediaPipe mang state tracking + smoothing qua các lần `process_frame`, và `extract_video` cũ không tạo instance mới giữa hai video. Đo trực tiếp — cùng một clip, đổi thứ tự xử lý:
  ```
  Cảm ơn 1.mov      giong het: False | lech max 1.358964 | lech tb 0.054683
  Xin chào 1.mov    giong het: False | lech max 1.375215 | lech tb 0.030724
  clip di dau tien -> khop chinh xac ban extract rieng le; clip di sau -> lech
  6 clip, lech max tung clip: [0.0, 1.71, 0.75, 1.20, 1.99, 1.13]
  ```
  Lệch **1,36–1,99** trong đơn vị đã chuẩn hóa theo độ rộng vai, tức hơn một vai — không phải nhiễu làm tròn.
- **Hệ quả**: `cache_key` và `data_fingerprint` **đều không** mã hóa thứ tự đi cây, nên hai lần chạy có `data_fingerprint` giống hệt vẫn ăn hai bộ landmark khác nhau. Cache còn đóng băng vĩnh viễn thứ tự của lần chạy đầu → cache nguội và cache ấm cho ra dữ liệu train khác nhau.
- **Vì sao dễ bỏ sót**: dùng chung một instance là cách viết tự nhiên nhất, và với **một** clip thì nó đúng. Lỗi chỉ xuất hiện từ clip thứ hai, và không có gì báo.
- **Avoidance/fix**: `extract_video` mở instance MediaPipe **riêng cho mỗi video**; `process_frame` giữ instance dài hạn vì với camera thì mang state qua frame mới là đúng. Giá: 166 ms setup mỗi clip so với ~7 s inference → **+0%** wall clock (đo trên 6 clip: 41,76 s → 41,96 s). Bump `FEATURES_VERSION` lên 2 vì landmark đổi. Test `test_extract_video_is_not_contaminated_by_a_previous_video` dùng Holistic giả có state để khóa hành vi, và `test_live_camera_path_still_carries_state_between_frames` khóa chiều ngược lại.

## Bump `FEATURES_VERSION` tạo ra đúng cái mismatch âm thầm mà nó tồn tại để chặn

- **Symptom lịch sử**: sau khi bump `FEATURES_VERSION` 1 → 2, load `models/gesture_lstm.pt` chỉ ra **một** cảnh báo (về `pooling`), không cảnh báo nào về landmark — dù checkpoint đó được train trên landmark v1 và extractor lúc đó sinh v2. Hiện contract đã lên v3 với presence channels.
- **Cause**: `load_checkpoint` viết `if stored is not None and stored != FEATURES_VERSION`. Checkpoint cũ **không có** khóa `features_version` (nó ra đời trước khóa đó), nên `stored is None` và phép so sánh bị bỏ qua hoàn toàn. Tức đúng những artifact dễ cũ nhất là những artifact được miễn kiểm.
- **Avoidance/fix**: thiếu khóa nghĩa là "viết trước khi có khóa" = **version 1**, không phải "không cần kiểm". `int(config.get("features_version", 1))`. Lệch version/dimension giờ **raise luôn**, không có bypass vì checkpoint cũ không tương thích với feature v3. Cùng logic với `pooling` thiếu → `legacy_last_step`.
- **Bài học rộng hơn**: mỗi lần thêm một khóa metadata để phát hiện lệch, phải quyết **giá trị mặc định cho artifact chưa có khóa đó** ngay trong cùng lần sửa. Mặc định `None` + `is not None` là cách tự vô hiệu hóa cái guard vừa viết.

## Benchmark trên CPU chưa nguội cho số sai gấp 2–3 lần

- **Symptom**: cùng một hàm `augment_sequence`, ba lần đo ra 0,638 / 1,419 / 1,453 ms/mẫu. Chọn số nào cũng "có receipt" mà kết luận khác nhau hẳn.
- **Cause**: đo ngay sau `pytest` hoặc sau một lượt MediaPipe, CPU còn nóng/còn thread khác. `reviewer` cũng đạp đúng bẫy này và tự loại lần đo đầu (3,558 vs 1,75 ms cho cùng một hàm).
- **Avoidance/fix**: đo **xen kẽ** hai phương án trong cùng một vòng lặp (A,B,A,B…) rồi lấy `min` — drift CPU triệt tiêu, và tỉ số giữa hai bản là số đáng tin cả khi giá trị tuyệt đối trôi. Nhờ vậy mới ra `1,891 → 0,618 ms = 3,06x` cho việc gộp hai lần nội suy, trong khi đo rời rạc cho ra từ 8% đến 3x tùy lần. **Đừng bao giờ trích một con số tuyệt đối từ một lần đo duy nhất.**

## Tầng điều phối trong `main()` không có test, nên đổi tên khóa pass hết unit test rồi chết lúc chạy

- **Symptom**: đổi khóa history `train_loss` → `train_loss_smoothed` và `val_loss` → `val_loss_plain_ce`. **20/20 unit test pass.** Chạy `--loso` thật thì `KeyError: 'val_loss'` ở dòng tổng kết fold, sau khi đã extract xong 12 clip và train hết fold đầu.
- **Cause**: phần đọc `history[-1]["val_loss"]` nằm trong `main()`, chỉ chạy khi chạy CLI thật. Test unit gọi `train_model` (tạo ra history) và `signer_split` (chia fold) nhưng không có test nào gọi đoạn *tiêu thụ* history.
- **Avoidance/fix**: `TrainLoopTests` khóa đúng tập tên khóa của một history row (`test_history_keys_distinguish_smoothed_train_loss_from_plain_val_loss`), nên lần sau đổi tên là test đỏ chứ không phải run đỏ. Nguyên tắc rộng hơn: mỗi lần đổi hợp đồng dữ liệu giữa hai hàm, hỏi "test nào đang giữ hợp đồng này" — nếu không có thì viết trước khi đổi. Và **luôn chạy CLI thật một lượt sau refactor**, không dừng ở `pytest` xanh.

## `print` tiếng Việt ra console Windows bị UnicodeEncodeError

- **Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character 'ả'` khi in nhãn `Cảm ơn`.
- **Cause**: stdout Windows mặc định cp1252.
- **Avoidance/fix**: ba CLI gọi `configure_utf8_stdio()` nên chạy được cả khi người dùng quên biến môi trường; file JSON vẫn luôn mở bằng `encoding="utf-8"`.

## BiLSTM lấy `out[:, -1, :]` làm bỏ không một nửa model

- **Symptom**: không có symptom. Model train được, test pass, accuracy trông hợp lý. Không có gì báo lỗi.
- **Cause**: `GestureLSTM.forward` cũ trả `self.head(out[:, -1, :])`. Với `nn.LSTM(bidirectional=True)`, `out[:, t, :hidden]` là chiều thuận tại `t`, `out[:, t, hidden:]` là chiều **ngược** tại `t` — mà chiều ngược chạy từ frame cuối về đầu, nên tại `t = -1` nó mới nạp đúng **một** frame. Kết quả: 96 trong 192 chiều đưa vào `head` là hàm của một frame duy nhất. Đo được:
  ```
  doi 5/6 frame dau, giu frame cuoi:
    nua BACKWARD tai t=-1 doi khong? False   <- khong nhuc nhich
    nua FORWARD  tai t=-1 doi khong? True
  ```
  Kiểm chéo với `h_n`: `out[:, 0, hidden:] == h_n[1]` (chiều ngược đã xử lý hết chuỗi), còn `out[:, -1, hidden:] != h_n[1]`.
- **Vì sao dễ bỏ sót**: `out[:, -1, :]` là idiom đúng cho LSTM **một chiều** và copy-paste sang hai chiều thì shape vẫn khớp, không lỗi, không cảnh báo. Keras `Bidirectional(LSTM(...))` không có bẫy này vì nó tự ghép fwd-cuối với bwd-cuối, nên code TF tham chiếu không lộ vấn đề.
- **Avoidance/fix**: `pool_sequence` ghép `out[:, -1, :hidden]` với `out[:, 0, hidden:]`. Test `test_pool_takes_forward_last_and_backward_first` khóa hành vi. Checkpoint train bằng pooling cũ mang khóa `pooling` khác (hoặc thiếu khóa → mặc định `legacy_last_step`), nếu không nó sẽ dự đoán sai âm thầm.

## Early stopping trong leave-one-signer-out là tune tham số trên tập test

- **Symptom**: chưa xảy ra — bị bắt ở cổng spec trước khi code. Ghi lại vì rất dễ tự nhiên làm.
- **Cause**: bản nháp spec định lấy `--epochs` cho model ship ra từ "median epoch tốt nhất của các fold LOSO". Nhưng trong LOSO, người giữ ra **chính là tập test**. Dừng sớm theo accuracy của người đó, hoặc chọn epoch theo đó, là chọn siêu tham số bằng dữ liệu test → con số báo cáo bị lệch lạc quan và không còn là estimate của người mới.
- **Avoidance/fix**: không early stopping ở bất kỳ chế độ nào. `--epochs` là hằng chọn trước, ghi vào `loso_report.json` kèm câu nói rõ nó không được tune trên fold. Curve val theo epoch vẫn lưu để xem, không dùng để chọn. Muốn tune tử tế thì cần inner split lồng bên trong từng fold — chưa làm.

## `np.savez` tự thêm `.npz` vào tên file, phá pattern write-then-rename

- **Symptom**: bắt được khi đọc lại code trước lúc chạy, nên chưa bao giờ nổ thật. Nếu để nguyên thì cache landmark ghi ra `<key>.npz.tmp.npz` trong khi `staging.replace(cached)` trỏ tới `<key>.npz.tmp` → `FileNotFoundError` và không entry cache nào được tạo.
- **Cause**: `np.savez(path, ...)` append `.npz` nếu path chưa kết thúc bằng `.npz`. File staging cố ý kết thúc bằng `.tmp`. Đo trực tiếp:
  ```
  savez(path ending .tmp) -> ['a.npz.tmp.npz']
  savez(open handle)      -> ['b.npz.tmp']
  ```
- **Avoidance/fix**: truyền **file handle** đã mở, không truyền path: `with open(staging, "wb") as fh: np.savez(fh, ...)`. Giữ pattern ghi-rồi-rename để một run bị ngắt không để lại cache entry hỏng.

## Một frame thấy tay vẫn qua `--min-seconds` vì duration ăn cả word gap

- **Symptom**: `word_gap=0,45`, `min_seconds=0,35`; đúng một frame thấy tay ở `t=0`, hai frame không tay ở `t=0,10/0,46` tạo segment `duration=0,46` và bị đem đi classify.
- **Cause**: `Segment.end_time` là lúc gap đủ dài để đóng biên, không phải frame cuối còn thấy tay. Lấy `end_time - start_time` cộng toàn bộ idle tail vào độ dài cử chỉ, nên khi `word_gap > min_seconds` bộ lọc tối thiểu vô hiệu theo cấu trúc.
- **Avoidance/fix**: `Segment` giữ riêng `active_end_time=last_hand_time`; `duration` chỉ là active span, còn `end_time` vẫn giữ thời điểm đóng biên. Test một-frame khóa `duration == 0`.

## MediaPipe thấy bàn tay không có nghĩa người đang múa

- **Symptom**: người đứng hạ tay nhưng webcam vẫn ở `GESTURE`, chạm `--max-seconds (5.0)` và phân
  loại một đoạn chứa nhiều giây tư thế nghỉ.
- **Cause**: state machine cũ dùng `left_hand_present or right_hand_present` làm activity. Với khung
  toàn thân, MediaPipe vẫn detect được bàn tay đang nghỉ cạnh hông.
- **Avoidance/fix**: activity gate so cổ tay với đường hông trong tọa độ chuẩn hóa; presence chỉ mô tả
  observation. Giữ transition context hữu hạn 1,0 giây trước và 0,5 giây sau để không cắt mất động
  tác đưa/hạ tay mà model đã học. Fallback presence nếu pose/hông không đủ.

## Nhãn vắng ở tất cả người ký là vô hình nếu chỉ nhìn cây

- **Symptom**: hai người cùng có `Cảm ơn` nhưng cùng quên `Xin chào`; `validate_clips(..., loso=True)` vẫn pass vì bài toán một lớp tự nhất quán.
- **Cause**: tập "expected" cũ được suy từ chính clip đã discover. Không có thư mục thì không có dữ kiện để biết nhãn từng được mong đợi.
- **Avoidance/fix**: `vslr-train --data-dir` bắt buộc đối chiếu `dataset/labels.txt` trước extract; thiếu toàn cục hoặc thừa slug đều exit 1 và chưa tạo `model-dir`.

## Cache `.npz` đọc được chưa có nghĩa là cache hợp lệ

- **Symptom**: entry có đủ key nhưng sequence shape `(1, 201)` hoặc chứa NaN vẫn được trả `from_cache=True`; run chỉ chết muộn trong LayerNorm/DataLoader.
- **Cause**: đường cache-hit chỉ `.astype(np.float32)` và đọc scalar, không kiểm contract `(SEQUENCE_LENGTH, FEATURE_DIM)` hay finite/range.
- **Avoidance/fix**: `_read_cache_entry` kiểm shape `(60, 201)`, finite values, frame counts và `hand_frame_ratio`; sai thì xóa, cảnh báo và re-extract như cache ZIP hỏng.

## Cùng data fingerprint chưa đủ để gọi LOSO là số của checkpoint

- **Symptom**: LOSO chạy `--augment 0 --learning-rate 1e-4`, ship chạy mặc định 120 / `1e-3`; fingerprint/epochs/seed/features vẫn khớp nhưng hai quy trình khác hẳn.
- **Cause**: hướng dẫn ghép artifact cũ chỉ liệt kê bốn khóa, dù metadata đã có batch size, learning rate và augmentation.
- **Avoidance/fix**: chỉ ghép khi `training_signature` khớp trong report, metrics và config bên trong checkpoint; `checkpoint_sha256` khóa metrics vào đúng bytes của weights. Hash bao phủ data/nhãn, hyperparameter, recipe/model/augment, device và runtime.

## Cảnh báo thiếu manifest vẫn có thể kết thúc bằng exit 0

- **Symptom**: `vslr-check --labels-file typo.txt` cảnh báo không có manifest rồi vẫn in “Tất cả clip đạt và đủ độ phủ”.
- **Cause**: `expected_labels=None` làm bỏ hẳn `coverage_gaps`; `incomplete = bool(gaps and ...)` thành false.
- **Avoidance/fix**: manifest là input bắt buộc và được đọc trước MediaPipe. Mọi `OSError`/manifest rỗng-trùng nhãn exit 1; `_print_report` luôn nhận `list[str]`, không có nhánh “không biết expected nhưng vẫn thành công”.

## `except Exception` biến lỗi hệ thống thành clip quay hỏng

- **Symptom**: `PermissionError` ở 1/20 clip bị in `SKIP`; đúng 5% nên gate `> 5%` vẫn train 19 clip còn lại.
- **Cause**: quota chất lượng clip bắt mọi exception, không phân biệt lỗi dữ liệu và hạ tầng.
- **Avoidance/fix**: extractor raise `ClipExtractionError` riêng cho không mở được/quá ngắn/thấy tay <10%. `extract_all` chỉ bắt loại này và `FileNotFoundError`; permission, MediaPipe runtime và lỗi lập trình nổi lên làm abort.

## Chuẩn hóa Unicode không được phép sửa luôn cấu trúc thư mục

- **Symptom**: directory `' Cảm ơn'` từng qua manifest do `.strip()`; hai directory NFC/NFD của cùng signer từng gộp thành một class và làm phồng số clip.
- **Cause**: cùng một helper vừa NFC vừa strip, và discovery mất raw directory name sau chuẩn hóa.
- **Avoidance/fix**: `normalise_label` chỉ NFC. Parser manifest tự strip từng dòng như formatting; tên directory giữ whitespace để exact gate bắt. Discovery nhớ raw name theo signer và từ chối collision sau NFC.

## Path + mtime không định danh được nội dung video

- **Symptom**: ghi bytes mới vào cùng path rồi restore mtime làm `cache_key_same=True`, `data_fingerprint_same=True`; train có thể lấy landmark cũ và ghép LOSO/ship như chưa đổi data.
- **Cause**: timestamp là metadata có thể được bảo tồn có chủ ý; ngay cả thêm size vẫn bỏ sót thay nội dung cùng kích thước.
- **Avoidance/fix**: băm SHA-256 toàn bộ file cho cache key và data fingerprint, kèm size/mtime_ns để audit. Cache cũ miss một lần có chủ ý.

## Ghi version vào checkpoint mà loader không kiểm chỉ tạo cảm giác an toàn

- **Symptom**: checkpoint khai `model_architecture_version=999` nhưng vẫn load vì tensor shape khớp.
- **Cause**: `load_checkpoint` chỉ kiểm feature version; metadata architecture được ghi nhưng không có consumer.
- **Avoidance/fix**: thiếu architecture key được hiểu là topology gốc v1; version khác `MODEL_ARCHITECTURE_VERSION` luôn raise. Không có override vì code hiện tại không thể biết semantics của topology tương lai dù shape trùng.
