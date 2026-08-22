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

## Tầng điều phối trong `main()` không có test, nên đổi tên khóa pass hết unit test rồi chết lúc chạy

- **Symptom**: đổi khóa history `train_loss` → `train_loss_smoothed` và `val_loss` → `val_loss_plain_ce`. **20/20 unit test pass.** Chạy `--loso` thật thì `KeyError: 'val_loss'` ở dòng tổng kết fold, sau khi đã extract xong 12 clip và train hết fold đầu.
- **Cause**: phần đọc `history[-1]["val_loss"]` nằm trong `main()`, chỉ chạy khi chạy CLI thật. Test unit gọi `train_model` (tạo ra history) và `signer_split` (chia fold) nhưng không có test nào gọi đoạn *tiêu thụ* history.
- **Avoidance/fix**: `TrainLoopTests` khóa đúng tập tên khóa của một history row (`test_history_keys_distinguish_smoothed_train_loss_from_plain_val_loss`), nên lần sau đổi tên là test đỏ chứ không phải run đỏ. Nguyên tắc rộng hơn: mỗi lần đổi hợp đồng dữ liệu giữa hai hàm, hỏi "test nào đang giữ hợp đồng này" — nếu không có thì viết trước khi đổi. Và **luôn chạy CLI thật một lượt sau refactor**, không dừng ở `pytest` xanh.

## `print` tiếng Việt ra console Windows bị UnicodeEncodeError

- **Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode character 'ả'` khi in nhãn `Cảm ơn`.
- **Cause**: stdout Windows mặc định cp1252.
- **Avoidance/fix**: đặt `PYTHONIOENCODING=utf-8` trước khi chạy script có in nhãn tiếng Việt; luôn mở file JSON bằng `encoding="utf-8"`.

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
