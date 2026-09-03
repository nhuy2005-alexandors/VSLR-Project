# VSLR - 30 cử chỉ sang câu nói

Prototype nhận 30 cử chỉ tiếng Việt bằng MediaPipe + BiLSTM. Camera nhận từng cử chỉ, lưu các từ vào bộ đệm và đọc nguyên câu sau khi người dùng dừng.

## Cấu trúc

```text
src/prototype_3_gestures/  Source train và realtime
tests/                     Unit tests
dataset/recordings_v1/     Cây quay P01–P04, 30 nhãn × 6 clip/người
dataset/recording_plan.json Hợp đồng người và quota; không lặp danh sách nhãn
dataset/processed/         Cache landmark per-clip, tự sinh lại được
dataset/legacy/            Bộ từ điển VSL crawl từ bên thứ ba (4362 video / 3315 nhãn)
models/                    Checkpoint và metadata
```

## Cài đặt

```powershell
python -m pip install -e ".[tts]"
```

## Camera test

Checkpoint đang track được train bằng landmark cũ, còn extractor hiện tại là feature v3 (landmark + presence L/R). Lệnh mặc định
**từ chối** chạy artifact không tương thích cho tới khi train lại:

```powershell
vslr-camera --no-tts
```

Không có cờ bypass cho checkpoint lệch dimension/feature contract. Sau khi train checkpoint v3
và có reject policy calibration, chạy bình thường. Trước khi có negative/idle/OOV calibration,
realtime fail-closed và không phát âm segment bị reject; chỉ dùng `--allow-uncalibrated` cho demo
tạm thời.

Mỗi lần múa xong một cử chỉ, hạ tay khoảng 0,45 giây. Sau cử chỉ cuối, giữ nghỉ khoảng 2,2 giây để hệ thống đọc cả câu. Phím `SPACE` chốt segment, `S` đọc ngay, `C` xóa câu, `Q` thoát.

## Kết quả hiện tại

**Chưa có con số accuracy nào báo cáo được.**

- 27 clip nguồn, 9 clip mỗi cử chỉ. Tên file **không** mã hóa người ký, và không ai xác nhận được clip nào của người nào — nên không chạy được leave-one-signer-out trên bộ này.
- Checkpoint trong `models/` được train bằng **pipeline cũ**, đúng 1 epoch, pooling cũ và feature v1. Loader hiện chặn nó vì extractor sinh feature v3. **Không** dùng artifact này để báo bất kỳ số nào.
- Muốn có con số: quay dataset có ID người theo `docs/specs/dataset-recording.md`, rồi chạy `--loso`.

## Train

Hai chế độ, một lệnh. Chi tiết trong `docs/technical_specs/pipeline-restructure.md`.

```powershell
# Kiểm cấu trúc, độ phủ và chất lượng clip trước; không train
vslr-check --data-dir dataset/recordings_v1

# ĐO: leave-one-signer-out, ghi models/loso_report.json, KHÔNG ghi checkpoint
vslr-train --data-dir dataset/recordings_v1 --loso --num-workers 4

# SẢN XUẤT: train mọi clip, ship weight cuối, KHÔNG kèm accuracy
vslr-train --data-dir dataset/recordings_v1 --num-workers 4
```

Các CLI tự cấu hình stdout/stderr UTF-8 trên Windows.

## Tạo cây quay và tự gán nhãn

`dataset/labels.txt` là manifest V1 gồm 30 nhãn đã chốt. Tool đọc file này nên không viết cứng
20, 25 hay 30 lớp. Xem trước kế hoạch rồi tạo cây cho 4 người, 6 clip/người/nhãn:

```powershell
vslr-init-dataset --data-dir dataset/recordings_v1 --people-count 4 --clips-per-label 6 --dry-run
vslr-init-dataset --data-dir dataset/recordings_v1 --people-count 4 --clips-per-label 6
```

Kết quả là 120 thư mục nhãn và kế hoạch 720 video. Tool không tạo video rỗng, không di chuyển
hay ghi đè clip. Lưu từng lần quay thành `001.mov` đến `006.mov` (hoặc `.mp4`) trong thư mục
đúng nhãn. Pipeline tự lấy `person=P01` và `label=Cảm ơn` từ đường dẫn, nên không cần gán nhãn
từng file bằng tay. `recording_plan.json` là nguồn quota/người chung của init, check và train;
file `dataset/labels.txt` vẫn là nguồn duy nhất của class order.

## Nạp video mới

`--data-dir` cần cây `DIR/<người>/<cử chỉ>/*.mov`. Mỗi tên thư mục cử chỉ phải khớp
**chính xác sau chuẩn hóa Unicode NFC** với một dòng trong `dataset/labels.txt`; cây directory mode
phải đủ đúng P01–P04 và đúng 6 video cho mọi cặp người/nhãn. Thiếu, thừa, người ngoài plan hoặc
hai video trùng bytes đều bị chặn trước khi chạy MediaPipe:

```text
dataset/recordings_v1/P01/Cảm ơn/001.mov
dataset/recordings_v1/P01/Cảm ơn/002.mov
dataset/recordings_v1/P02/Cảm ơn/001.mov
```

Khoảng trắng đầu/cuối trong tên thư mục là lỗi, không được tự bỏ. Nếu cùng một người có đồng thời
hai thư mục NFC/NFD quy về cùng một nhãn, pipeline cũng dừng để tránh gộp clip và làm phồng độ phủ.
`vslr-check` bắt buộc đọc được `--labels-file`; sai đường dẫn manifest là exit 1 trước MediaPipe,
không thể báo “đủ độ phủ”.

Không đổi tên hoặc tráo thứ tự dòng trong `dataset/labels.txt` sau khi đã train V1. Nếu tạo bộ 20
nhãn riêng, dùng một manifest mới và truyền nó nhất quán cho cả ba lệnh init/check/train.

`dataset/raw/` hiện **phẳng** (`dataset/raw/<cử chỉ>/*.mov`, không có cấp người) nên `--data-dir dataset/raw` báo lỗi ở cả hai chế độ — đúng thiết kế, nó không đoán ID người:

```
error: No .mov/.mp4 clips under dataset\raw. Expected layout DIR/<person>/<gesture>/*.mov
```

Train bộ 27 clip hiện tại phải dùng chế độ một-người:

```powershell
$trainArgs = @()
$sets = @(
  @("Cảm ơn", "dataset/raw/cam_on"),
  @("Chuyện gì", "dataset/raw/chuyen_gi"),
  @("Xin chào", "dataset/raw/xin_chao")
)
foreach ($set in $sets) {
  foreach ($i in 1..9) {
    $trainArgs += "--video"
    $trainArgs += "$($set[0])=$(Join-Path $set[1] "$($set[0]) $i.mov")"
  }
}
vslr-train @trainArgs
```

Chế độ đó gán mọi clip là cùng một người, không dùng `dataset/labels.txt` và từ chối `--loso`, nên
chỉ phù hợp để tạo artifact thử nghiệm từ bộ legacy.

## Đo ghép câu offline

`vslr-sentence` dùng chính `SegmentTracker`, presence-aware preprocessing và reject policy của
webcam. Ground truth batch nằm trong CSV `person,clip,words` (các nhãn trong `words` ngăn bằng
`|`; để trống là negative/idle/OOV). Không có calibration âm thì lệnh ghi rõ `uncalibrated` và
reject mặc định; dùng `--allow-uncalibrated` chỉ khi chấp nhận kết quả tạm thời.

```powershell
vslr-sentence --dir dataset/raw_sentences --manifest dataset/raw_sentences/sentences.csv --model models/gesture_lstm.pt
vslr-sentence --clip dataset/raw_sentences/P01/cau_01.mov --expect "Xin chào|Cảm ơn" --model models/gesture_lstm.pt
```

Báo cáo batch có exact sentence accuracy, WER/edit distance, count mismatch và false accept
trên các dòng negative. Chưa có raw negative/idle/OOV và checkpoint feature v3 để chạy metric
thực tế trong workspace hiện tại.
