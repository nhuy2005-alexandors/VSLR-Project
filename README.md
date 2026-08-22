# VSLR - 3 cử chỉ sang câu nói

Prototype nhận ba cử chỉ tiếng Việt `Cảm ơn`, `Chuyện gì`, `Xin chào` bằng MediaPipe + BiLSTM. Camera nhận từng cử chỉ, lưu các từ vào bộ đệm và đọc nguyên câu sau khi người dùng dừng.

## Cấu trúc

```text
src/prototype_3_gestures/  Source train và realtime
tests/                     Unit tests
dataset/raw/               27 clip train (1–9 mỗi nhãn), tên file không mã hóa người ký
dataset/processed/         Cache landmark per-clip, tự sinh lại được
dataset/legacy/            Bộ từ điển VSL crawl từ bên thứ ba (4362 video / 3315 nhãn)
models/                    Checkpoint và metadata
```

## Cài đặt

```powershell
python -m pip install -e ".[tts]"
```

## Camera test

Test nhận dạng, chưa phát giọng:

```powershell
vslr-camera --no-tts
```

Bật TTS:

```powershell
vslr-camera
```

Mỗi lần múa xong một cử chỉ, hạ tay khoảng 0,45 giây. Sau cử chỉ cuối, giữ nghỉ khoảng 2,2 giây để hệ thống đọc cả câu. Phím `SPACE` chốt segment, `S` đọc ngay, `C` xóa câu, `Q` thoát.

## Kết quả hiện tại

**Chưa có con số accuracy nào báo cáo được.**

- 27 clip nguồn, 9 clip mỗi cử chỉ. Tên file **không** mã hóa người ký, và không ai xác nhận được clip nào của người nào — nên không chạy được leave-one-signer-out trên bộ này.
- Checkpoint trong `models/` được train bằng **pipeline cũ** (chọn epoch theo source-holdout rồi refit; cả hai đã bị xóa). Nó train đúng 1 epoch và dùng pooling BiLSTM cũ. Chạy được cho demo camera, **không** dùng để báo bất kỳ số nào. Chi tiết: `docs/GOTCHAS.md`.
- Muốn có con số: quay dataset có ID người theo `docs/specs/dataset-recording.md`, rồi chạy `--loso`.

## Train

Hai chế độ, một lệnh. Chi tiết trong `docs/technical_specs/pipeline-restructure.md`.

```powershell
$env:PYTHONIOENCODING = "utf-8"

# ĐO: leave-one-signer-out, ghi models/loso_report.json, KHÔNG ghi checkpoint
vslr-train --data-dir dataset/raw --loso --num-workers 4

# SẢN XUẤT: train mọi clip, ship weight cuối, KHÔNG kèm accuracy
vslr-train --data-dir dataset/raw --num-workers 4
```

`--data-dir` cần cây `DIR/<người>/<cử chỉ>/*.mov`, tên thư mục cử chỉ là tiếng Việt có dấu vì nó đi thẳng vào `labels.json` và vào câu TTS đọc ra:

```text
dataset/raw/P1/Cảm ơn/Cảm ơn 01.mov
```

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

Chế độ đó gán mọi clip là cùng một người và từ chối `--loso`, nên nó chỉ ra được artifact để demo.
