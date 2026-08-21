# VSLR - 3 cử chỉ sang câu nói

Prototype nhận ba cử chỉ tiếng Việt `Cảm ơn`, `Chuyện gì`, `Xin chào` bằng MediaPipe + BiLSTM. Camera nhận từng cử chỉ, lưu các từ vào bộ đệm và đọc nguyên câu sau khi người dùng dừng.

## Cấu trúc

```text
src/prototype_3_gestures/  Source train và realtime
tests/                     Unit tests
dataset/raw/               9 clip train đã chọn (1, 4, 7 mỗi nhãn)
dataset/processed/         Landmark dataset đã augmentation
dataset/legacy/            Bộ video VSLR lớn được giữ để dùng sau
models/                    Checkpoint và metadata vừa train
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

- Train bằng 3 clip/cử chỉ: clip 1, 4, 7.
- Source-holdout trong quá trình chọn epoch: 100%.
- Kiểm tra độc lập trên clip 2, 5, 9: 8/9 (88,9%).
- `Chuyện gì 5.mov` vẫn bị nhầm thành `Cảm ơn`; đây là giới hạn đã biết của dataset nhỏ.

## Train lại

Lệnh train nhận đúng ba nhãn, mỗi nhãn ít nhất hai video. Ví dụ:

```powershell
vslr-train `
  --video "Cảm ơn=dataset/raw/cam_on/Cảm ơn 1.mov" `
  --video "Cảm ơn=dataset/raw/cam_on/Cảm ơn 4.mov" `
  --video "Cảm ơn=dataset/raw/cam_on/Cảm ơn 7.mov" `
  --video "Chuyện gì=dataset/raw/chuyen_gi/Chuyện gì 1.mov" `
  --video "Chuyện gì=dataset/raw/chuyen_gi/Chuyện gì 4.mov" `
  --video "Chuyện gì=dataset/raw/chuyen_gi/Chuyện gì 7.mov" `
  --video "Xin chào=dataset/raw/xin_chao/Xin chào 1.mov" `
  --video "Xin chào=dataset/raw/xin_chao/Xin chào 4.mov" `
  --video "Xin chào=dataset/raw/xin_chao/Xin chào 7.mov"
```
