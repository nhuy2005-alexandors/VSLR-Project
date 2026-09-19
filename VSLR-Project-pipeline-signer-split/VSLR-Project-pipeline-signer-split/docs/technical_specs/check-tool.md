# As-built — `vslr-check`, nghiệm thu clip tại chỗ quay

_Xong: 2026-08-23. Spec: `docs/specs/record-to-test.md` lỗ hổng 1. Verify hiện tại: **78 passed, 31 subtests passed**._

## Vì sao cần

`docs/specs/dataset-recording.md` bắt kiểm 180 clip vừa quay, xem `hand_frame_ratio`, clip nào < 0,5 quay lại ngay. Trước khi có lệnh này, muốn xem tỷ lệ đó phải chạy cả lượt train; người quay về rồi mới biết clip tệ là mất một buổi.

## Hai quyết định đến từ cổng spec-critic

**Không gọi `validate_clips`.** Buổi quay đi theo vòng: sau vòng 1 mọi nhãn mới có đúng một clip, mà `validate_clips` đòi ≥ 2 clip mỗi nhãn (`prepare_train.py:99-101`) và ở chế độ LOSO đòi ≥ 2 người. Gọi nó là abort ngay lần chạy đầu tiên ở chỗ quay. `vslr-check` là **công cụ báo cáo, không phải cổng** — nó không từ chối gì, chỉ đếm và `exit 1`.

**Tập nhãn mong đợi đến từ `dataset/labels.txt`, không suy từ cây.** Nhãn chưa ai quay thì không thể xuất hiện trong cây thư mục, nên cây không bao giờ tự nói được là còn thiếu gì — đúng cùng lý do `missing_labels_after_drop` tồn tại trong `prepare_train.py`. Manifest là bắt buộc: thiếu/sai đường dẫn thì exit 1 **trước MediaPipe**. Cảnh báo rồi tiếp tục từng tạo ra kết quả “Tất cả clip đạt và đủ độ phủ” giả khi `expected_labels=None`.

## Dùng thế nào

```powershell
# cay nhieu nguoi: DIR/<nguoi>/<nhan>/*.mov
vslr-check --data-dir dataset/recordings_v1

# o cho quay chi co P1, cay la DIR/<nhan>/*.mov -> co --person TUONG MINH
vslr-check --data-dir dataset/raw/P1 --person P1

vslr-check --data-dir dataset/recordings_v1 --clips-per-label 6 --min-hand-ratio 0.5
```

`--person` là cờ tường minh, không đoán theo cấu trúc: đoán sai là biến tên nhãn thành tên người.

## Ba trạng thái lỗi tách riêng, vì ba hành động khác nhau

| Trạng thái | Nghĩa | Làm gì |
|---|---|---|
| `KHONG MO DUOC` | `cv2` không mở được file | Thường là lỗi truyền file, **không** phải lỗi quay |
| `QUA NGAN` | dưới 8 frame đọc được | Quay lại, dài hơn |
| `KHONG THAY TAY` | MediaPipe thấy tay dưới 10% frame | Quay lại, tay rõ trong khung |
| `QUAY LAI` | `hand_frame_ratio < --min-hand-ratio` | Quay lại |
| `LOI HE THONG` | quyền file/cache/lỗi phần mềm không thuộc ba lỗi quay ở trên | **Không quay lại**; sửa lỗi được in kèm |
| `ok` | đạt | — |

Bản spec gốc gộp ba cái đầu thành một `FAIL`; `spec-critic` chỉ ra chúng cần ba hành động khác nhau.

## Bảng in ra

Lỗi trước, rồi `ok` tăng dần theo `hand_frame_ratio` và chỉ in 5 dòng đầu — dòng đầu tiên luôn là clip đáng lo nhất. Cuối bảng là độ phủ: cặp `(người, nhãn)` chưa có clip nào, cặp còn thiếu clip, và nhãn có trong cây mà **không** có trong `labels.txt`.

Cảnh báo riêng cho cây trộn: thư mục cấp 1 nào chứa video trực tiếp (không có thư mục nhãn con) thì `discover_clips` bỏ qua **không một lời** — `find_skipped_dirs` phát hiện và in ra. Đây là tình huống sẽ xảy ra thật khi quay mới vào `dataset/raw/P1/` bên cạnh `dataset/raw/cam_on/` cũ.

## Dùng chung cache với `vslr-train`

Mặc định `--cache-dir dataset/processed/landmark_cache`, giống `vslr-train`. Clip đã check không phải extract lại lúc train. Đo thật: lần đầu 27 clip mất ~3 phút, chạy lại tức thì.

Cache key hiện chứa SHA-256 bytes nguồn. Cache cũ trước 2026-08-23 miss một lần; sau đó check/train dùng lại bình thường. Copy đè video rồi giữ nguyên timestamp vẫn buộc extract lại.

## Verify

Chạy thật trên 27 clip hiện có:

```
27 clip ok, 0 clip cần xử lý.
Độ phủ: 1 người × 27 nhãn × 9 clip = 243 clip mong đợi
  Nhãn KHÔNG có trong dataset/labels.txt: ['cam_on', 'chuyen_gi', 'xin_chao']
  Chưa có clip nào (27 cặp): [('P0', 'Bao nhiêu'), ('P0', 'Chuyện gì'), ...]
```

Nó bắt đúng một chuyện thật ngay lần đầu: 27 clip cũ nằm trong thư mục **slug** (`cam_on`), không khớp nhãn tiếng Việt trong `labels.txt`. Đó chính là loại lệch mà nếu không phát hiện thì `labels.json` và câu TTS sẽ ra `cam_on`.

`hand_frame_ratio` thấp nhất 59,7% — khớp khoảng 0,59–0,91 ghi trong `models/metrics.json`, tức đường extract cho ra cùng số như pipeline train.

Hai đường exit đều kiểm:

- đủ độ phủ (`--clips-per-label 9`, labels khớp) → **exit 0**, in "Tất cả clip đạt và đủ độ phủ."
- thiếu (`--clips-per-label 12`) → **exit 1**, liệt kê `[('P0', 'cam_on', 9), ...]`
- thiếu/sai `--labels-file` → **exit 1 trước MediaPipe**, không in câu xác nhận đủ độ phủ

**Chưa verify**: chưa chạy trên cây nhiều người thật (chưa có dữ liệu), và chưa chạy trên clip lỗi thật (mọi trạng thái lỗi được kiểm bằng unit test `clip_status`, không bằng file hỏng thật).

## Watch-outs

- Nhãn được so bằng NFC. Một raw directory NFD được chấp nhận và quy về manifest NFC; nhưng hai directory NFC+NFD cùng tồn tại cho một người bị từ chối để không gộp/inflate clip. Khoảng trắng trong tên directory không được tự strip.
- `--clips-per-label` mặc định 6 theo kế hoạch V1. Kiểm giữa buổi thì đặt bằng số vòng đã quay, không thì mọi cặp đều báo thiếu.
- Lệnh này **không** ghi vào `models/`. Nó chỉ đọc, và ghi cache landmark.
- CLI tự cấu hình UTF-8, nên help và nhãn tiếng Việt in được trên console Windows không cần biến môi trường.
