# As-built — `vslr-check`, nghiệm thu clip tại chỗ quay

_Xong: 2026-08-22. Spec: `docs/specs/record-to-test.md` lỗ hổng 1 (spec đó rev 1 BLOCKED; hai blocker liên quan tới `vslr-check` đã trả lời và bản này làm theo câu trả lời). Test: **43 passed**._

## Vì sao cần

`docs/specs/dataset-recording.md` bắt "extract 108 clip vừa quay, xem `hand_frame_ratio`, clip nào < 0,5 quay lại ngay". Không có lệnh nào làm việc đó — muốn xem `hand_frame_ratio` phải chạy cả lượt train. Người quay về rồi mới biết clip tệ là mất một buổi.

## Hai quyết định đến từ cổng spec-critic

**Không gọi `validate_clips`.** Buổi quay đi theo vòng: sau vòng 1 mọi nhãn mới có đúng một clip, mà `validate_clips` đòi ≥ 2 clip mỗi nhãn (`prepare_train.py:99-101`) và ở chế độ LOSO đòi ≥ 2 người. Gọi nó là abort ngay lần chạy đầu tiên ở chỗ quay. `vslr-check` là **công cụ báo cáo, không phải cổng** — nó không từ chối gì, chỉ đếm và `exit 1`.

**Tập nhãn mong đợi đến từ `dataset/labels.txt`, không suy từ cây.** Nhãn chưa ai quay thì không thể xuất hiện trong cây thư mục, nên cây không bao giờ tự nói được là còn thiếu gì — đúng cùng lý do `missing_labels_after_drop` tồn tại trong `prepare_train.py`. Thiếu file đó thì lệnh vẫn chạy nhưng in cảnh báo rằng nó **không** phát hiện được nhãn chưa quay.

## Dùng thế nào

```powershell
# cay nhieu nguoi: DIR/<nguoi>/<nhan>/*.mov
vslr-check --data-dir dataset/raw

# o cho quay chi co P1, cay la DIR/<nhan>/*.mov -> co --person TUONG MINH
vslr-check --data-dir dataset/raw/P1 --person P1

vslr-check --data-dir dataset/raw --clips-per-label 4 --min-hand-ratio 0.5
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

**Chưa verify**: chưa chạy trên cây nhiều người thật (chưa có dữ liệu), và chưa chạy trên clip lỗi thật (mọi trạng thái lỗi được kiểm bằng unit test `clip_status`, không bằng file hỏng thật).

## Watch-outs

- Nhãn được so bằng NFC. Tên NFD (kiểu macOS/iOS) in ra **giống hệt** NFC nhưng là chuỗi khác; `normalise_label` chuẩn hóa cả hai phía nên không báo "thiếu" một nhãn trông y như nhãn đang có. Có test.
- `--clips-per-label` mặc định 4 theo `dataset-recording.md`. Kiểm giữa buổi thì đặt bằng số vòng đã quay, không thì mọi cặp đều báo thiếu.
- Lệnh này **không** ghi vào `models/`. Nó chỉ đọc, và ghi cache landmark.
- CLI tự cấu hình UTF-8, nên help và nhãn tiếng Việt in được trên console Windows không cần biến môi trường.
