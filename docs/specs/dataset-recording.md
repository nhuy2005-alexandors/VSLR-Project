# Spec — Quay dataset 27 cử chỉ, 5 người

_Viết: 2026-08-22. Rev 2 — thay thế bản 3 cử chỉ / 3 người / 6 clip. Trạng thái: chưa quay._

## Goal

540 clip train (5 người × 27 nhãn × 4 clip) + 20 clip câu (5 người × 4 clip), đủ để chạy leave-one-signer-out 5 fold và chỉnh tham số ghép câu bằng dữ liệu thật.

Danh sách 27 nhãn **chưa chốt** (chủ dự án lên sau). Giao thức trong file này không phụ thuộc vào chọn từ nào; đổi số nhãn chỉ đổi số clip mỗi fold.

## Kịch bản 4 người hay 5 người

Không chọn trước. Quay **xong hẳn từng người** theo thứ tự P1 → P5. Mỗi người hoàn thành đủ 108 clip mới sang người sau.

Hệ quả: dừng ở bất kỳ người nào cũng có một dataset **hoàn chỉnh** ở quy mô đó, không phải một dataset 5 người bị dở. P5 hủy hẹn → còn đúng bộ 4 người dùng được ngay, không phải quay lại gì.

Nếu quay đủ 5 thì tốt hơn thật, và lý do không phải số fold:

| | 4 người | 5 người |
|---|---|---|
| Số fold LOSO | 4 | 5 |
| **Người model được học** mỗi fold | **3** | **4** |
| Clip train mỗi fold | 324 | 432 |
| Clip test mỗi fold | 108 | 108 |
| Model ship ra học từ | 4 người | 5 người |
| Con số LOSO mô tả model học từ | 75% dữ liệu | 80% dữ liệu |
| Tổng clip phải quay | 432 | 540 |

Thứ quyết định model có chạy với người mới không phải là số fold mà là **số người model được học**. 3 → 4 người là tăng 33% đa dạng, ở quy mô nhỏ như này thì đó là biến quan trọng nhất.

Và `docs/specs/dataset-recording.md` rev 1 (dòng 91) đã tự chốt ngưỡng này rồi: với 3 người thì **không được** viết "hoạt động với người mới", muốn claim đó cần tối thiểu 5–6 người. 4 người vẫn dưới ngưỡng đó; 5 người là cạnh dưới của nó.

Giá phải trả: 108 clip = một buổi 60–90 phút của một người.

Người thứ 6 nếu mời được thì lấy — lợi ích còn tăng đến khoảng 8 người rồi mới phẳng. Nhưng đừng đợi người thứ 6 mà hoãn quay P1.

## Quay P1 trước, kiểm, rồi mới quay tiếp

Bắt buộc, không phải khuyến nghị. Quay xong 108 clip của P1 → chạy extract → đọc `hand_frame_ratio` → chạy pipeline hết một lượt → mới hẹn P2.

Lý do: một lỗi setup có hệ thống (khung quá chật, tay áo che bàn tay, ngược sáng, áo trùng màu da) nếu phát hiện sau khi quay xong cả 5 người thì mất 540 clip. Phát hiện sau P1 thì mất 108.

## Requirements

Setup — kiểm một lần đầu buổi, giữ nguyên cho cả 5 người:

- [ ] Camera cố định, không dịch trong suốt cả 5 buổi.
- [ ] Vạch đứng dán trên sàn; cả 5 người đứng đúng vạch đó.
- [ ] Khung chỉnh theo người cao nhất: giơ hai tay hết tầm vẫn trong khung, thấy từ hông lên đỉnh đầu.
- [ ] Đủ sáng, không ngược sáng. Tay áo không che bàn tay. Áo không trùng màu da.
- [ ] Cả 5 người dùng cùng tay thuận.
- [ ] Ghi khoảng cách người–camera và độ cao camera vào Decisions của file này sau buổi P1.

Mỗi clip train:

- [ ] Đúng một cử chỉ.
- [ ] Bắt đầu: tay xuôi hai bên, đứng yên 1 giây.
- [ ] Múa dứt khoát.
- [ ] Kết thúc: hạ tay xuôi, đứng yên 1 giây, rồi mới cắt.
- [ ] Không cắt/trim bằng tay — code tự trim.

Mỗi người quay **4 vòng**, mỗi vòng múa đủ cả 27 cử chỉ:

| Vòng | Cách múa | Số thứ tự file |
|---|---|---|
| 1, 2 | chậm, đúng tốc độ sẽ demo | `01`, `02` |
| 3 | tốc độ bình thường | `03` |
| 4 | biên độ gọn lại, như lúc múa cho nhanh | `04` |

Quay **theo vòng**, không quay 4 clip cùng một cử chỉ liền nhau. Bốn lần múa liên tiếp cùng một dấu sẽ giống nhau gần như hệt (quán tính vận động), nên 4 clip đó mang gần như cùng một thông tin. Tách chúng ra bằng vòng thì mỗi clip mang phương sai độc lập. Số vòng chính là số thứ tự file nên nhìn tên file biết clip quay ở điều kiện nào.

Vòng 4 biên độ gọn là **bắt buộc**. Lúc demo thật không ai múa chuẩn như lúc quay, và biên độ nhỏ đúng là thứ augmentation không tạo được — nó chỉ xoay ±5°, scale ±6%, jitter σ=0,012.

Clip câu — quay cuối buổi của mỗi người, 4 clip:

- [ ] 2 câu ghép từ 2 cử chỉ, 2 câu ghép từ 3 cử chỉ (chọn từ danh sách nhãn khi đã chốt).
- [ ] Giữa hai cử chỉ: hạ tay ~0,5 giây. Hết câu: hạ tay giữ ~2,5 giây rồi mới cắt.

Cấu trúc lưu file — **người ở cấp trên, nhãn tiếng Việt có dấu**:

```
dataset/raw/P1/Cảm ơn/Cảm ơn 01.mov        ... đến 04
dataset/raw/P1/Xin chào/Xin chào 01.mov
dataset/raw/P2/... , P3/... , P4/... , P5/...
dataset/raw_sentences/P1/cau_01.mov        ... đến cau_04
```

Tên thư mục cử chỉ đi thẳng vào `models/labels.json` và vào câu TTS đọc ra, nên nó **phải** là tiếng Việt có dấu, không phải slug `cam_on`. Xem `docs/specs/pipeline-restructure.md`.

Nghiệm thu — làm ngay khi người đó còn ở chỗ quay:

- [ ] Extract 108 clip vừa quay, xem `hand_frame_ratio`.
- [ ] Clip nào `hand_frame_ratio < 0,5` → quay lại ngay. (27 clip hiện có nằm trong 0,59–0,91, xem `models/metrics.json`.)
- [ ] Clip múa lỡ / tay ra khỏi khung → xóa và quay lại, không giữ để "sau lọc".
- [ ] Đủ 27 nhãn × 4 clip. Thiếu một nhãn ở một người là `raise` ở phía pipeline, không tự bỏ qua.

## Constraints

Mỗi ràng buộc đến từ code đang có, không phải sở thích:

| Ràng buộc | Vì sao | Nguồn |
|---|---|---|
| Khoảng cách camera lúc quay ≈ lúc demo | augmentation chỉ scale 0,94–1,06, phủ được ±6% | `vsl3/features.py:115` |
| Cả 5 người cùng tay thuận | augmentation cố ý không mirroring | `vsl3/features.py:98` |
| Tay rõ trong khung | MediaPipe thấy tay dưới 10% số frame → clip bị raise, loại thẳng | `vsl3/features.py:194-198` |
| 1 giây tay hạ ở hai đầu clip | trim về vùng thấy tay, chỉ chừa margin 4 frame | `vsl3/features.py:200-206` |
| Nhịp 0,5s / 2,5s trong clip câu | khớp `--word-gap 0.45` và `--sentence-gap 2.2` | `realtime.py:77-78` |
| Múa chậm nhưng dưới ~5 giây mỗi cử chỉ | `--max-frames` (mặc định 300) cắt segment rồi **bỏ** phần đuôi của cử chỉ đó | `realtime.py`, `SegmentTracker` |

Bug "một cử chỉ thành hai từ" đã vá (`docs/technical_specs/realtime-segmentation.md`). Ràng buộc còn lại: quá `--max-frames` thì đuôi cử chỉ bị bỏ, nên cứ quay chậm bình thường và đừng để một cử chỉ vượt ~5 giây.

## Decisions

**4 clip mỗi người mỗi nhãn, không phải 6.** Rev 1 chốt 6 vì ở 3 nhãn mỗi fold chỉ test 3 clip nên accuracy nhảy bậc 33%. Ở 27 nhãn mỗi fold test 108 clip, sai một clip là 0,93% — lý do cũ hết hiệu lực. 4 clip × 27 nhãn = 108 clip/người, vừa một buổi.

**Thứ tự P1 → P5 thay vì quyết trước 4 hay 5.** Quay tuần tự làm mọi mốc dừng đều là một dataset hoàn chỉnh.

**Chỉ quay góc thẳng.** Demo chạy trên webcam đặt trước mặt, nên frontal-only là giới hạn đã chọn có chủ ý.

**Không quay lại 27 clip cũ.** Chúng không mã hóa người ký và không ai xác nhận được mapping, nên chúng ở lại chế độ một-người, không vào bộ LOSO. Không xóa, không đoán ID.

Ghi sau buổi P1: khoảng cách camera `___ m`, độ cao camera `___ m`, tay thuận `___`, ngày quay `___`, số người chốt được `___`.

## Out of scope

- Người thứ 6 trở lên (mời được thì mở rộng, không hoãn P1 để đợi).
- Đổi góc camera, đổi bối cảnh, đổi ánh sáng giữa các người.
- Dùng clip câu để train — pipeline một clip một nhãn.
- Chốt danh sách 27 nhãn (việc riêng, làm trước khi hẹn P1).
- Bộ `dataset/legacy/` — không dùng vòng này.
