# Spec — Quay dataset V1: 30 cử chỉ, 4 người

_Viết: 2026-08-22. Rev 3: 2026-08-24 — manifest đã chốt. Trạng thái: chưa quay._

## Goal

720 clip train (4 người × 30 nhãn × 6 clip) + 16 clip câu (4 người × 4 clip), đủ để chạy leave-one-signer-out 4 fold và thử luồng demo bằng dữ liệu thật.

Danh sách và thứ tự 30 nhãn V1 đã chốt trong `dataset/labels.txt`.

## Quy mô bốn người

Quay **xong hẳn từng người** theo thứ tự P01 → P04. Mỗi người hoàn thành đủ 180 clip mới sang người sau.

LOSO bốn fold cho phép kiểm mỗi lượt trên một người chưa xuất hiện trong tập train của fold đó. Đây là bằng chứng phù hợp cho prototype/demo, chưa phải bằng chứng đủ mạnh để tuyên bố hệ thống tổng quát cho mọi người ký.

Nếu sau demo cần tăng khả năng chạy với người mới, bổ sung P05 trở đi vào một phiên bản dữ liệu mới mà không đổi 30 class ID hiện có.

## Quay P01 trước, kiểm, rồi mới quay tiếp

Bắt buộc, không phải khuyến nghị. Quay xong 180 clip của P01 → chạy `vslr-check` → đọc `hand_frame_ratio` → mới quay P02.

Lý do: một lỗi setup có hệ thống nếu phát hiện sau khi quay xong cả 4 người thì mất 720 clip. Phát hiện sau P01 thì mất tối đa 180 clip.

## Requirements

Setup — kiểm một lần đầu buổi, giữ nguyên cho cả 4 người:

- [ ] Camera cố định, không dịch trong suốt cả 4 buổi.
- [ ] Vạch đứng dán trên sàn; cả 4 người đứng đúng vạch đó.
- [ ] Khung chỉnh theo người cao nhất: giơ hai tay hết tầm vẫn trong khung, thấy từ hông lên đỉnh đầu.
- [ ] Đủ sáng, không ngược sáng. Tay áo không che bàn tay. Áo không trùng màu da.
- [ ] Cả 4 người dùng cùng tay thuận.
- [ ] Ghi khoảng cách người–camera và độ cao camera vào Decisions của file này sau buổi P01.

Mỗi clip train:

- [ ] Đúng một cử chỉ.
- [ ] Bắt đầu: tay xuôi hai bên, đứng yên 1 giây.
- [ ] Múa dứt khoát.
- [ ] Kết thúc: hạ tay xuôi, đứng yên 1 giây, rồi mới cắt.
- [ ] Không cắt/trim bằng tay — code tự trim.

Mỗi người quay **6 vòng**, mỗi vòng múa đủ cả 30 cử chỉ:

| Vòng | Cách múa | Số thứ tự file |
|---|---|---|
| 1, 2 | tự nhiên, đúng nghĩa | `001`, `002` |
| 3, 4 | tự nhiên sau khi nghỉ hoặc đổi phiên | `003`, `004` |
| 5, 6 | tự nhiên, cho phép sai khác nhỏ về tay/cơ thể | `005`, `006` |

Quay **theo vòng**, không quay cả 6 clip cùng một cử chỉ liền nhau. Có thể quay hai vòng, nghỉ hoặc đổi phiên rồi quay hai vòng tiếp. Mỗi clip vẫn múa tự nhiên; không cần cố diễn cực nhanh hay cực chậm. Số vòng chính là số thứ tự file.

Khác biệt nhỏ giữa các vòng là có ích, nhưng không được đổi ý nghĩa hoặc rút gọn động tác để tạo biến thiên giả.

Clip câu — quay cuối buổi của mỗi người, 4 clip:

- [ ] 2 câu ghép từ 2 cử chỉ, 2 câu ghép từ 3 cử chỉ (chọn từ danh sách nhãn khi đã chốt).
- [ ] Giữa hai cử chỉ: hạ tay ~0,5 giây. Hết câu: hạ tay giữ ~2,5 giây rồi mới cắt.

Cấu trúc lưu file — **người ở cấp trên, nhãn tiếng Việt có dấu**:

```
dataset/recordings_v1/P01/Cảm ơn/001.mov        ... đến 006.mov
dataset/recordings_v1/P01/Xin chào/001.mov
dataset/recordings_v1/P02/... , P03/... , P04/...
dataset/raw_sentences/P01/cau_01.mov            ... đến cau_04.mov
```

Tên thư mục cử chỉ đi thẳng vào `models/labels.json` và vào câu TTS đọc ra, nên nó **phải** là tiếng Việt có dấu, không phải slug `cam_on`. Xem `docs/specs/pipeline-restructure.md`.

Nghiệm thu — làm ngay khi người đó còn ở chỗ quay:

- [ ] Kiểm 180 clip vừa quay bằng `vslr-check`, xem `hand_frame_ratio`.
- [ ] Clip nào `hand_frame_ratio < 0,5` → quay lại ngay. (27 clip hiện có nằm trong 0,59–0,91, xem `models/metrics.json`.)
- [ ] Clip múa lỡ / tay ra khỏi khung → xóa và quay lại, không giữ để "sau lọc".
- [ ] Đủ 30 nhãn × 6 clip. Thiếu một nhãn ở một người làm check/train thất bại, không tự bỏ qua.

## Constraints

Mỗi ràng buộc đến từ code đang có, không phải sở thích:

| Ràng buộc | Vì sao | Nguồn |
|---|---|---|
| Khoảng cách camera lúc quay ≈ lúc demo | augmentation chỉ scale 0,94–1,06, phủ được ±6% | `vsl3/features.py:115` |
| Cả 4 người cùng tay thuận | augmentation cố ý không mirroring | `vsl3/features.py:98` |
| Tay rõ trong khung | MediaPipe thấy tay dưới 10% số frame → clip bị raise, loại thẳng | `vsl3/features.py:194-198` |
| 1 giây tay hạ ở hai đầu clip | trim về vùng thấy tay, chỉ chừa margin 4 frame | `vsl3/features.py:200-206` |
| Nhịp 0,5s / 2,5s trong clip câu | khớp `--word-gap 0.45` và `--sentence-gap 2.2` | `realtime.py:77-78` |
| Múa chậm nhưng dưới ~5 giây mỗi cử chỉ | `--max-seconds` (mặc định 5,0) cắt segment rồi **bỏ** phần đuôi của cử chỉ đó | `realtime.py`, `SegmentTracker` |

Bug "một cử chỉ thành hai từ" đã vá (`docs/technical_specs/realtime-segmentation.md`). Ràng buộc còn lại: quá `--max-seconds` thì đuôi cử chỉ bị bỏ, nên cứ quay chậm bình thường và đừng để một cử chỉ vượt ~5 giây.

## Decisions

**6 clip mỗi người mỗi nhãn.** Đây là kế hoạch V1 do chủ dự án chốt: 180 clip/người, 720 clip tổng. Sáu clip được tách thành ba phiên hai vòng để tăng biến thiên tự nhiên.

**Bốn người P01 → P04.** Quay xong và nghiệm thu từng người trước khi chuyển sang người tiếp theo.

**Chỉ quay góc thẳng.** Demo chạy trên webcam đặt trước mặt, nên frontal-only là giới hạn đã chọn có chủ ý.

**Không đưa 27 clip cũ vào V1.** Chúng không mã hóa người ký và không ai xác nhận được mapping, nên chúng ở lại chế độ một-người, không vào bộ LOSO. Không xóa, không đoán ID.

Ghi sau buổi P01: khoảng cách camera `___ m`, độ cao camera `___ m`, tay thuận `___`, ngày quay `___`, số người chốt được `___`.

## Out of scope

- Người thứ 5 trở lên; nếu bổ sung thì tạo phiên bản dữ liệu mới, giữ nguyên class ID.
- Đổi góc camera, đổi bối cảnh, đổi ánh sáng giữa các người.
- Dùng clip câu để train — pipeline một clip một nhãn.
- Thêm hoặc đổi nhãn sau khi V1 đã bắt đầu quay.
- Bộ `dataset/legacy/` — không dùng vòng này.
