# As-built — Sửa segmentation realtime + time warp augmentation

_Xong: 2026-08-22; hardening cập nhật 2026-08-28. Contract hiện hành: timestamp + presence-aware feature v3; xem `docs/technical_specs/pipeline-hardening.md`._

## Bug đã sửa: một cử chỉ ra hai từ

`realtime.py` cũ, khi segment chạm `--max-frames`:

```python
if in_segment and len(segment) >= args.max_frames:
    print("Segment reached max length; forcing a boundary.")
    commit_segment()          # dat in_segment = False
```

Vòng lặp kế tiếp, tay **vẫn đang giơ giữa cử chỉ**:

```python
if obs.hands_present:
    if not in_segment:
        segment = []
        in_segment = True     # mo segment moi ngay giua cu chi
```

Nên múa chậm quá `--max-frames` là ra hai từ. Với mục tiêu "múa từ từ thành câu hoàn chỉnh" thì đây là bug bắn trúng giữa mục tiêu.

## Cách sửa: tách state machine ra khỏi `main()`

Toàn bộ luật biên nằm inline trong `main()` (dòng 72–188 bản cũ), tức không test được mà không có camera — đúng cái bẫy vừa ghi vào `docs/GOTCHAS.md`. Nên fix đi kèm việc tách:

```python
class SegmentTracker:
    def feed(self, hands_present, features, now) -> list[np.ndarray] | None
    def force_boundary(self) -> list[np.ndarray] | None      # phim SPACE
    def reset(self) -> None                                  # phim C
```

Không model, không camera, không I/O — nhận một frame, trả về segment đã xong hoặc `None`.

Bản sửa là cờ `awaiting_hand_drop`: khi `max_frames` cắt giữa cử chỉ, cờ bật; mọi frame còn-giơ-tay sau đó bị bỏ; cờ chỉ tắt khi tay thật sự hạ. Segment mới chỉ mở được sau đó.

`--max-frames` mặc định 150 → **300**. Sau khi cắt cưỡng bức, phần đuôi của cử chỉ đó bị **bỏ** (tốt hơn là biến thành một từ rác), nên ngưỡng phải cao hơn cử chỉ chậm nhất định múa. Help string nói rõ điều này.

`main()` giờ chỉ còn ghép: `handle_segment(tracker.feed(...))`, cộng `speak_sentence()` gom ba chỗ phát TTS trùng nhau thành một.

## Giữ nguyên có chủ ý

Frame không thấy tay trong khe ngắn hơn `--word-gap` **vẫn** được nối vào đuôi segment (như bản cũ). Chỗ này lệch với lúc train — `extract_video` trim về vùng thấy tay và chỉ chừa margin 4 frame, còn ở đây có thể nối tới ~13 frame tay hạ. Đó là một vấn đề thật nhưng **không sửa cùng lượt này**: đổi nó là đổi dự đoán, cần đo trước. Ghi vào `docs/CHECKPOINT.md` mục Next.

## Time warp augmentation

Xem `docs/DECISIONS.md` ADR-006 cho lý do đầy đủ. Tóm tắt: **time stretch toàn cục là no-op** ở pipeline này, vì `extract_video` đã resample về đúng 60 frame nên thời lượng tuyệt đối bị chuẩn hóa mất. Thứ thật sự khác nhau giữa người ký là nhịp *bên trong* cử chỉ, nên `time_warp_sequence` bẻ trục thời gian bằng `t → t + w·sin(πt)`, `|w| ≤ 0,30`.

Hai đầu bị ghim nên vị trí bắt đầu/kết thúc không đổi; đơn điệu khi `|w| < 1/π` nên không chạy ngược thời gian.

## Vòng audit — findings đã xử lý

`reviewer` đọc commit `293dee5` và pipeline end-to-end, verdict **BLOCKED**. Hai blocker đều nói về chính bản vá ở trên.

| Finding | Đã làm gì |
|---|---|
| **BLOCKER** `awaiting_hand_drop` bị xóa ngay ở frame **đầu tiên** không thấy tay, trong khi đường đóng segment coi khoảng ngắn hơn `word_gap` là nhiễu. Một frame mất nhận diện (motion blur) là đủ để mở segment mới → vẫn ra hai từ. Receipt: 40 frame giơ tay, một frame trống ở index 20 → **2 segment** | Cờ chỉ xóa khi khoảng không-thấy-tay **đạt đủ `word_gap`** — cùng ngưỡng mà đường đóng dùng |
| **BLOCKER** không test nào phân biệt bản lỗi với bản đúng: mutation "bản vá ứng viên" **sống sót** cả suite | 3 test mới, mỗi cái nhắm một cơ chế. Mutation lại: **5/5 bị bắt** |
| `--max-frames` ghi là "hard cap" mà nhánh nối frame tay-hạ không kiểm cap → đo được segment **98 frame** với `max_frames=10` | Gộp mọi chỗ nối vào một `_append()` duy nhất; cap áp trên mọi đường vào. Kèm chi tiết đúng: cap chạm *trong* đuôi thì tay đã hạ nên **không** bật `awaiting_hand_drop`, không thì từ kế tiếp bị ăn mất |
| `realtime.py` không validate tham số số học — `--max-frames 0` hay `--min-frames > --max-frames` cho ra demo nhận diện được **không gì cả** mà không báo | Kiểm sau `parse_args`: `--confidence` ∈ (0,1], `--word-gap`/`--sentence-gap` > 0, `--min-frames` ≥ 1, `--max-frames` ≥ `--min-frames` |
| `time_warp_sequence` gọi 201 lần `np.interp` chồng lên 201 lần của `resample_sequence`: augment chậm +47%, và thêm một lần low-pass ngoài ý muốn | Gộp thành **một** phép nội suy: `_warp_positions()` sinh vị trí, `_sample_at()` lerp vector hóa cả 201 cột một lần. Đo xen kẽ: **1,891 → 0,618 ms/mẫu = 3,06x** |
| Con số `~79 s/epoch` sai ở 4 chỗ trong docs | Cập nhật thành ~40 s/epoch, LOSO 5 fold ~2,2 h |
| `CHECKPOINT.md` viết "chưa commit gì" ngay trong commit đã sửa chính file đó | Sửa: liệt kê 3 commit trên nhánh và những gì **cố ý** không commit |
| `max_warp` không có guard, giá trị > 1/π làm ánh xạ mất đơn điệu | `_warp_positions` raise nếu `max_warp` ngoài `[0, 1/π)` |

**Chưa làm:** overlay hiện `IDLE` khi tay còn giơ trong lúc `awaiting_hand_drop`, và `SPACE` không có tác dụng trong quãng đó. Cả hai chỉ là hiển thị/UX, không sai kết quả — để lượt sau.

## Lỗi nặng nhất lại không nằm ở commit này

Audit tìm ra một lỗi trong pipeline **đã commit** trước đó: landmark của một clip phụ thuộc clip nào được extract trước nó, lệch tới 1,99 độ-rộng-vai. Chi tiết và cách sửa trong `docs/GOTCHAS.md`; `FEATURES_VERSION` lên 2.

## Ngưỡng đổi từ frame sang giây

`--max-frames`/`--min-frames` đã thành `--max-seconds` (5,0) / `--min-seconds` (0,35), và `SegmentTracker.feed` trả về một `Segment` mang `start_time`/`end_time`/`forced` cùng presence/timestamp tracks. Mọi max-boundary đều chờ hand drop trước khi mở gesture tiếp theo.

Sửa audit sau đó: `end_time` là lúc `word_gap` hết nên không được dùng làm độ dài active. `Segment` hiện có thêm `active_end_time`; `duration = active_end_time - start_time`. Một frame thấy tay không còn tự vượt `min_seconds` chỉ vì chờ 0,45 giây để đóng biên.

Lý do đo được: clip trong `dataset/raw` là 59,96 fps, còn MediaPipe trên máy này chạy 19,5–21,5 fps (46,6 ms/frame ở 1080p). Nên `--max-frames 300` là **5,0 giây** khi đọc file nhưng **14,0 giây** trên webcam. Nguyên mẫu đo cùng một cử chỉ 8 giây ở hai nhịp frame: ngưỡng frame cắt ở 60 fps mà **không** cắt ở 21,5 fps; ngưỡng giây cắt đúng 5,0 giây ở cả hai. Test `test_cap_is_the_same_duration_at_any_frame_rate` khóa điều đó.

Quan trọng cho bước sau: không có việc này thì `vslr-sentence` (đo ghép câu từ file) không thể tái lập hành vi demo.

## File đã đổi

| File | Đổi gì |
|---|---|
| `src/prototype_3_gestures/realtime.py` | thêm `SegmentTracker`; `main()` bỏ `segment`/`in_segment`/`last_hand_time`/`commit_segment`, thêm `handle_segment` + `speak_sentence`; `--max-frames` 150 → 300 kèm help |
| `src/prototype_3_gestures/vsl3/features.py` | thêm `_sample_at` / `_warp_positions` / `time_warp_sequence`; `augment_sequence` gộp crop-resample và warp thành một phép nội suy; `extract_video` dùng instance MediaPipe riêng mỗi clip; `FEATURES_VERSION` 1 → 2 |
| `tests/test_core.py` | 24 → **35 test** (7 test `SegmentTracker`, 2 test time warp, 2 test cách ly extractor) |
| `docs/DECISIONS.md` | ADR-006 |

## Verify

```powershell
python -m pytest          # 35 passed
vslr-camera --no-tts      # can camera, chua chay duoc trong moi truong nay
```

Đã chạy:

- 30 test pass. `main()` không còn `commit_segment` / `in_segment =` / `last_hand_time` (kiểm bằng `inspect.getsource`).
- Mutation testing trên `SegmentTracker`, **3/3 bị bắt**: bỏ `awaiting_hand_drop` (tái hiện đúng bug cũ) → `test_new_gesture_starts_only_after_hands_drop` + `test_slow_gesture_does_not_become_two_words` đỏ; bỏ nối frame trong khe ngắn → `test_word_gap_closes_a_segment_and_short_blips_do_not` đỏ; `reset()` không xóa segment → `test_force_boundary_and_reset` đỏ.
- Time warp: 500 seed đều đơn điệu; điểm giữa của một ramp thời gian chạy 0,211 → 0,807 (gốc 0,508); hai đầu giữ nguyên 0,0 và 1,0.

**Chưa verify:** `vslr-camera` thật — không có camera. Toàn bộ luật biên được kiểm qua `SegmentTracker` chứ không qua một lần chạy camera, và phần ghép trong `main()` (đọc `tracker.awaiting_hand_drop` để in cảnh báo, phím C/S/SPACE) vẫn chưa có test.

## Watch-outs

- Sau cắt cưỡng bức ở `--max-seconds`, đuôi cử chỉ **bị bỏ**. Nếu demo thấy cử chỉ chậm bị nhận sai, tăng `--max-seconds` trước khi nghi model.
- Phân phối augment đã đổi (thêm warp), nên số đo trước và sau thay đổi này không so sánh trực tiếp được.
- `--word-gap` vẫn nối frame tay hạ vào đuôi segment — lệch với lúc train, xem mục "Giữ nguyên có chủ ý".

## Activity gate cho tay nghỉ vẫn nhìn thấy

MediaPipe có thể tiếp tục trả hand landmarks khi người đứng hạ tay dọc thân. Presence vì vậy không
còn là tín hiệu mở/đóng segment. `gesture_activity_from_features()` dùng landmark đã chuẩn hóa và
coi frame active khi ít nhất một cổ tay cao hơn đường hông trung bình `0,5` lần bề rộng vai. Nếu hông
không nằm trong khung hoặc landmark pose không hợp lệ, hàm fallback về presence để camera crop nửa
người không bị mất toàn bộ cử chỉ.

`SegmentTracker` giữ riêng activity mask và presence mask. Activity điều khiển state machine,
`active_end_time`, `word_gap` và `--max-seconds`; presence vẫn điều khiển dữ liệu bàn tay thực tế đưa
vào model. Một rolling buffer giữ tối đa 1,0 giây tay-nghỉ có quan sát trước khi active và tối đa 0,5
giây sau active làm transition context. Frame hoàn toàn không thấy tay không được đưa vào pre-roll.

Verify 2026-09-03:

- `python -m pytest -q`: 123 passed, 45 subtests passed.
- Ba clip QIPEDC ngoài tập train: 3/3 đúng (`Siêu thị` 98,5%, `Tạm biệt` 82,9%, `Xin lỗi` 97,4%).
- Một clip `001` của mỗi 25 nhãn P01: 25/25 đúng, không segment nào `forced`.
- Chưa verify camera thật sau thay đổi; đây là bước nghiệm thu tiếp theo.
