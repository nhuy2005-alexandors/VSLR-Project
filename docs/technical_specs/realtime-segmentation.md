# As-built — Sửa segmentation realtime + time warp augmentation

_Xong: 2026-08-22. Test: **30 passed** (3/3 mutation của segmentation bị bắt). Không có spec riêng: hai việc này đều nằm ở mục Out of scope của `docs/specs/pipeline-restructure.md`._

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

## File đã đổi

| File | Đổi gì |
|---|---|
| `src/prototype_3_gestures/realtime.py` | thêm `SegmentTracker`; `main()` bỏ `segment`/`in_segment`/`last_hand_time`/`commit_segment`, thêm `handle_segment` + `speak_sentence`; `--max-frames` 150 → 300 kèm help |
| `src/prototype_3_gestures/vsl3/features.py` | thêm `time_warp_sequence`, gọi trong `augment_sequence` sau crop |
| `tests/test_core.py` | 24 → **30 test** (4 test `SegmentTracker`, 2 test time warp) |
| `docs/DECISIONS.md` | ADR-006 |

## Verify

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m pytest          # 30 passed
vslr-camera --no-tts      # can camera, chua chay duoc trong moi truong nay
```

Đã chạy:

- 30 test pass. `main()` không còn `commit_segment` / `in_segment =` / `last_hand_time` (kiểm bằng `inspect.getsource`).
- Mutation testing trên `SegmentTracker`, **3/3 bị bắt**: bỏ `awaiting_hand_drop` (tái hiện đúng bug cũ) → `test_new_gesture_starts_only_after_hands_drop` + `test_slow_gesture_does_not_become_two_words` đỏ; bỏ nối frame trong khe ngắn → `test_word_gap_closes_a_segment_and_short_blips_do_not` đỏ; `reset()` không xóa segment → `test_force_boundary_and_reset` đỏ.
- Time warp: 500 seed đều đơn điệu; điểm giữa của một ramp thời gian chạy 0,211 → 0,807 (gốc 0,508); hai đầu giữ nguyên 0,0 và 1,0.

**Chưa verify:** `vslr-camera` thật — không có camera. Toàn bộ luật biên được kiểm qua `SegmentTracker` chứ không qua một lần chạy camera, và phần ghép trong `main()` (đọc `tracker.awaiting_hand_drop` để in cảnh báo, phím C/S/SPACE) vẫn chưa có test.

## Watch-outs

- Sau cắt cưỡng bức ở `--max-frames`, đuôi cử chỉ **bị bỏ**. Nếu demo thấy cử chỉ chậm bị nhận sai, tăng `--max-frames` trước khi nghi model.
- Phân phối augment đã đổi (thêm warp), nên số đo trước và sau thay đổi này không so sánh trực tiếp được.
- `--word-gap` vẫn nối frame tay hạ vào đuôi segment — lệch với lúc train, xem mục "Giữ nguyên có chủ ý".
