# Spec — Đóng khoảng cách "quay clip → test"

_Viết: 2026-08-22. **Trạng thái: rev 1, `spec-critic` trả BLOCKED — CHƯA ĐƯỢC IMPLEMENT.** 5 blocker đã có câu trả lời (dưới đây) nhưng thân spec chưa viết lại theo. Đừng code từ bản này._

## Kết quả cổng spec-critic — 5 blocker và câu trả lời

| Blocker | Trả lời (chưa gộp vào thân spec) |
|---|---|
| `vslr-check` gọi `validate_clips` sẽ abort giữa buổi quay: quay theo vòng nên sau vòng 1 mọi nhãn có đúng 1 clip, mà `validate_clips` đòi ≥ 2. Và "cặp còn thiếu" luôn rỗng vì tập nhãn được suy **từ chính cây** | `vslr-check` **không gọi `validate_clips`** — nó là công cụ báo cáo, không phải cổng. Tập nhãn mong đợi đến từ **file `dataset/labels.txt`**, không suy từ cây. Đó cũng chính là thứ dự án đang thiếu: một danh sách nhãn viết ra giấy |
| `--expect-from-filename` xuất hiện ở khối "Định nghĩa xong" mà Requirements không định nghĩa, và `cau_01.mov` không mang được nội dung câu | Xóa cờ đó. Ground truth vào manifest `dataset/raw_sentences/sentences.csv` (`person,clip,words`), điền lúc quay |
| Đồng hồ offline ≠ đồng hồ demo. `--word-gap`/`--sentence-gap` tính bằng giây nên chuyển được; `--max-frames`/`--min-frames` đếm frame nên **không** | Đổi hai cờ đó sang **giây**. Đo được: clip 59,96 fps, MediaPipe 19,5–21,5 fps trên máy này → `--max-frames 300` là 5,0 giây offline nhưng 14,0 giây trên webcam. Nguyên mẫu xác nhận ngưỡng giây làm hai đường trùng hành vi |
| "dùng lại `SegmentTracker`" và "không refactor `realtime.main()`" không cùng đúng được: lọc `min_frames`, ngưỡng confidence, ghép câu đều là closure trong `main()` | Kéo phần đó ra. **Refactor `realtime.main()` vào scope.** Một bản duy nhất có thẩm quyền |
| `predictions[]` đưa qua đâu: nhét vào history → 21.600 dòng + vỡ 2 test; trả 3-tuple → vỡ 3 chỗ unpack; evaluate lần hai → val loader dựng hai nơi | Gọi **evaluate lần hai** trong `main()`, tách phần dựng val loader thành helper dùng chung. `train_model` giữ 2-tuple → không test nào vỡ. `confidence` = **softmax-max**, khớp `predict_sequence` |

Ba should-fix cũng phải gộp vào khi viết lại: `suspect_clips` "dưới trung vị" **tự thỏa mãn** (một nửa tập nào cũng dưới trung vị của chính nó) → dùng ngưỡng tuyệt đối 0,5; `FAIL` gộp ba nguyên nhân ba hành động → tách; `vslr-sentence` thiếu `--model` và thiếu kiểm `--expect` với `labels`.

Và một điều chỉnh về tiền đề: critic đúng khi nói **lỗ hổng 2 không phải bức tường** — `--loso` đã cho ra con số có receipt. Per-label chỉ làm việc hành động *nhanh hơn*. Chỉ `predictions[]` và dòng tổng kết console là load-bearing; ba khóa report còn lại là hàm thuần của `predictions[]` + `extraction[]` nên đừng lưu.


## Goal

Sau buổi quay, người dùng chỉ chạy lệnh — không viết thêm code, không sửa file tay — và nhận được:

1. **Ngay tại chỗ quay**: clip nào tệ, phải quay lại clip nào, trước khi người đó về.
2. **Sau khi quay đủ**: một con số kèm **nhãn nào sai** và **clip nào sai**, đủ để quyết định làm gì tiếp.
3. **Cho mục tiêu thật của dự án**: câu ghép từ clip câu có đúng không, đo được, không cần camera.

Hiện cả ba đều thiếu. Đây không phải bug — pipeline chạy đúng — mà là ba việc nó không làm.

## Lỗ hổng 1: không có lệnh nghiệm thu tại chỗ quay — **ĐÃ LÀM** (`vslr-check`)

_Xong 2026-08-22. Xem `docs/technical_specs/check-tool.md`. Phần dưới là spec gốc, giữ để đối chiếu._


`docs/specs/dataset-recording.md` bắt "trích xuất 108 clip vừa quay, xem `hand_frame_ratio`, clip nào < 0,5 quay lại ngay". Nhưng **không có lệnh nào làm việc đó**. Muốn xem `hand_frame_ratio` hiện phải chạy cả lượt train.

### `vslr-check --data-dir DIR`

- [ ] Entry point mới trong `pyproject.toml` → `prototype_3_gestures.check:main`.
- [ ] Dùng lại `discover_clips`, `validate_clips`, `extract_all` từ `prepare_train` — **không train gì**, không ghi `models/`.
- [ ] Warm chung `--cache-dir` với `vslr-train`, nên clip đã check không phải extract lại lúc train.
- [ ] In bảng sắp theo `hand_frame_ratio` tăng dần: `người | nhãn | file | sampled | trimmed | hand_ratio | trạng thái`.
- [ ] Trạng thái: `FAIL` nếu extract raise, `QUAY LẠI` nếu `hand_frame_ratio < --min-hand-ratio` (mặc định 0,5), `ok` còn lại.
- [ ] Cuối bảng: đếm clip mỗi (người, nhãn), liệt kê **cặp còn thiếu** để biết còn phải quay gì.
- [ ] Exit code 1 nếu có clip `FAIL`/`QUAY LẠI`, để dùng được trong script.
- [ ] Chạy được trên một người: `vslr-check --data-dir dataset/raw/P1` — nhưng cây đó là `<nhãn>/*.mov` chứ không có cấp người. Xem "Quyết định" bên dưới.

### Cảnh báo cây trộn

`dataset/raw/` hiện phẳng (`<nhãn>/*.mov`). Khi quay mới vào `dataset/raw/P1/`, cây thành nửa cũ nửa mới, và `discover_clips` **âm thầm bỏ qua** thư mục cấp 1 nào chỉ chứa file (không có thư mục nhãn con) — tức 27 clip cũ vô hình mà không ai được báo.

- [ ] `discover_clips` warn khi một thư mục cấp 1 chứa file video trực tiếp: nói rõ nó bị bỏ qua và vì sao.

## Lỗ hổng 2: LOSO ra một con số, không nói được nhãn nào sai

`train_model` chỉ đếm `argmax` rồi cộng (`prepare_train.py:371`). Ở 27 nhãn × 4 clip × 5 người, mỗi fold test **108 clip mà báo một số**. Không phân biệt được "sai rải đều 27 nhãn" với "25 nhãn hoàn hảo, 2 nhãn chết hẳn" — hai tình huống cần hai hành động hoàn toàn khác nhau.

- [ ] `accuracy()` (đổi tên thành `evaluate()`) trả thêm danh sách `(index, target, predicted, confidence)` cho từng mẫu.
- [ ] `loso_report.json` mỗi fold thêm `predictions[]`: `{video, person, label, predicted, confidence, correct}`. Video path lấy từ clip thật, nên join được với `extraction[]`.
- [ ] Report thêm `per_label_accuracy` gộp mọi fold: `{nhãn: {clip, đúng, accuracy}}`.
- [ ] Report thêm `worst_labels`: 5 nhãn accuracy thấp nhất, kèm nhãn nào bị nhận nhầm thành nhãn nào (cặp confusion hay gặp nhất).
- [ ] Report thêm `suspect_clips`: clip sai **và** `hand_frame_ratio` dưới trung vị — tức clip vừa sai vừa quay tệ, ứng viên quay lại. Join qua `video`.
- [ ] In ra console: accuracy trung bình, fold tệ nhất, **5 nhãn tệ nhất**, và số clip nghi ngờ. Đó là bốn thứ quyết định được hành động.

Ràng buộc: `predictions[]` ở 540 clip × 5 fold = 540 dòng (mỗi clip test đúng một lần trong đúng một fold). Không phình.

## Lỗ hổng 3: mục tiêu của dự án là câu, mà không có gì đo câu

Mục tiêu là "múa từ từ thành câu hoàn chỉnh". Pipeline chỉ đo **một clip một nhãn**. `dataset/raw_sentences/` trong spec quay **không có code nào đọc**. Ghép câu chỉ tồn tại trong `realtime.py`, không test, không số.

Tức phần được đo và phần được demo là hai thứ khác nhau — đúng lỗi mà repo tham chiếu `photienanh/Vietnamese-Sign-Language-Recognition` mắc phải (demo trông mượt vì không bao giờ nói "không biết", và không có con số nào tồn tại).

### `vslr-sentence --clip path.mov [--expect "Xin chào,Cảm ơn"]`

- [ ] Entry point mới → `prototype_3_gestures.sentence:main`.
- [ ] Đọc video bằng `HolisticExtractor.process_frame` từng frame, dựng `now` từ số frame và FPS của file (không dùng `time.monotonic` — đây là offline, phải tái lập được).
- [ ] Đưa qua **chính** `SegmentTracker` mà `realtime.py` dùng, cùng `--word-gap` / `--max-frames` / `--min-frames`, rồi `classify_segment` + `should_accept_prediction`. Không được nhân bản logic.
- [ ] In từng segment: `frame bắt đầu–kết thúc | nhãn | confidence | nhận/loại`, rồi câu ghép cuối.
- [ ] `--expect` cho danh sách từ mong đợi → in khớp/lệch và exit 1 nếu lệch. Đây là thứ biến demo thành test.
- [ ] `--dir DIR` chạy cả thư mục clip câu và in tổng: bao nhiêu câu đúng hoàn toàn, bao nhiêu sai một từ, bao nhiêu sai số lượng từ.

Cái này cũng là cách duy nhất hiện có để **đo `--confidence` / `--word-gap` bằng dữ liệu** thay vì bốc số: chạy lại cùng bộ clip câu ở vài giá trị ngưỡng rồi xem cái nào ghép đúng nhiều nhất.

## Constraints

| Ràng buộc | Vì sao |
|---|---|
| `vslr-check` và `vslr-sentence` **không** ghi vào `models/` | chỉ `vslr-train` không có `--loso` được sinh artifact |
| `vslr-sentence` dùng lại `SegmentTracker`, không tự viết luật biên | hai bản luật biên là hai hành vi khác nhau lúc demo |
| Cả hai lệnh mới dùng chung `--cache-dir` với `vslr-train` | check xong không phải extract lại lúc train |
| Không đổi `FEATURE_DIM` / `SEQUENCE_LENGTH` / `augment_sequence` | không phải việc của spec này |
| `predictions[]` phải mang `video` path | không có nó thì không join được với `extraction[]`, mất `suspect_clips` |
| Không tune `--epochs` / `--confidence` tự động trong `--loso` | người giữ ra trong LOSO là tập test; xem ADR-004 |

## Quyết định

**Ba lệnh, ba việc rõ ràng.** `vslr-check` (quay xong kiểm ngay), `vslr-train` (đo bằng `--loso`, hoặc sản xuất artifact), `vslr-sentence` (đo câu). Không nhồi thêm cờ vào `vslr-train`: người chạy `vslr-check` đang ở chỗ quay làm việc khác hẳn, và lệnh có chữ "train" ở đó gây hiểu nhầm là đang train.

**`vslr-check` nhận cả cây một người và cây nhiều người.** Ở chỗ quay chỉ có P1, cây là `dataset/raw/P1/<nhãn>/`. Nên `vslr-check --data-dir dataset/raw` (thấy P1) và `vslr-check --data-dir dataset/raw/P1 --single-person` (coi cấp 1 là nhãn). Cờ tường minh, không đoán theo cấu trúc — đoán sai là gán nhãn thành tên người.

**Không tự động quay lại kết luận từ `suspect_clips`.** Nó chỉ liệt kê clip vừa sai vừa quay tệ. Quyết định quay lại là của người, vì "sai" cũng có thể là model dở chứ không phải clip dở.

**`worst_labels` báo cặp confusion, không báo cả ma trận.** Ma trận 27×27 in ra console là vô dụng. Nếu cần cả ma trận thì dựng từ `predictions[]` — dữ liệu đã đủ.

## Out of scope

- Tự tune `--confidence` / `--word-gap` / `--epochs`. Spec này chỉ làm cho việc đo tay khả thi.
- Tách `main()` của `prepare_train.py` (157 dòng) và `realtime.py` (109 dòng) — nợ kỹ thuật thật, nhưng là refactor riêng.
- Sửa chuyện `--word-gap` nối frame tay-hạ vào đuôi segment (lệch với lúc train).
- Pretrain trên `dataset/legacy/`.
- Ghép câu theo ngữ pháp VSL (hiện `" ".join` theo đúng thứ tự múa).
- Train lại checkpoint.

## Định nghĩa "xong"

Chạy được đúng chuỗi này, không sửa code ở giữa:

```powershell
$env:PYTHONIOENCODING = "utf-8"

# 1. quay P1 xong, kiem ngay tai cho
vslr-check --data-dir dataset/raw --min-hand-ratio 0.5        # exit 1 neu co clip phai quay lai

# 2. quay du 5 nguoi -> do
vslr-train --data-dir dataset/raw --loso --num-workers 4      # in 5 nhan te nhat + clip nghi ngo

# 3. san xuat artifact de demo
vslr-train --data-dir dataset/raw --num-workers 4

# 4. do phan ghep cau
vslr-sentence --dir dataset/raw_sentences --expect-from-filename

# 5. demo
vslr-camera --no-tts
```
