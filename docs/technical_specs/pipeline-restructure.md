# As-built — Tái cấu trúc pipeline train

_Xong: 2026-08-22. Spec: `docs/specs/pipeline-restructure.md` (rev 2, sau `spec-critic`). Test: **24 passed** (7/7 mutation bị bắt). Đã qua một vòng `reviewer` (verdict: BLOCKED → 2 blocker + 6 should-fix, đã xử lý hết bên dưới)._

## Đã xây gì

`vslr-train` tách thành hai chế độ dùng chung một lệnh, và data model mang ID người.

```bash
vslr-train --data-dir dataset/raw --loso     # ĐO   -> models/loso_report.json, không ghi checkpoint
vslr-train --data-dir dataset/raw            # SHIP -> gesture_lstm.pt + labels.json + metrics.json
vslr-train --video "Nhãn=path.mov" ...       # chế độ một-người, --loso bị từ chối
```

## Luồng end-to-end

```
--data-dir DIR                              --video LABEL=PATH
  ↓ discover_clips()                          ↓ parse_video_specs()
  DIR/<person>/<gesture>/*.mov|*.mp4          person = "unknown"
  ↓
validate_clips(require_signer_split=args.loso)
  LOSO: cần ≥ 2 người, và mọi cặp (nhãn, người) phải có mặt → raise kèm danh sách cặp thiếu
  không LOSO: cần ≥ 2 clip mỗi nhãn
  ↓
extract_all() → extract_with_cache() mỗi clip
  cache hit  → đọc .npz (kèm sampled_frames / trimmed_frames / hand_frame_ratio)
  cache miss → HolisticExtractor.extract_video → ghi staging → rename vào chỗ
  clip lỗi   → gom vào failures, đi tiếp; abort nếu > 5%
  ↓
validate_clips() LẦN HAI trên clip còn lại (clip bị drop có thể phá invariant)
  ↓
labels = label_order(clips);  targets = [label_to_idx[c.label] ...]
  ↓
├─ --loso: mỗi người một fold
│    signer_split() → train_model(val_sequences=…)  → fold accuracy trên clip THẬT (không augment)
│    → loso_report.json: mean, worst_fold, folds[].history, extraction[], failed_clips
│
└─ ship: train_model() trên toàn bộ clip, không val
     → save_checkpoint(config có "pooling" + "features_version")
     → metrics.json (mode=ship, signer_split=false, warning nói rõ không có holdout)
```

## File đã đổi

| File | Đổi gì |
|---|---|
| `src/prototype_3_gestures/prepare_train.py` | viết lại. Thêm `Clip`, `discover_clips`, `validate_clips`, `signer_split`, `cache_key`, `extract_with_cache`, `GestureDataset`, `train_model`, `extract_all`, `build_parser`. Xóa `build_dataset`, `source_holdout_split`, `best_state`/`best_epoch`/`final_fit_epochs`, khối refit, ghi npz |
| `src/prototype_3_gestures/vsl3/model.py` | `pool_sequence()` + tham số `pooling`; `load_checkpoint` mặc định `legacy_last_step` khi thiếu khóa |
| `src/prototype_3_gestures/vsl3/features.py` | ban đầu thêm `FEATURES_VERSION = 1`; hiện là **2** sau khi cách ly MediaPipe theo clip |
| `tests/test_core.py` | 7 test → **17 test** |

Không đổi: `normalize_landmarks`, `augment_sequence`, `resample_sequence`, `HolisticExtractor`, `realtime.py`.

## Vòng reviewer — findings đã xử lý

`reviewer` xác nhận phần lõi sạch: **không có đường leakage** trong `signer_split`, pooling đúng khi đối chiếu với `h_n`, val/test không bao giờ augment, mode tách đúng, cache warm trả stats giống hệt. Hai blocker đều nằm ở **nhánh xử lý lỗi** — đúng phần mà mấy lần verify đầu không chạm tới.

| Finding | Đã làm gì |
|---|---|
| **BLOCKER** nhãn mất **hết** clip thì biến khỏi class set; `validate_clips` chỉ thấy survivor nên không phát hiện. 27 nhãn thành 26 mà vẫn báo con số dưới tên 27 nhãn | Snapshot `expected_labels` **trước** extract; `missing_labels_after_drop()` so lại sau khi drop, `SystemExit` kèm danh sách nhãn mất. Có unit test |
| **BLOCKER** cache `.npz` hỏng bị tính là "clip quay tệ", không retry, và người quay bị bảo đi quay lại một clip tốt | `np.load` bọc try/except → `unlink()` entry hỏng → rơi xuống extract lại. Ghi cache thành best-effort (`_write_cache_entry`), lỗi IO chỉ warn, không vào danh sách clip lỗi. Có unit test dựng cache hỏng thật |
| `--epochs 0` ship checkpoint random-init, exit 0; `--epochs 0 --loso` crash `IndexError` | `type=positive_int` cho `--epochs`/`--batch-size`, `non_negative_int` cho `--augment`/`--seed`/`--num-workers` |
| Re-validate sau drop nằm ngoài try/except → traceback trần sau cả lượt extract | Bọc vào `SystemExit` kèm số clip đã drop |
| Augment tính lại mỗi epoch dù seed làm nó giống hệt: ~40 s/epoch ở 540 clip, ~2,2 h cho LOSO | Thêm `--num-workers` (+ `persistent_workers`). Đã chạy thật `--num-workers 2` trên Windows, exit 0 |
| Report không tái lập được: thiếu `seed`, `learning_rate`, `batch_size`, timestamp; và hai file không chứng minh được là cùng một dữ liệu | `run_metadata()` dùng chung cho cả hai file, thêm `created_at` + `data_fingerprint` (sha1 trên `(person, label, path, mtime)` đã sort). Đã kiểm: hai file cùng fingerprint `66149052a58a…` |
| `features_version` ghi vào checkpoint mà không ai đọc | `load_checkpoint` so với hiện tại; lệch giờ **raise mặc định**, chỉ opt-in legacy mới warn |
| Checkpoint mất khóa `pooling` mispredict âm thầm | `load_checkpoint` warn khi thiếu khóa. Đã kiểm: checkpoint cũ → 1 cảnh báo; checkpoint mới → 0 |
| Test pool nửa vô nghĩa: `pooled[0,4:] == zeros` mà fixture `out[:,0,4:]` cũng là zeros, nên bản `zeros_like` cũng pass | Ramp bắt đầu từ 1 → t=0 ra 10.0. Thêm `test_pool_matches_lstm_final_hidden_states` chạy trên `nn.LSTM` thật, so với `cat(h_n[0], h_n[1])` |
| Test cache key không đổi `feature_dim`/`sequence_length` | Đã thêm hai assert |
| `train_loss` (label_smoothing 0.03) và `val_loss` (CE thường) khác thang mà nằm cùng một row | Đổi tên `train_loss_smoothed` / `val_loss_plain_ce`, `note` trong report nói rõ đừng đọc chung |
| Staging file không có pid → hai run song song đụng nhau | `{key}.{pid}.tmp` |
| `mean_accuracy` là mean không trọng số, `note` không nói macro hay pooled | Đổi tên `macro_mean_accuracy`, **thêm** `pooled_accuracy` kèm số clip |
| `checkpoint_config` hardcode `pooling`/`hidden_size`/`num_layers` | Đọc từ `model.lstm` / `model.pooling` |

**Chưa làm:** ghi `pooling` vào `model_state` như buffer (reviewer đề xuất để khóa cứng). Warn đã đủ cho vòng này; nếu sau này có nhiều checkpoint cùng lưu hành thì làm.

## Quyết định phát sinh lúc code

**Bỏ early stopping ở CẢ HAI chế độ, không chỉ chế độ ship.** Spec rev 2 lúc đầu định lấy `--epochs` từ median epoch tốt nhất của LOSO. Đó là chọn siêu tham số trên tập test, vì người giữ ra trong LOSO chính là test. `--epochs` thành hằng chọn trước (mặc định 40), report ghi rõ nó không được tune trên fold. Xem `docs/GOTCHAS.md`.

**Gộp hai guard LOSO thành một.** Bản đầu có cả "≥ 2 người mỗi nhãn" và "không thiếu cặp (nhãn, người)". Với ≥ 2 người và không cặp nào thiếu thì điều kiện đầu tự thỏa, nên chỉ còn: `len(people) < 2` → raise (bắt luôn chế độ `--video`), rồi liệt kê cặp thiếu. Ít code hơn, và message chỉ đúng cặp cần quay bù.

**`np.savez` nhận file handle, không nhận path.** `np.savez` append `.npz` vào path chưa có đuôi đó, làm file staging thành `<key>.npz.tmp.npz` và `replace()` trượt. Xem `docs/GOTCHAS.md`.

**Đường lùi cho checkpoint cũ là bắt buộc, không phải tiện tay.** `models/gesture_lstm.pt` đang track trong git được train bằng pooling cũ. Load nó bằng pooling mới sẽ dự đoán sai **âm thầm**. Nên khóa `pooling` thiếu → `legacy_last_step`.

**`validate_clips` chạy hai lần.** Lần đầu trước khi extract (fail nhanh, chưa tốn MediaPipe). Lần hai sau khi drop clip lỗi, vì một clip hỏng có thể làm một nhãn mất hẳn một người và phá điều kiện LOSO.

## Verify

```powershell
python -m pytest                                    # 24 passed
vslr-train --data-dir <tree> --loso --epochs 2 --augment 2 --model-dir <tmp> --cache-dir <tmp>
vslr-train --data-dir <tree>      --epochs 2 --augment 2 --model-dir <tmp> --cache-dir <tmp>
```

Đã chạy thật trên cây tạm 2 người × 3 nhãn × 2 clip (hardlink từ `dataset/raw/`):

- LOSO 2 fold, mỗi fold train 6 / test 6 clip; `loso_report.json` có 12 entry `extraction[]`; **chỉ** `loso_report.json` được ghi.
- Chạy lại ở chế độ ship: cả 12 clip báo `(cache)` với `hand_frame_ratio` **giống hệt** lần chạy MediaPipe → cache đúng và stats không mất.
- `metrics.json` history chỉ có `epoch`/`train_loss_smoothed`/`train_accuracy` — không khóa val nào, vì không có holdout.
- `--video --loso` → `error: Leave-one-signer-out needs clips from at least 2 different people, found ['unknown']…`
- `load_checkpoint('models/gesture_lstm.pt')` (checkpoint cũ) hiện **raise** vì feature v1 ≠ v2; opt-in legacy mới load `legacy_last_step` kèm cảnh báo. Checkpoint mới → `fwd_last_bwd_first`, 0 cảnh báo.
- `--num-workers 2` trên Windows: exit 0, LOSO chạy hết 2 fold.
- Một clip không mở được (file rác `.mov`): `SKIP` rồi `1 of 12 clips failed extraction (8.3% > 5%)`, exit 1, **`--model-dir` rỗng** — không ship gì.
- `--epochs 0` → `argument --epochs: must be >= 1, got 0`; `--augment -1` → `must be >= 0, got -1`.
- `loso_report.json` và `metrics.json` cùng `data_fingerprint` `66149052a58a…`, cùng `seed`/`epochs`/`features_version`.

Cây tạm và file tạm đã xóa. `models/` **không** bị ghi trong quá trình verify (mtime vẫn 2026-08-21_16:57).

**Chưa verify end-to-end:** nhánh "một nhãn mất hết clip nhưng vẫn dưới ngưỡng 5%" — cần ≥ 80 clip mới xuống dưới ngưỡng, quá tốn cho một smoke test. Nhánh này được phủ bằng unit test `test_missing_labels_after_drop_catches_a_label_that_lost_every_clip` cộng với đọc lại 4 dòng wiring trong `main()`, không phải bằng một lần chạy thật.

## Watch-outs

- **`FEATURES_VERSION` phải tăng tay.** Sửa `normalize_landmarks`, trim/margin, stride, hay ngưỡng confidence mà quên tăng nó → cache trả landmark cũ và mọi số phía sau mô tả extractor cũ, không có gì báo. Checkpoint lệch version bị `load_checkpoint` chặn mặc định.
- **`--num-workers` mặc định 0.** Augment tốn ~0,62 ms/mẫu single-thread → ~40 s/epoch ở 540 clip × 120, tức ~27 phút cho `--epochs 40` và ~2,2 h cho LOSO 5 fold (ngoại suy từ benchmark xen kẽ 250×11 vòng, chưa chạy thật ở 540 clip). Trên run thật nhớ đặt `--num-workers 4` trở lên. Seed theo vị trí nên số worker **không** đổi mẫu sinh ra.
- **`dataset/processed/dataset_3gestures.npz` không còn được sinh ra.** File 136 MB cũ vẫn nằm trên đĩa (ignored), xóa được.
- **Checkpoint hiện tại trong `models/` chưa được train lại.** Vẫn là model 1 epoch, pooling cũ, landmark v1; camera chặn mặc định vì code sinh landmark v2. Train lại cần quyết định của chủ dự án vì nó ghi đè artifact đang track.
- **`--epochs 40` là con số chọn tay, chưa tune.** Muốn tune tử tế cần inner split lồng trong từng fold.
- **Hai file report chỉ ghép được khi `training_signature` khớp.** Hash bao phủ data fingerprint, thứ tự nhãn, epochs, augment, batch size, learning rate, seed, pooling và feature version. `--loso` cố tình không xóa checkpoint cũ nên report mới vẫn có thể nằm cạnh artifact cũ; ship in `PAIR OK`/`PAIR MISMATCH` để nói rõ.
- **`pytest` xanh không thay cho một lần chạy CLI.** Tầng điều phối trong `main()` không có test; một lần đổi tên khóa history pass 20/20 unit test rồi chết giữa run thật. Xem `docs/GOTCHAS.md`.
- **`--loso` chạy N lần train.** Cache landmark chỉ tiết kiệm phần extract, không tiết kiệm phần train.
