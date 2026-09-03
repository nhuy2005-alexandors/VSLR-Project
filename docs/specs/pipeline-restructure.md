# Spec — Tái cấu trúc pipeline train

> Contract directory hiện hành dùng `dataset/recording_plan.json` + `dataset/recordings_v1` và
> feature v3; xem `docs/specs/pipeline-hardening.md` cho các invariant mới.

_Viết: 2026-08-22. Rev 2 sau `spec-critic` (verdict rev 1: BLOCKED — 5 blocker / 9 should-fix). **Đã implement**; as-built ở `docs/technical_specs/pipeline-restructure.md` và `video-ingest-readiness.md`._

## Goal

Pipeline chạy được ở 27 nhãn × 5 người × 4 clip (540 clip) và cho ra **một con số accuracy báo cáo được** (leave-one-signer-out), thay vì quy mô 3 nhãn / 27 clip và một checkpoint train đúng 1 epoch.

Không viết lại `src/`. `vsl3/features.py` và `vsl3/model.py` là lõi đã có test; spec này đổi tầng điều phối trong `prepare_train.py` và thêm ID người vào data model.

## Pipeline hiện tại — lỗi từng tầng

| Tầng | Nguồn | Lỗi |
|---|---|---|
| `parse_video_specs` | `prepare_train.py:33-34` | `raise` nếu số nhãn `!= 3` → chặn 27 nhãn |
| `extract_video` | `prepare_train.py:151` | MediaPipe chạy lại toàn bộ mỗi lần train |
| `build_dataset` | `prepare_train.py:57`, `:61` | `enumerate` theo **clip** → `source_ids` là chỉ số clip |
| `build_dataset` | `prepare_train.py:44-72` | materialize 121 bản/clip; peak ~9,4 GB ở 540 clip (list + `np.stack` + `astype` tại `:68`) |
| `savez_compressed` | `prepare_train.py:163-172` | không code nào đọc file này (grep: chỉ có writer) |
| `source_holdout_split` | `prepare_train.py:75-90` | giữ ra một **clip** mỗi nhãn, không phải một **người** |
| early stop | `prepare_train.py:221-236` | val 363 mẫu = clip giữ ra + 120 bản augment của chính nó; clip khác **cùng buổi quay** nằm ở train |
| `best_state` | `prepare_train.py:238-241` | load ra chỉ để tính metric rồi bỏ |
| refit | `prepare_train.py:243-260` | model mới từ random init, train `max(1, best_epoch)` = 1 epoch |
| `metrics.json` | `prepare_train.py:271-294` | số đo trên `best_state`, artifact lưu là `final_model` |

**Không phải một nguyên nhân duy nhất** — rev 1 nói vậy và sai. Có hai nguyên nhân độc lập:

1. **Split sai đơn vị** → val không đo được người mới. Sửa bằng split theo người.
2. **Cơ chế `best_epoch` → refit là control flow vô điều kiện** → artifact ship ra không phải model đã đo. Sửa bằng cách xóa hẳn cơ chế đó.

Nguyên nhân 2 **không** tự khỏi khi sửa nguyên nhân 1. Bằng chứng trong `models/metrics.json`: `history` 16 epoch có `holdout_val_loss` thấp nhất ở epoch 1 (0,016007), không epoch nào phá được, `holdout_val_accuracy` = 1.0 suốt, `train_accuracy` = 1.000 từ epoch 3. Val khó hơn không đảm bảo `best_epoch > 1` — khi model không còn cải thiện trên val thì epoch 1 thắng bằng nhiễu.

## Hai chế độ, một lệnh

```bash
vslr-train --data-dir dataset/raw --loso        # ĐO: N fold, ghi models/loso_report.json, KHÔNG ghi checkpoint
vslr-train --data-dir dataset/raw --epochs 40   # SẢN XUẤT: train mọi clip, ship weight cuối
```

| | `--loso` (đo) | mặc định (sản xuất) |
|---|---|---|
| Dữ liệu train | N−1 người mỗi fold | **toàn bộ** người |
| Validation | 1 người giữ ra, **không augment** | **không có** |
| Early stopping | **không có** | **không có** |
| Chọn epoch | không chọn — `--epochs` cố định | không chọn — `--epochs` cố định |
| Artifact ghi ra | `models/loso_report.json` | `gesture_lstm.pt`, `labels.json`, `metrics.json` |

Chế độ sản xuất **không có validation nên không thể chọn sai epoch**. Xóa hẳn `best_state`, `best_epoch`, `final_fit_epochs` và khối refit — xóa cơ chế đã sinh ra bẫy #3 trong `docs/GOTCHAS.md`, không phải vá nó.

**Không early stopping ở chế độ `--loso` nữa** (rev 2 lúc đầu định lấy `--epochs` từ median epoch tốt nhất của LOSO — đó là **chọn tham số trên tập test**, vì người giữ ra trong LOSO chính là tập test). `--epochs` là hằng chọn trước, mặc định 40, ghi vào report; muốn tune nó tử tế thì cần inner split lồng bên trong mỗi fold — ngoài scope. Curve val theo epoch vẫn được ghi vào report để xem, nhưng **không** dùng để chọn gì.

## Requirements

Phát hiện dữ liệu:

- [ ] `--data-dir DIR` quét `DIR/<person>/<gesture>/*.mov|*.mp4`. Người = thư mục cấp 1, nhãn = thư mục cấp 2.
- [ ] **Tên thư mục cử chỉ là tiếng Việt có dấu**, dùng nguyên văn làm nhãn: `dataset/raw/P1/Cảm ơn/…`. Không slug, không file mapping. Repo đã track path UTF-8 (`dataset/raw/cam_on/Cảm ơn 1.mov`) nên UTF-8 trong tên thư mục là đường đã đi được.
- [ ] Data model mỗi clip mang `(label, person, path)`. Không suy người từ số thứ tự file.
- [ ] `--video LABEL=PATH` giữ nguyên, nhưng **là chế độ một-người**: mọi clip nhận `person="unknown"`, `--loso` bị từ chối, `metrics.json` ghi `"signer_split": false`.
- [ ] 27 clip hiện có **không đổi chỗ, không gán ID người**. Mapping `1-3/4-6/7-9` chưa ai xác nhận; đoán ID người tạo ra một con số LOSO giả. Chúng chạy được ở chế độ một-người, và không bao giờ được dùng để báo generalization.
- [ ] Ràng buộc thay `>= 2 clip mỗi nhãn` bằng **`>= 2 người mỗi nhãn`** ở chế độ LOSO. 2 clip từ 1 người thỏa ràng buộc cũ mà vẫn không split được.
- [ ] Nhãn thiếu ở một người → `raise`, in đủ danh sách cặp `(nhãn, người)` bị thiếu. Không âm thầm bỏ fold hay bỏ nhãn.

Cache landmark:

- [ ] Khóa cache `(source_sha256, path, mtime_ns, size, FEATURE_DIM, SEQUENCE_LENGTH, FEATURES_VERSION)`. Content hash ngăn copy/restore giữ timestamp trả landmark của video cũ. `FEATURES_VERSION` là hằng trong `features.py`, tăng tay khi sửa `normalize_landmarks` / trim / stride / ngưỡng confidence.
- [ ] Cache lưu **kèm** `sampled_frames`, `trimmed_frames`, `hand_frame_ratio`. `metrics.json` phải luôn có `extraction[]` đủ mọi clip kể cả khi cache hit — quy tắc số liệu trong `CLAUDE.md` và bước nghiệm thu trong `docs/specs/dataset-recording.md` đều đọc khóa này.
- [ ] Ghi cache **từng clip một** ngay khi xong → chạy lại là resume, không mất công đã làm.
- [ ] Clip lỗi (`extract_video` raise: dưới 8 frame, hoặc `hand_frame_ratio < 0.10`) → ghi nhận rồi **đi tiếp**, không chết cả run 540 clip. Cuối run in danh sách clip lỗi; `raise` nếu quá 5% clip lỗi.

Dataset và augment:

- [ ] Augment trong `Dataset.__getitem__`, không materialize. Bỏ `--dataset-output` và file npz.
- [ ] `__len__ = số_clip × (--augment + 1)`. Index `i` → clip `i // (aug+1)`; `i % (aug+1) == 0` trả về sequence **gốc chưa augment**. Giữ nguyên ngữ nghĩa 121 mẫu/clip như hiện tại, nên `--epochs 70` và patience 15 không đổi nghĩa.
- [ ] RNG augment seed theo `(--seed, clip_idx, aug_idx)`, không theo epoch → tái lập được và an toàn với DataLoader worker.
- [ ] Phía val / LOSO **không augment**. Accuracy báo cáo tính trên clip thật: mỗi fold 27 nhãn × 4 clip = 108 clip.

Báo cáo:

- [ ] `--loso` ghi `models/loso_report.json`: accuracy từng fold, mean, **fold tệ nhất**, curve val theo epoch (chỉ để xem, không dùng chọn), `extraction[]` đủ mọi clip, danh sách người mỗi fold, và câu ghi rõ `--epochs` được cố định trước chứ không tune trên fold. Không chạm `gesture_lstm.pt` / `labels.json` / `metrics.json`.
- [ ] `metrics.json` (chế độ sản xuất) thêm `"mode": "ship"`, `"signer_split": false`, `"epochs": N`, kèm câu ghi rõ: model này không có holdout, số để báo cáo nằm ở `loso_report.json`.

## Constraints

| Ràng buộc | Vì sao | Nguồn |
|---|---|---|
| Không đổi `FEATURE_DIM` 201 / `SEQUENCE_LENGTH` 60 | `sequence_length` do `prepare_train.py:262-268` ghi vào config, `realtime.py:86` là consumer duy nhất (`load_checkpoint` **không** đọc khóa này) | `prepare_train.py:262-268`, `realtime.py:86` |
| `augment_sequence(sequence, rng)` giữ chữ ký | test gọi trực tiếp | `tests/test_core.py:26` |
| Nhãn giữ nguyên tiếng Việt có dấu | vào checkpoint (`model.py:44-54`), ra ở `realtime.py:85`, đọc thành tiếng ở `realtime.py:144-148` | `models/labels.json` |
| Không mirroring | tay trái múa ngược là mẫu khác hẳn | `vsl3/features.py:98` |
| `--loso` không ghi `models/gesture_lstm.pt` | tách "số để báo cáo" khỏi "model để demo" | nguyên nhân gốc của bẫy #3 |
| `extraction[]` phải đủ dù cache hit | quy tắc số liệu của repo đọc khóa này | `CLAUDE.md`, `docs/GOTCHAS.md` |

## Decisions

**Chế độ sản xuất không có validation.** Không có val thì không có `best_epoch` để chọn sai. Đây là lời đáp cho blocker của critic: split theo người **không** tự sửa `best_epoch = 1`, vì `history` trong `metrics.json` cho thấy val_loss không bao giờ phá được epoch 1 sau khi model fit xong train ở epoch 3. `docs/GOTCHAS.md:33` nêu ba cách tránh; rev 1 chỉ lấy cách thứ ba — đúng cách duy nhất không đụng tới `best_epoch`.

**Split trước augment.** Split trên danh sách clip gốc → augment của người đang test không thể vào train **theo cấu trúc**, không chỉ theo test.

**Không đoán ID người cho 27 clip cũ.** Đoán sai tạo ra LOSO giả — tệ hơn không có số, vì nó trông như có số.

**Nhãn = tên thư mục tiếng Việt, không phải slug.** Rev 1 viết "nhãn = tên thư mục cử chỉ" trong khi thư mục là `cam_on` mà artifact chứa `Cảm ơn`. Làm đúng chữ đó thì TTS đọc "cam_on chuyen_gi".

**Augment tái lập được, không ngẫu nhiên mỗi epoch.** Seed theo `(seed, clip_idx, aug_idx)` cho đúng 121 mẫu cố định mỗi clip. Đổi lại: mất lợi ích "mỗi epoch thấy mẫu mới". Với báo cáo NCKH thì tái lập quan trọng hơn.

**Xóa cơ chế thay vì vá.** `best_state` + `best_epoch` + `final_fit_epochs` + refit ≈ 25 dòng bị xóa, không dòng nào thêm vào để thay.

## Cần làm kèm (ngoài file này)

- [ ] `docs/DECISIONS.md`: ADR-003 thay thế ADR-001 (split theo clip → theo người). Ghi rõ trade-off cũ ở `DECISIONS.md:13` (checkpoint refit trên mọi clip) nay vẫn được giữ, bằng cách khác: chế độ sản xuất train trên **toàn bộ** người nên không mất dữ liệu nào.
- [ ] `CLAUDE.md:49`: npz ghi 45 MB, thật là **136,2 MB**. Sửa hoặc bỏ dòng cùng lúc bỏ npz.
- [ ] `docs/specs/dataset-recording.md`: đánh dấu superseded, trỏ sang spec này.

## Test (TDD — viết đỏ trước)

Import ở `tests/test_core.py:8` là module-level: xóa `build_dataset` mà không sửa import làm **cả 7 test** chết vì collection error, không phải "5 passed".

| Test hiện có | Số phận |
|---|---|
| `test_build_dataset_tracks_source_video_and_class` (`:55`) | **chết** cùng `build_dataset` → thay bằng `test_dataset_len_and_first_index_is_unaugmented` |
| `test_source_holdout_split_never_leaks_a_video_between_train_and_val` (`:66`) | **viết lại** thành `test_signer_split_never_puts_a_person_in_both_sides` |
| `test_parse_video_specs_accepts_three_sources_per_three_labels` (`:40`) | vỡ khi `parse_video_specs` trả 3 field (unpack 2-tuple ở `:53`) → cập nhật, bỏ khẳng định về đúng 3 nhãn |

Test mới:

- [ ] `test_discover_clips_reads_person_and_vietnamese_label_from_path`
- [ ] `test_signer_split_never_puts_a_person_in_both_sides`
- [ ] `test_loso_raises_when_a_label_is_missing_for_one_person`
- [ ] `test_dataset_len_and_first_index_is_unaugmented`
- [ ] `test_augment_is_reproducible_for_same_seed_and_index`
- [ ] `test_cache_key_changes_when_features_version_changes`
- [ ] `test_video_mode_refuses_loso`

## Out of scope

- Thêm `time_stretch` vào `augment_sequence` — đổi phân phối augment nên phải đo lại, làm sau khi pipeline đứng.
- Sửa `realtime.py` (bug tách một cử chỉ thành hai từ).
- Face landmark / `N_POSE` > 25.
- Pretrain trên `dataset/legacy/` (4362 clip, 3315 nhãn) — spec riêng.
- Sentence-level label. Pipeline vẫn một clip một nhãn.

**Không còn out of scope:** `GestureLSTM.forward` lấy `out[:, -1, :]` (`model.py:41`) — nửa backward chỉ thấy một frame (đo trực tiếp: đổi 5/6 frame đầu, nửa đó không nhúc nhích). Phải sửa **trước** khi chạy `--loso`, nếu không con số LOSO đo trên một model biết là hỏng và phải đo lại lần hai cho cùng một mốc.

## Chưa quyết — cần chủ dự án

- **4 clip hay 6 clip mỗi người mỗi nhãn.** Spec này chọn 4 và đảo `dataset-recording.md:83`. Lý do cũ đã hết hiệu lực: nó lập luận 6 clip vì ở 3 nhãn mỗi fold chỉ test 3 clip nên accuracy nhảy bậc 33%. Ở 27 nhãn × 5 người mỗi fold test **108 clip**, sai một clip là 0,93%. Chênh lệch công quay: 540 clip so với 810 clip.
- **Danh sách 27 nhãn** là giả định lúc viết spec. V1 sau đó được chốt thành 30 nhãn theo ADR-013; thiết kế không đổi theo số lớp.
