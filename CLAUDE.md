# CLAUDE.md — VSLR 3 cử chỉ

Project NCKH: nhận ba cử chỉ tiếng Việt (`Cảm ơn`, `Chuyện gì`, `Xin chào`) bằng MediaPipe + BiLSTM, ghép thành câu và đọc bằng TTS.

## Stack

Python >= 3.11, setuptools + pytest, PyTorch, MediaPipe 0.10.21, OpenCV 4.11.

| Việc | Lệnh |
|---|---|
| Cài | `python -m pip install -e ".[tts]"` |
| Test | `python -m pytest` (`pythonpath=src`, `testpaths=tests` trong `pyproject.toml`) |
| Build | không có build step — package thuần Python, cài editable là đủ |
| Nghiệm thu clip | `vslr-check --data-dir dataset/raw` — clip nào phải quay lại, còn thiếu cặp nào. Không train. Exit 1 nếu chưa đạt |
| Train | `vslr-train --data-dir dataset/raw` — train mọi clip, **không holdout, không kèm accuracy** |
| Đo accuracy | `vslr-train --data-dir dataset/raw --loso` — ghi `models/loso_report.json`, **không** ghi checkpoint |
| Train một người | `vslr-train --video "Nhãn=path.mov" ...` (≥ 2 clip mỗi nhãn; `--loso` bị từ chối vì không có ID người) |
| Camera | `vslr-camera --no-tts` (thêm TTS: bỏ cờ) |

In nhãn tiếng Việt ra console Windows: đặt `PYTHONIOENCODING=utf-8` trước, không thì `UnicodeEncodeError` (xem `docs/GOTCHAS.md`).

## Kỷ luật số liệu (quan trọng với project này)

- Không bao giờ báo accuracy mà không đối chiếu `extraction[].video` trong `models/metrics.json` với danh sách clip test — repo này đã một lần train trên chính clip test (`docs/GOTCHAS.md`).
- Docs assert số → cross-check với artifact thật. Lệch thì báo user, không im lặng.
- Không weaken test/assertion/ngưỡng để lấy màu xanh.

## Subagent Orchestration (khi model chính là Opus)

Opus = orchestrator, tự làm là mặc định. Spawn sub qua tool `Agent` với `subagent_type` chỉ khi: output sẽ làm ngập context rồi bị bỏ đi (scan nhiều file, log build/test dài, hunt nhiều vòng), hoặc cần người chấm ≠ người làm, hoặc việc song song thật sự độc lập.

Ba cổng bắt buộc: `spec-critic` (spec đụng > 1 file, hoặc đụng data model / pipeline train) → `debugger` (hai lần fix cùng một lỗi đã thất bại, hoặc error không tự chỉ ra nguyên nhân) → `reviewer` (diff nhiều file, logic dễ sai, hoặc bất kỳ claim "đã verify/pass" mà chính mình không xem chạy). Người làm không tự chấm việc mình. BLOCKER của critic là **câu hỏi**, Opus trả lời chứ critic không tự sửa.

Prompt cho sub phải self-contained (path, dòng, spec, kết quả mong đợi) — sub không thấy hội thoại. Đòi receipt (`file:line`, lệnh chính xác, khóa JSON) và đọc receipt đó trước khi tin.

## Spec & Checkpoint Workflow (REQUIRED)

- **Đầu session**: đọc `docs/CHECKPOINT.md`.
- **Trước feature**: đọc `docs/specs/<feature>.md`. Chưa có → bàn spec với user trước, không code mù.
- **Feature/bugfix mới (TDD)**: viết test fail TRƯỚC (red) → làm pass (green) → refactor.
- **Sau feature lớn + verify pass**: viết `docs/technical_specs/<feature>.md`.
- **Trước merge**: audit (`/code-review` hoặc `reviewer` sub) đọc diff thật.
- **Cuối task lớn**: cập nhật `docs/CHECKPOINT.md`.
- Quyết định kiến trúc → `docs/DECISIONS.md`. Bẫy mới → `docs/GOTCHAS.md`.
- Task UI mà chưa có `docs/design-principles.md` → hỏi taste của user trước, đừng đoán.

## Dữ liệu

| Path | Nội dung | Git |
|---|---|---|
| `dataset/labels.txt` | **nguồn sự thật duy nhất** cho tập nhãn; `vslr-check` đọc nó để biết còn thiếu nhãn nào (cây không nói được) | tracked |
| `dataset/raw/` | 27 clip train (1–9 mỗi nhãn); tên file **không** mã hóa người ký nên chỉ chạy được chế độ một-người | tracked |
| `dataset/processed/landmark_cache/` | cache landmark `.npz` mỗi clip, khóa theo `(path, mtime, FEATURES_VERSION)` — xóa được, tự sinh lại | ignored |
| `dataset/legacy/` | bộ từ điển VSL crawl từ bên thứ ba: `Videos/` 4362 `.mp4`, `Text/label_all.csv` 3315 nhãn (~1,5 clip/nhãn) | ignored |
| `models/` | `gesture_lstm.pt`, `labels.json`, `metrics.json` | tracked |
| `models/backups/` | snapshot lần train cũ (có 1 npz 106 MB) | untracked |

Data quality issue → sửa file gốc trên đĩa cho sạch + backup revert được, không chỉ vá in-memory trong script.
