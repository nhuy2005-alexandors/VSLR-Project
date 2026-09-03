# Specs (thiết kế TRƯỚC khi code)

Mỗi feature → 1 file `<feature>.md` viết TRƯỚC khi code:

- **Goal**: feature làm gì (1-2 câu).
- **Requirements**: checklist `[ ]` các điều kiện phải đạt.
- **Constraints**: giới hạn kỹ thuật / bảo mật / dữ liệu.
- **Decisions**: chọn gì, vì sao.
- **Out of scope**: vòng này KHÔNG làm gì.

Spec xong, trước khi viết dòng code đầu tiên: chạy `spec-critic` nếu feature đụng nhiều hơn một file, hoặc đụng data model / pipeline train / cách báo số liệu.

Trạng thái:

- `pipeline-restructure.md` — đã implement; contract hiện hành được harden trong `pipeline-hardening.md`.
- `dataset-recording.md` — chưa quay; manifest V1 đã chốt 30 nhãn, kế hoạch 4 người × 6 clip.
- `record-to-test.md` — lỗ hổng 1–3 đã implement; metric thật còn chờ raw sentence/negative data và checkpoint v3.
