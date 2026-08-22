# Specs (thiết kế TRƯỚC khi code)

Mỗi feature → 1 file `<feature>.md` viết TRƯỚC khi code:

- **Goal**: feature làm gì (1-2 câu).
- **Requirements**: checklist `[ ]` các điều kiện phải đạt.
- **Constraints**: giới hạn kỹ thuật / bảo mật / dữ liệu.
- **Decisions**: chọn gì, vì sao.
- **Out of scope**: vòng này KHÔNG làm gì.

Spec xong, trước khi viết dòng code đầu tiên: chạy `spec-critic` nếu feature đụng nhiều hơn một file, hoặc đụng data model / pipeline train / cách báo số liệu.
