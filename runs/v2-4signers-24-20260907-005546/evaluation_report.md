# Tóm tắt Đánh giá Thử nghiệm LOSO 4-Fold

- **Thời gian đánh giá**: 2026-09-06T19:14:28.232891+00:00
- **Tổng số clip test**: 576 clips (4 người ký x 24 cử chỉ x 6 clips)
- **Số clip đoán đúng**: 524 / 576
- **Pooled Accuracy**: 90.97%
- **Macro Mean Accuracy**: 90.97%
- **Trạng thái External Review**: PENDING

## Kết quả từng Fold

| Fold (Người ký kiểm thử) | Số clip test | Số clip đúng | Accuracy | Val Loss (CE) |
|---|:---:|:---:|:---:|:---:|
| Fold P01 | 144 | 140 | 97.22% | 0.1274 |
| Fold P02 | 144 | 134 | 93.06% | 0.3197 |
| Fold P03 | 144 | 138 | 95.83% | 0.2642 |
| Fold P04 | 144 | 112 | 77.78% | 0.9171 |

## Độ chính xác từng Cử chỉ (Xếp từ thấp đến cao, 24 Cử chỉ)

| Cử chỉ | Đúng / Tổng | Tỉ lệ (%) |
|---|:---:|:---:|
| Bạn tên gì | 15 / 24 | 62.5% |
| Sao thế | 17 / 24 | 70.8% |
| Cảm ơn | 19 / 24 | 79.2% |
| Tôi khỏe | 19 / 24 | 79.2% |
| Bạn có cần giúp đỡ không | 20 / 24 | 83.3% |
| Bạn quê ở đâu | 20 / 24 | 83.3% |
| Bạn đang làm gì | 20 / 24 | 83.3% |
| Xin lỗi | 20 / 24 | 83.3% |
| Đi đâu | 21 / 24 | 87.5% |
| Được không | 21 / 24 | 87.5% |
| Gọi xe cứu thương | 23 / 24 | 95.8% |
| Rất vui được gặp bạn | 23 / 24 | 95.8% |
| Tôi bình thường | 23 / 24 | 95.8% |
| Tạm biệt | 23 / 24 | 95.8% |
| Bạn có vấn đề gì không | 24 / 24 | 100.0% |
| Chuyện gì | 24 / 24 | 100.0% |
| Hôm nay bạn khỏe không | 24 / 24 | 100.0% |
| Lâu rồi không gặp | 24 / 24 | 100.0% |
| Mấy tuổi | 24 / 24 | 100.0% |
| Như thế nào | 24 / 24 | 100.0% |
| Siêu thị | 24 / 24 | 100.0% |
| Tôi không khỏe | 24 / 24 | 100.0% |
| Về nhà cẩn thận | 24 / 24 | 100.0% |
| Xin chào | 24 / 24 | 100.0% |

## Toàn bộ 20 Cặp Nhầm Lẫn (Tổng cộng 52 clips sai)

| STT | Cử chỉ thực tế (True Label) | Dự đoán nhầm sang (Predicted) | Số clips |
|:---:|---|---|:---:|
|  1 | Sao thế | Được không | 7 |
|  2 | Bạn tên gì | Bạn có vấn đề gì không | 5 |
|  3 | Tôi khỏe | Lâu rồi không gặp | 5 |
|  4 | Bạn có cần giúp đỡ không | Bạn có vấn đề gì không | 4 |
|  5 | Bạn tên gì | Bạn đang làm gì | 4 |
|  6 | Xin lỗi | Gọi xe cứu thương | 4 |
|  7 | Bạn đang làm gì | Được không | 3 |
|  8 | Cảm ơn | Như thế nào | 3 |
|  9 | Được không | Tạm biệt | 3 |
| 10 | Bạn quê ở đâu | Được không | 2 |
| 11 | Cảm ơn | Siêu thị | 2 |
| 12 | Đi đâu | Được không | 2 |
| 13 | Bạn quê ở đâu | Tôi khỏe | 1 |
| 14 | Bạn quê ở đâu | Đi đâu | 1 |
| 15 | Bạn đang làm gì | Bạn có vấn đề gì không | 1 |
| 16 | Gọi xe cứu thương | Mấy tuổi | 1 |
| 17 | Rất vui được gặp bạn | Siêu thị | 1 |
| 18 | Tôi bình thường | Sao thế | 1 |
| 19 | Tạm biệt | Đi đâu | 1 |
| 20 | Đi đâu | Tạm biệt | 1 |
| **Tổng** | **20 hướng nhầm lẫn** | — | **52** |