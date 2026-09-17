# Spec — Tiền đăng ký Nghiệm thu Realtime, OOD & Hiệu chuẩn Từ chối (V3 Realtime, OOD & Rejection Acceptance)

_Phiên bản: 1.0. Ngày: 2026-09-16. Trạng thái: Tiền đăng ký Đóng băng (Pre-registered & Frozen)._

---

## 1. Cơ sở Kiểm định & Khóa Bất biến (Provenance & Frozen Baseline)

1. **Đóng Đánh giá Ngoài Offline P05**:
   - P05 external evaluation đã hoàn thành và niêm phong bất biến tại `evaluation/p05-v3-final-20260916/`.
   - Kết quả ghi nhận: Raw Top-1 48/48 (100.00%), Top-3 48/48 (100.00%).
   - **`P05_CONSUMED = true`**: Tập P05 đã bị tiêu thụ hoàn toàn như một bằng chứng độc lập, vĩnh viễn không được coi là "unseen" cho bất kỳ vòng lựa chọn mô hình, tuning siêu tham số hoặc hiệu chuẩn ngưỡng nào trong tương lai.
2. **Khóa Bất biến Trọng số Mô hình (Model Invariant)**:
   - Checkpoint Candidate V3: `runs/v3-ship-4signers-24-8clips-13ep-20260916-020145/gesture_lstm.pt`.
   - SHA-256 yêu cầu: `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`.
   - Tuyệt đối không huấn luyện lại (NO retrain), không fine-tune, không tối ưu hóa trọng số dựa trên dữ liệu realtime hoặc calibration.
   - Trọng số sản xuất `models/gesture_lstm.pt` (SHA-256 `5e202eb9108c...`) giữ nguyên tuyệt đối, không ghi đè tự động.

---

## 2. Danh mục Thử nghiệm Tiền đăng ký (Pre-registered Test Categories A–L)

Trước khi tiến hành thử nghiệm camera realtime, danh mục 12 kịch bản kiểm thử sau đây được đóng băng tiền đăng ký (pre-registered). Nghiêm cấm việc quan sát webcam rồi mới điều chỉnh tiêu chí.

| Mã | Tên Kịch bản | Mô tả chi tiết | Tiêu chuẩn Đạt (Passing Criteria) |
|---|---|---|---|
| **A** | Known-class positive gestures | 24 cử chỉ trong từ điển, người ký thực hiện hoàn chỉnh, rõ ràng. | Model nhận diện đúng nhãn, vượt qua ngưỡng từ chối. |
| **B** | Neutral / rest pose | Người ngồi yên trước camera, hai tay buông thõng hoặc để trên bàn, không cử động. | Model từ chối 100% (không kích hoạt nhận diện, không phát sinh từ). |
| **C** | Random hand movement | Người vẫy tay tự do, gãi đầu, vuốt tóc, chỉ trỏ, gõ bàn phím, cầm cốc nước. | Model từ chối 100% (không kích hoạt cử chỉ VSL, không phát âm TTS). |
| **D** | Partial gesture | Cử chỉ bắt đầu nhưng dừng giữa chừng (dưới 50% quỹ đạo) rồi hạ tay. | Bị từ chối bởi bộ lọc tin cậy/thời lượng, không kích hoạt từ sai. |
| **E** | Transition between gestures | Động tác hạ tay/đổi tay giữa hai cử chỉ liên tiếp trong câu. | Không nhận diện frame chuyển tiếp thành cử chỉ thứ 3; không kích hoạt TTS giữa chừng. |
| **F** | Person entering/leaving frame | Người bước vào khung hình hoặc đứng dậy rời đi trong lúc camera đang chạy. | Không sinh false positive khi cơ thể cắt ngang biên khung hình. |
| **G** | One hand missing | Cố tình chỉ dùng 1 tay thực hiện cử chỉ vốn yêu cầu 2 tay (hoặc 1 tay bị giấu). | Bị từ chối do thiếu presence mask hoặc confidence tụt sâu dưới ngưỡng. |
| **H** | Hands occluded | Hai bàn tay đan chéo/che khuất lẫn nhau, hoặc cầm đồ vật che khuất lòng bàn tay. | MediaPipe báo thiếu landmark hoặc reject policy từ chối an toàn. |
| **I** | Face/body partially outside | Người ngồi lệch góc, camera chỉ thấy một phần ngực/mặt hoặc mất hông. | Pipeline fallback an toàn sang presence, không crash, từ chối nếu không đủ landmark. |
| **J** | Different distance | Thử nghiệm ở 3 cự ly: Gần (0.5m), Tiêu chuẩn (1.2m), Xa (2.5m). | Nhận diện ổn định ở cự ly tiêu chuẩn; từ chối an toàn khi quá xa/quá gần. |
| **K** | Different lighting | Thử nghiệm 3 điều kiện: Ngược sáng (backlight), Ánh sáng yếu (<50 lux), Ánh sáng gắt lệch bên. | Không crash, duy trì tỷ lệ từ chối trên OOD, báo cảnh báo chất lượng nếu cần. |
| **L** | Different background | Hậu cảnh phức tạp: văn phòng đông người đi lại phía sau, rèm cửa bay, đồ đạc lộn xộn. | Holistic loại bỏ chuyển động nền; không kích hoạt cử chỉ từ người thứ hai phía sau. |

---

## 3. Thu thập & Phân tách Dữ liệu Hiệu chuẩn (Calibration Dataset)

- **Quy tắc Vàng**: Tuyệt đối **KHÔNG** sử dụng P05 để hiệu chuẩn chính sách từ chối.
- Tập hiệu chuẩn (Calibration Split) phải được ghi hình và lưu trữ tách biệt hoàn toàn với cả tập huấn luyện (P01–P04) và tập kiểm thử P05.
- Thư mục chuẩn: `dataset/calibration_v3/` (hoặc tập tương đương mang định danh provenance rõ ràng).
- **Tuyệt đối không gọi tập hiệu chuẩn là TEST**.
- **Cấu trúc Dữ liệu Hiệu chuẩn**:
  1. **Positive Calibration**:
     - Các cử chỉ chuẩn trong 24 nhãn từ điển.
     - Lặp lại nhiều lần trên $\ge 2$ người ký độc lập.
  2. **Negative / OOD Calibration**:
     - `idle_stationary`: Ngồi yên, tư thế nghỉ, không di chuyển tay.
     - `oov_motion`: Cử chỉ ngẫu nhiên ngoài từ điển, động tác sinh hoạt hàng ngày.
     - `transitions`: Các đoạn chuyển tiếp tay giữa các cử chỉ.
     - `partial_gestures`: Cử chỉ làm dở dang.
     - `occlusions_no_person`: Khung hình trống không có người, hoặc đưa đồ vật che camera.
- Mỗi mẫu hiệu chuẩn phải được băm SHA-256 và lập chỉ mục trong `CALIBRATION_MANIFEST.json` với trường `split = "calibration"`.

---

## 4. Chính sách Từ chối Đa chiều (Multi-factor Rejection Policy)

Chính sách từ chối (`RejectPolicy`) được thiết kế và fit tự động **duy nhất trên tập Calibration**, đóng băng trước khi bước vào nghiệm thu. Không chỉ dùng độ tin cậy Top-1 đơn thuần, chính sách phải thẩm định tối thiểu 4 chiều:

1. **Ngưỡng Độ tin cậy Top-1 ($\tau_{\text{conf}}$)**:
   - Softmax score $P(\hat{y} \mid X) \ge \tau_{\text{conf}}$.
   - Ngưỡng $\tau_{\text{conf}}$ được tính toán phân tách giữa phân phối positive và negative của tập calibration:
     $$\tau_{\text{conf}} = \max(\text{negative\_max} + \epsilon, \frac{\text{negative\_max} + \text{positive\_min}}{2})$$
2. **Khoảng cách Top-1 và Top-2 ($\tau_{\text{margin}}$)**:
   - Margin $P(\hat{y}_1) - P(\hat{y}_2) \ge \tau_{\text{margin}}$ để loại bỏ các trường hợp mô hình phân vân giữa 2 cử chỉ tương đồng.
3. **Độ ổn định Thời gian (Temporal Dwell Duration)**:
   - Cử chỉ phải duy trì active state tối thiểu $T_{\text{min}} \ge 0.35$ giây liên tục.
   - Các chuyển động tay chớp nhoáng dưới 0.35 giây tự động bị hủy (blip discard) trước khi vào mô hình.
4. **Chất lượng Quan sát Bàn tay (Hand Presence Quality)**:
   - Tỷ lệ khung hình nhìn thấy tay có ý nghĩa $\ge 50\%$ tổng số frame của phân đoạn.

---

## 5. Cơ chế An toàn Thời gian thực & Ngăn chặn Kích hoạt Sai (Realtime Safety Behavior)

Nhằm đảm bảo an toàn tuyệt đối khi đưa ra loa TTS thời gian thực:
1. **Cấm Kích hoạt Từ Frame Đơn lẻ**:
   - Tuyệt đối không gọi TTS trực tiếp từ 1 frame hoặc 1 cửa sổ trượt ngắn đơn độc.
2. **Xác nhận Dwell / Cửa sổ Liên tiếp**:
   - Phân đoạn cử chỉ phải kết thúc tự nhiên qua cơ chế hạ tay (`word_gap` $\ge 0.45$s) và đạt đủ thời lượng tối thiểu ($0.35$s $\le \Delta t \le 5.0$s).
3. **Chống Phát âm Lặp lại (Duplicate Phrase Suppression Cooldown)**:
   - Nếu cử chỉ mới nhận diện trùng với cử chỉ vừa được chấp nhận trong khoảng thời gian cooldown ($T_{\text{cooldown}} = 1.5$s), từ này bị chặn (suppressed), không đưa vào câu phát âm.
4. **Khởi tạo lại khi Bị Từ chối (Rejection Reset)**:
   - Khi một phân đoạn bị từ chối (REJECT), trạng thái theo dõi ứng viên liên tiếp lập tức bị xóa trắng (`last_accepted_label = None`), ngăn ngừa việc ghép nối sai lệch.
5. **Bộ đệm Khoảng lặng Câu (Sentence Idle Gap)**:
   - TTS chỉ được kích hoạt phát âm sau khi người dùng hạ tay hoàn toàn và giữ yên tĩnh tối thiểu $T_{\text{sentence}} = 2.2$ giây.

---

## 6. Bộ Chỉ số Nghiệm thu Tách biệt (Acceptance Metrics Separation)

Kết quả nghiệm thu trên tập Acceptance độc lập phải được báo cáo theo 3 khối riêng biệt, **tuyệt đối không gộp chung thành một số accuracy bình quân duy nhất**:

### 6.1. Nhóm Cử chỉ Đúng trong Từ điển (Known-Gesture Metrics)
- **Raw Top-1 Accuracy**: Tỷ lệ đoán đúng không qua bộ lọc từ chối ($\ge 90.0\%$).
- **Accepted Accuracy**: Tỷ lệ đoán đúng trong số các mẫu vượt qua bộ lọc ($\ge 95.0\%$).
- **Coverage**: Tỷ lệ mẫu được chấp nhận ($\ge 85.0\%$).
- **False Reject Rate (FRR)**: Tỷ lệ mẫu đúng bị từ chối oan ($\le 15.0\%$).

### 6.2. Nhóm Ngoài Từ điển / Âm tính (Negative / OOD Metrics)
- **Rejection Rate**: Tỷ lệ từ chối thành công trên các chuyển động OOD/nhiễu ($\ge 95.0\%$).
- **False Accept Rate (FAR)**: Tỷ lệ nhận diện nhầm cử chỉ OOD thành từ VSL ($\le 5.0\%$).

### 6.3. Nhóm Chỉ số Ổn định Thời gian (Temporal & Trigger Metrics)
- **False Trigger Count**: Số lần TTS phát âm sai khi người ký ở trạng thái nghỉ hoặc chuyển động ngẫu nhiên (Yêu cầu: **0 lần**).
- **Duplicate Trigger Count**: Số lần lặp từ ngoài ý muốn trong khoảng cooldown.
- **Transition Trigger Count**: Số lần frame chuyển tiếp bị nhận nhầm thành từ VSL (Yêu cầu: **0 lần**).
- **Median Recognition Latency**: Độ trễ trung vị từ khi kết thúc cử chỉ đến khi phân loại hoàn tất ($\le 600$ ms).
- **P95 Recognition Latency**: Độ trễ phân vị 95 ($\le 1000$ ms).

---

## 7. Yêu cầu Đa Người ký Nghiệm thu (Multi-Signer Requirement)

- Do P05 chỉ chứng minh khả năng tổng quát hóa trên duy nhất một người ký bên ngoài, đợt nghiệm thu Realtime Acceptance phải có sự tham gia của **tối thiểu 2 người ký độc lập mới** (chưa từng tham gia huấn luyện P01–P04 và chưa phải P05).
- Các người ký được mã hóa ẩn danh: `R01`, `R02`, `R03`...
- Tuyệt đối không bổ sung các mẫu thu thập này vào tập huấn luyện của mô hình.

---

## 8. Chốt Kiểm định Độc lập 10 Tiêu chí (Independent Review Checklist)

Một reviewer độc lập (fresh subagent/auditor) phải xác minh đủ 10 tiêu chí trước khi đề xuất phát hành:
1. P05 tuyệt đối không bị tái sử dụng cho calibration hoặc model selection.
2. Checkpoint Candidate SHA-256 giữ nguyên tuyệt đối: `e7a85bf25eee163ddcfd37a98b1b4b6cc0ec06a4b524d0daced3fe67a89d1fa4`.
3. Tập dữ liệu Calibration và Acceptance hoàn toàn tách rời (disjoint), không trùng lặp video hoặc hash.
4. Ngưỡng và chính sách từ chối đã được đóng băng TRƯỚC khi chạy tập Acceptance.
5. Toàn bộ 12 danh mục thử nghiệm A–L (đặc biệt là Negative/OOD/Idle) đều có dữ liệu thử nghiệm thực tế.
6. Chính sách kích hoạt thời gian thực (temporal confirmation, cooldown, reset) được kiểm thử trực tiếp bằng mã lệnh.
7. TTS tuyệt đối không thể phát âm từ một frame đơn lẻ hoặc dao động chập chờn.
8. Bảng số liệu thô (raw predictions, per-label, per-category) được lưu trữ đầy đủ kèm mã băm SHA-256.
9. Không có bất kỳ dòng lệnh nào thực hiện huấn luyện lại mô hình (zero retraining).
10. Checkpoint sản xuất `models/gesture_lstm.pt` giữ nguyên nguyên trạng SHA-256 `5e202eb9108c...`.

---

## 9. Quy tắc Ra Quyết định Phát hành (Promotion Decision Rule)

- Chỉ khi toàn bộ dữ liệu thực nghiệm trên tập Calibration và Acceptance được thu thập đầy đủ, vượt qua toàn bộ các ngưỡng tại Mục 6 và 10 tiêu chí tại Mục 8:
  $$\implies \textbf{READY\_FOR\_PRODUCTION\_PROMOTION}$$
- Khi dữ liệu thực nghiệm chưa được thu thập đầy đủ hoặc bất kỳ tiêu chí nào chưa đạt:
  $$\implies \textbf{HOLD}$$
- Việc sao chép candidate vào `models/gesture_lstm.pt` là quyền quyết định tối cao của Root, không được tự động thực hiện trong bất kỳ hoàn cảnh nào.
