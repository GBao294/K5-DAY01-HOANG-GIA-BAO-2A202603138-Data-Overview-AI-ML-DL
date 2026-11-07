# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`. Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS. Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output; họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| :---- | :---- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1\. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

&nbsp;

1. Chép record hạng 1: `class_id`\=… ; `class_name`\=… ; `rank`\=1 ; `score`\=… ; `taxonomy_name`\=…

&nbsp;&nbsp;{

&nbsp;&nbsp;&nbsp;&nbsp;"sample\_id": "traffic",

&nbsp;&nbsp;&nbsp;&nbsp;"coco\_image\_id": 210273,

&nbsp;&nbsp;&nbsp;&nbsp;"image\_width": 640,

&nbsp;&nbsp;&nbsp;&nbsp;"image\_height": 428,

&nbsp;&nbsp;&nbsp;&nbsp;"task": "image\_classification",

&nbsp;&nbsp;&nbsp;&nbsp;"taxonomy\_name": "ImageNet-1K",

&nbsp;&nbsp;&nbsp;&nbsp;"model\_file": "yolo11n-cls.pt",

&nbsp;&nbsp;&nbsp;&nbsp;"model\_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",

&nbsp;&nbsp;&nbsp;&nbsp;"ultralytics\_version": "8.4.145",

&nbsp;&nbsp;&nbsp;&nbsp;"rank": 1,

&nbsp;&nbsp;&nbsp;&nbsp;"class\_id": 468,

&nbsp;&nbsp;&nbsp;&nbsp;"class\_name": "cab",

&nbsp;&nbsp;&nbsp;&nbsp;"score": 0.510915

&nbsp;&nbsp;},

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này nói về cả bức ảnh. Vì Classification thực hiện phân loại hình ảnh dựa trên nội dung tổng thể, đưa ra lớp mà model cho là phù hợp nhất. Nó không xác định vị trí hay phân loại riêng từng vật thể xuất hiện trong ảnh.&nbsp;

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Danh sách tên lớp đi kèm với taxonomy/checkpoint của model đã được huấn luyện trước. Model không tự nghĩ ra tên lớp mới mà chỉ dự đoán dựa trên những lớp có sẵn.&nbsp;

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

Vì mỗi thông tin có mục đích khác nhau:

`class_id`: Mã định danh của lớp.

`class_name`: Tên lớp để con người dễ đọc và hiểu.

`taxonomy_name`: Cho biết hệ thống phân loại đang sử dụng.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Guideline cần quy định rõ tiêu chí chọn lớp chính, chẳng hạn dựa trên chủ thể nổi bật nhất, vật thể chiếm diện tích lớn nhất hoặc nội dung chính của bức ảnh. Người gán nhãn cần áp dụng thống nhất một quy tắc thay vì tự chọn theo cảm tính.&nbsp;

6. Vì sao không được lấy `score` của model làm ground truth?

Vì `score` chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đó chính xác. Ground truth phải được xác định dựa trên dữ liệu thực tế và guideline do con người xây dựng.&nbsp;

## 2\. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`\=… ; `score`\=… ; `bbox_xyxy`\=\[…, …, …, …\] ; `bbox_width`\=… ; `bbox_height`\=…

{

&nbsp;&nbsp;&nbsp;&nbsp;"sample\_id": "kitchen",

&nbsp;&nbsp;&nbsp;&nbsp;"coco\_image\_id": 397133,

&nbsp;&nbsp;&nbsp;&nbsp;"image\_width": 640,

&nbsp;&nbsp;&nbsp;&nbsp;"image\_height": 427,

&nbsp;&nbsp;&nbsp;&nbsp;"task": "object\_detection",

&nbsp;&nbsp;&nbsp;&nbsp;"taxonomy\_name": "COCO-80",

&nbsp;&nbsp;&nbsp;&nbsp;"model\_file": "yolo11n.pt",

&nbsp;&nbsp;&nbsp;&nbsp;"model\_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",

&nbsp;&nbsp;&nbsp;&nbsp;"ultralytics\_version": "8.4.145",

&nbsp;&nbsp;&nbsp;&nbsp;"score\_threshold": 0.35,

&nbsp;&nbsp;&nbsp;&nbsp;"class\_id": 0,

&nbsp;&nbsp;&nbsp;&nbsp;"class\_name": "person",

&nbsp;&nbsp;&nbsp;&nbsp;"score": 0.912624,

&nbsp;&nbsp;&nbsp;&nbsp;"coordinate\_unit": "pixel",

&nbsp;&nbsp;&nbsp;&nbsp;"bbox\_format": "xyxy",

&nbsp;&nbsp;&nbsp;&nbsp;"bbox\_xyxy": \[

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;385.33,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;69.24,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;498.92,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;348.92

&nbsp;&nbsp;&nbsp;&nbsp;\],

&nbsp;&nbsp;&nbsp;&nbsp;"bbox\_width": 113.58,

&nbsp;&nbsp;&nbsp;&nbsp;"bbox\_height": 279.68

&nbsp;&nbsp;},

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

Bounding box của đối tượng `person` được xác định bằng hai điểm:

Góc trên-trái: tọa độ (385.33, 69.24).

Góc dưới-phải: tọa độ (498.92, 348.92)

Như vậy, box có chiều rộng 113.58 pixel và chiều cao 279.68 pixel, xác định vị trí của người trong ảnh. .

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

threshold=0.20: 17 vật thể → \['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle'\]

threshold=0.35: 11 vật thể → \['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl'\]

threshold=0.60: 6 vật thể → \['person', 'bowl', 'bowl', 'oven', 'oven', 'person'\]

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

Reviewer cần kiểm tra thêm các bounding box có score thấp, xác định những dự đoán sai, box bị trùng lặp hoặc không phù hợp với vật thể thực tế. Sau đó, reviewer cần loại bỏ hoặc yêu cầu chỉnh sửa những nhãn không đạt yêu cầu.&nbsp;

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

Bounding box cần bao quanh toàn bộ phần vật thể nhìn thấy, bám sát đường viền của vật thể và hạn chế lấy thêm vùng nền xung quanh. Mỗi vật thể riêng biệt cần có một bounding box riêng.&nbsp;

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Có thể xem xét đối tượng `oven` có bounding box bắt đầu tại tọa độ x \= 0.11, rất sát mép trái ảnh. Đây là trường hợp vật thể có khả năng bị cắt mép.

Tôi sẽ đưa trường hợp này lên hỏi Lab Coach để xác nhận cách xử lý đối với vật thể bị cắt khỏi khung hình, tránh tự đưa ra quy tắc khi chưa có hướng dẫn cụ thể.

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`\=… ; `class_name`\=… ; `score`\=… ; số điểm polygon=`polygon_point_count`\=… ; 3–5 điểm đầu của `polygon_xy`\=…

instance\_id \= kitchen-001

class\_name \= person

score \= 0.899318

polygon\_point\_count \= 348

5 điểm đầu của polygon\_xy:

\[446.0, 70.0\]

\[445.0, 71.0\]

\[444.0, 71.0\]

\[443.0, 72.0\]

\[442.0, 72.0\]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

Box chỉ xác định vùng hình chữ nhật bao quanh vật thể, còn polygon mô tả đường biên thực tế của vật thể chi tiết hơn. Nhờ đó, polygon thể hiện được hình dạng, kích thước và những phần lồi lõm của vật thể, thay vì chỉ biết vật thể nằm trong một khung chữ nhật.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

`instance_id` dùng để phân biệt từng đối tượng riêng biệt trong cùng một ảnh, kể cả khi chúng thuộc cùng một class.

Ví dụ, trong ảnh có hai người thì cả hai đều có `class_id` \= 0 và `class_name` \= `person`, nhưng mỗi người sẽ có `instance_id` khác nhau.

Nó không phải class ID vì không dùng để xác định loại đối tượng, cũng không phải tracking ID vì không có chức năng theo dõi đối tượng xuyên suốt nhiều frame hoặc nhiều thời điểm.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Khi vẽ mask, cần bám sát đường biên thực tế của vật thể, không vẽ lấn ra nền hoặc ăn vào phần của vật thể khác. Chỉ đánh dấu những vùng có thể xác định rõ thuộc về đối tượng đang gán nhãn

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

Cần quan sát trực tiếp ảnh `segmentation_prediction.png` để xác định chính xác vị trí biên mờ, vật thể chạm nhau hoặc bị che khuất.

Nếu trường hợp đã có quy định trong guideline thì áp dụng guideline. Nếu không thể xác định ranh giới chính xác hoặc gặp tình huống chưa được hướng dẫn, cần escalation để người phụ trách xem xét, tránh tự suy đoán và tạo ra nhãn không nhất quán.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| :---- | :---- | :---- | :---- | :---- |
| Phân loại ảnh | Một hoặc nhiều nhãn class đại diện cho nội dung của toàn bộ ảnh.&nbsp; | Khó quyết định ảnh có nhiều đối tượng hoặc nhiều hoạt động thì nên chọn class nào.&nbsp; | Quan sát ảnh, chọn nhãn đúng theo guideline.&nbsp; | Kiểm tra nhãn đúng và không bị thiếu.&nbsp; |
| Phát hiện vật thể | Mỗi vật thể có một nhãn class và một bounding box (`bbox_xyxy`).&nbsp; | Box có thể bao thừa nền hoặc bỏ sót vật thể.&nbsp; | Vẽ box bao quanh từng vật thể.&nbsp; | Kiểm tra class, số lượng và vị trí box.&nbsp; |
| Instance segmentation | Mỗi vật thể có một nhãn class và một mask riêng, thường được biểu diễn bằng polygon.&nbsp; | Record `kitchen-001` có 348 điểm polygon, biên phức tạp nên dễ vẽ thừa hoặc thiếu.&nbsp; | Vẽ mask sát biên vật thể, không lấn sang nền hoặc vật thể khác.&nbsp; | Kiểm tra độ chính xác của mask, class và instance ID.&nbsp; |

5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Không tự ý chia sẻ, sao chép hoặc tải dữ liệu lab lên các nền tảng bên ngoài; chỉ sử dụng dữ liệu đúng mục đích và trong phạm vi được cho phép.&nbsp;

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Dừng xử lý và báo ngay cho Lab Coach để được hướng dẫn xử lý.&nbsp;

## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)  
- [ ] `day1_lab_outputs/classification_predictions.json`  
- [ ] `day1_lab_outputs/detection_predictions.json`  
- [ ] `day1_lab_outputs/segmentation_predictions.json`  
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`  
- [ ] `day1_lab_outputs/visuals/classification_top5.png`  
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`  
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`  
- [ ] Ô validation cuối notebook báo `PASS`  
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output  
- [ ] Đã nộp **link repository** trên LMS