# Lab Day 2 — DeepWeeds classification

**Sinh viên:** Hoàng Trung Khải — 2A202602947  
**Trạng thái:** đã hoàn thành sàng lọc backbone (5 cấu hình, seed 0); chưa hoàn thành ablation/chung kết/test

## Tóm tắt

Bài toán phân loại ảnh DeepWeeds thành 9 lớp. Đã chuẩn bị pipeline, kiểm tra dữ liệu fold 0 và chạy sàng lọc 5 backbone trên seed 0. ConvNeXt-Tiny (B03) đạt macro-F1 validation cao nhất 0,9694 và top-1 0,9763. Đây chỉ là kết quả sàng lọc một seed, chưa phải kết luận chung kết. Chưa có đánh giá test; không đưa số tham khảo của bài báo thành kết quả thực nghiệm.

## Dữ liệu và thiết lập

Dùng ảnh Zenodo `images.zip`, MD5 `b7b30f96d466fba86016aa5a26606e0f`, cùng các CSV nhãn gốc của tác giả. Fold 0 có 10.501 ảnh train, 3.501 ảnh validation và 3.507 ảnh test. Các cặp split giao nhau 0 ảnh; hợp có 17.509 ảnh và mọi file đều tồn tại.

| Lớp | Train | Val | Test |
|---|---:|---:|---:|
| Chinee Apple | 675 | 225 | 226 |
| Lantana | 637 | 213 | 213 |
| Parkinsonia | 618 | 206 | 207 |
| Parthenium | 613 | 204 | 205 |
| Prickly Acacia | 637 | 212 | 213 |
| Rubber Vine | 605 | 202 | 202 |
| Siam Weed | 644 | 215 | 215 |
| Snake Weed | 609 | 203 | 204 |
| Negatives | 5.463 | 1.821 | 1.822 |

`Negatives` chiếm hơn nửa dữ liệu; vì vậy macro-F1 là chỉ số chính và accuracy không đủ để so sánh. Tập đầy đủ có 9.106 ảnh Negative và lớp ít nhất có 1.009 ảnh, chênh khoảng 9 lần. Biểu đồ phân bố và ảnh mẫu được lưu trong thư mục `eda/`.

## Kiểm tra pipeline

- Focal loss với `gamma=0` trùng cross-entropy trong phép kiểm tra tensor (sai số 0).
- Mixup và CutMix trả về batch/nhãn trộn đúng kích thước.
- CE khởi tạo trên batch ảnh thật 8 mẫu là 2,2014, gần giá trị tham chiếu `ln(9)=2,1972`; sau 80 bước tối ưu trên batch nhỏ, loss về 0,0.
- ResNet-50 khởi tạo ngẫu nhiên tạo logits 9 lớp; 23,526 triệu tham số. Ước lượng 4,087 GMAC từ hook Conv/Linear.
- Batch ảnh sau augmentation và đường cong overfit được lưu trong `eda/augmented_batch.png` và `eda/tiny_batch_overfit.png`.
- Các lượt huấn luyện đã chạy trên CUDA theo log. Latency GPU chưa được benchmark; cần đo riêng để so sánh chi phí suy luận.

## So sánh backbone

Đã chạy cùng công thức nền ở fold 0, seed 0. Bảng chi tiết nằm trong sheet `Backbones` của `results.xlsx`; log theo epoch, cấu hình, validation logits, biểu đồ và dự đoán validation được lưu trong `runs/`, `curves/` và `predictions/`.

| exp_id | Backbone | Macro-F1 val | Top-1 val | Tham số (M) | GMAC | s/epoch |
|---|---|---:|---:|---:|---:|---:|
| B01 | ResNet-50 | 0,7742 | 0,8343 | 23,526 | 4,087 | 57,20 |
| B02 | ResNeXt-50 32x4d | 0,6479 | 0,7529 | 22,998 | 4,228 | 54,16 |
| B03 | ConvNeXt-Tiny | **0,9694** | **0,9763** | 27,827 | 4,455 | 47,47 |
| B04 | DeiT-Small | 0,9511 | 0,9640 | 21,669 | 4,241 | 36,54 |
| B05 | MobileNetV3-Large | 0,6564 | 0,7589 | 4,214 | 0,215 | 31,72 |

Thời gian là số ghi trong log của môi trường huấn luyện. Latency batch-1 chưa đo, nên chưa thể so sánh chi phí suy luận.

## Ablation công thức huấn luyện

Chưa chạy. Script sweep chuẩn bị các đối chứng một yếu tố so với T00: đóng băng backbone, color jitter, label smoothing, focal loss, balanced sampler và Mixup; T07 là cấu hình kết hợp được đánh dấu riêng. So sánh phải dựa trên validation và phải thận trọng vì vòng sàng chỉ dùng một seed.

## Phương pháp suy luận và độ trễ

Chưa đo. Cần so sánh ít nhất bốn phương pháp ngoài 1-view, gồm lật ngang, multi-crop/scale, gộp xác suất/logit, temperature scaling khớp trên validation và/hoặc ensemble. Latency cần warmup ≥10 lượt, đồng bộ CUDA và ≥50 lượt đo để báo p50/p95/p99.

## Chung kết, phân tích lỗi và kết luận

Chưa thực hiện. B03 là lựa chọn tạm thời theo validation seed 0; cần kiểm tra độ ổn định qua ablation và ít nhất ba seed trước khi gọi là cấu hình cuối. Sau đó huấn luyện B03 và mốc T00 đủ seed, chạy test một lần mỗi seed, lưu dự đoán test theo định dạng `eval.py`, rồi báo cáo mean ± std, F1 từng lớp, ECE và ma trận nhầm lẫn. Không dùng test để lựa chọn cấu hình.

## Hạn chế

Kết quả backbone hiện mới có một seed screening; chưa có ablation, đo latency, dự đoán test hay đánh giá chung kết nhiều seed. Đánh giá dựa trên một fold có thể lạc quan vì split ngẫu nhiên không phân theo địa điểm. Thí nghiệm cuối cần ghi phần cứng, phiên bản thư viện, seed, thời gian và cấu hình đầy đủ.
