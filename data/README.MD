# Bộ Dữ Liệu Ảnh Phân Loại Khối U Não (MRI)

## 1️ Giới thiệu
Bộ dữ liệu này được dùng để **phân loại ảnh MRI não** thành 4 nhóm: 3 loại khối u khác nhau (`glioma`, `meningioma`, `pituitary`) và 1 nhóm không có u (`notumor`).

Dữ liệu đã được **làm sạch** và **chuẩn hóa kích thước ảnh** về **128x128 pixel** (thay vì 256x256 như ghi chú ban đầu) để đồng nhất với thông tin trong bảng và phần tiền xử lý.

---

## 2️ Cấu trúc thư mục
Dữ liệu được tổ chức theo cấu trúc thư mục chuẩn, với các lớp được phân tách thành các thư mục con riêng biệt trong cả tập `Training` và `Testing`.

---

## 3️ Thông tin dữ liệu
Bảng dưới đây tóm tắt thông tin chi tiết về bộ dữ liệu:

| Tập dữ liệu | Số ảnh | Kích thước ảnh | Số lớp | Định dạng ảnh |
| :--- | :--- | :--- | :--- | :--- |
| Training | ~5.700 ảnh | 128x128x3 (RGB) | 4 | `.jpg`, `.png` |
| Testing | ~1.300 ảnh | 128x128x3 (RGB) | 4 | `.jpg`, `.png` |
| **Tổng số** | **~7.000 ảnh** | | **4** | |

---

## 4️ Các lớp (labels)
Dữ liệu được phân loại thành 4 lớp (class) riêng biệt:

| Tên lớp | Mô tả |
| :--- | :--- |
| `glioma` | Ảnh MRI có u não dạng Glioma |
| `meningioma` | Ảnh MRI có u não dạng Meningioma |
| `pituitary` | Ảnh MRI có u não dạng Pituitary |
| `notumor` | Ảnh MRI bình thường, không có u |

---

## 5️ Tiền xử lý và chuẩn hóa
Trước khi đưa vào mô hình, dữ liệu đã được (hoặc cần được) xử lý qua các bước sau:

* **Làm sạch**: Loại bỏ ảnh lỗi, sai kích thước, hoặc định dạng không hợp lệ.
* **Resize**: Tất cả ảnh được đồng nhất về kích thước **128x128 pixel**.
* **Chuẩn hóa giá trị pixel**: Chia giá trị của mỗi pixel cho 255.0 để đưa dữ liệu về khoảng `[0, 1]`.
* **Mã hóa nhãn (Label Encoding)**:
    * Dùng **LabelEncoder** để chuyển nhãn tên (string) thành nhãn số (ví dụ: 0, 1, 2, 3) cho các mô hình như Logistic Regression.
    * Dùng **OneHotEncoder** (tùy chọn) để chuyển nhãn số thành vector nhị phân (cho mô hình neural network).
* **Trích xuất đặc trưng HOG (Histogram of Oriented Gradients)**:
    * Do các mô hình ML cổ điển (Naive Bayes, Logistic Regression) không thể xử lý trực tiếp ảnh thô (với $128 \times 128 \times 3 = 49.152$ đặc trưng), chúng ta cần trích xuất đặc trưng.
    * HOG là một kỹ thuật hiệu quả để nắm bắt thông tin về **hình dạng (shape)** và **kết cấu (texture)** (ví dụ: đường viền, góc cạnh của khối u) và biểu diễn chúng dưới dạng một *vector đặc trưng* (feature vector) có số chiều nhỏ hơn nhiều. Vector này sau đó được dùng làm đầu vào (`X`) cho mô hình.

---

## 6️ Mục đích sử dụng
Bộ dữ liệu được sử dụng để **huấn luyện và đánh giá mô hình phân loại ảnh**, cụ thể là so sánh các mô hình Machine Learning cơ bản:
* **Naive Bayes** (Gaussian Naive Bayes)
* **Logistic Regression**

Mục tiêu là so sánh độ chính xác (Accuracy, Precision, Recall, F1-Score) giữa các phương pháp phân loại này trên cùng một tập dữ liệu đã trích xuất đặc trưng HOG.

---

## 7️ Quy trình làm việc (Workflow) đề xuất
1.  **Đọc dữ liệu**: Duyệt qua các thư mục `Training` và `Testing` để đọc ảnh và gán nhãn (`y`) tương ứng.
2.  **Tiền xử lý & Trích xuất đặc trưng**: Với mỗi ảnh:
    * Resize về `128x128`.
    * Chuẩn hóa (chia 255).
    * Tính toán **vector đặc trưng HOG**.
3.  **Tập hợp dữ liệu**: Tạo ra các tập `X_train_hog`, `y_train_encoded`, `X_test_hog`, `y_test_encoded`.
4.  **Huấn luyện mô hình**: Dùng `X_train_hog` và `y_train_encoded` để `fit` các mô hình (Naive Bayes, Logistic Regression).
5.  **Đánh giá**: Dùng các mô hình đã huấn luyện để `predict` trên `X_test_hog` và so sánh với `y_test_encoded` để tính toán các chỉ số
    đánh giá.

---

## 8️ Ghi chú
* Mỗi lớp được lưu trong thư mục riêng, tương ứng với tên lớp.
* Ảnh có định dạng RGB (3 kênh màu), ngay cả khi chúng trông giống ảnh xám. HOG có thể được cấu hình để xử lý cả 3 kênh.
* Nên giữ nguyên cấu trúc thư mục để đảm bảo code đọc đúng dữ liệu.
* Nếu thêm dữ liệu mới, cần đảm bảo cùng kích thước và format.

---

## 9️ Nguồn dữ liệu gốc
* **Kaggle:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

*Ngày cập nhật:* 2025-10-28
*(Gemini hoàn thiện: 2025-11-06)*