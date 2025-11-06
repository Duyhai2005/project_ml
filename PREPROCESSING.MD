# README: Quy trình Tiền xử lý Dữ liệu (Preprocessing)

## 1. Giới thiệu

File notebook `preprocessing_data.ipynb` này là bước đầu tiên và quan trọng nhất trong dự án **Phân loại Khối u Não (Brain Tumor Classification)**.

Mục tiêu chính của notebook này là lấy dữ liệu ảnh thô (raw images) từ các thư mục, thực hiện một loạt các bước tiền xử lý và trích xuất đặc trưng, sau đó lưu lại các bộ dữ liệu "sạch" (dưới dạng mảng NumPy) để sẵn sàng cho việc huấn luyện và đánh giá mô hình Machine Learning ở các bước sau.

## 2. Mục đích

Quy trình này thực hiện các nhiệm vụ sau:

* **Tải dữ liệu**: Đọc ảnh từ các thư mục `Training` và `Testing`.
* **Tiền xử lý ảnh**: Resize ảnh về một kích thước đồng nhất (128x128).
* **Trích xuất đặc trưng**: Chuyển đổi mỗi ảnh từ dạng ma trận 2D/3D thành một **vector đặc trưng HOG** (Histogram of Oriented Gradients). Đây là bước then chốt để các mô hình ML cổ điển (như Logistic Regression, SVM) có thể "hiểu" được nội dung ảnh.
* **Mã hóa nhãn**: Chuyển đổi các nhãn dạng chữ (ví dụ: 'glioma', 'notumor') thành dạng số (0, 1, 2, 3) bằng `LabelEncoder`.
* **Chuẩn hóa dữ liệu**: Sử dụng `StandardScaler` để chuẩn hóa các vector đặc trưng HOG, đảm bảo các đặc trưng có cùng thang đo (mean=0, variance=1).
* **Giảm chiều dữ liệu**: Áp dụng PCA và LDA để giảm số chiều của vector đặc trưng, giúp tăng tốc độ huấn luyện và có thể cải thiện hiệu suất mô hình.
* **Lưu kết quả**: Lưu tất cả các mảng dữ liệu đã xử lý (`X_train_scaled`, `X_train_pca`, `y_train_encoded`, v.v.) vào bộ nhớ đệm (sử dụng `%store`) để các notebook khác có thể tái sử dụng.

## 3. Các bước thực thi chi tiết

Notebook được chia thành các bước logic rõ ràng:

1.  **Import thư viện**: Tải các thư viện cần thiết (`numpy`, `cv2`, `os`, `sklearn`, `skimage`, `matplotlib`...).
2.  **Định nghĩa đường dẫn**: Xác định vị trí của các thư mục `train_path` và `test_path`.
3.  **Tải và Resize ảnh**:
    * Duyệt qua các thư mục con (tương ứng với các lớp: `glioma`, `meningioma`, `pituitary`, `notumor`).
    * Đọc ảnh bằng `cv2.imread`.
    * Resize ảnh về kích thước `(128, 128)`.
    * Tạo ra các tập `X_train`, `y_train`, `X_test`, `y_test`.
4.  **Trích xuất đặc trưng HOG**:
    * Định nghĩa hàm `feature_extractor` sử dụng `skimage.feature.hog`.
    * Áp dụng hàm này cho tất cả ảnh trong `X_train` và `X_test` để tạo ra `X_train_hog` và `X_test_hog`.
5.  **Mã hóa nhãn (Label Encoding)**:
    * Sử dụng `LabelEncoder` để `fit_transform` trên `y_train` và `transform` trên `y_test`.
    * Kết quả là `y_train_encoded` và `y_test_encoded`.
6.  **Chuẩn hóa dữ liệu (Scaling)**:
    * Sử dụng `StandardScaler` để `fit_transform` trên `X_train_hog` và `transform` trên `X_test_hog`.
    * Kết quả là `X_train_scaled` và `X_test_scaled`.
7.  **Giảm chiều (Dimensionality Reduction)**:
    * **PCA**: Giảm dữ liệu HOG đã chuẩn hóa, 1 lần giảm còn 6 chiều, 1 lần giảm để giữ 90% thông tin.
    * **LDA**: Giảm dữ liệu HOG đã chuẩn hóa xuống còn 3 thành phần (số lớp - 1).
8.  **Phân tích (Tùy chọn)**:
    * Thực hiện K-Means clustering (K=4) trên cả dữ liệu PCA và LDA.
    * Tính toán các chỉ số (Silhouette, Davies-Bouldin, ARI, NMI) để đánh giá xem các cụm (cluster) không giám sát tìm được có khớp với các lớp (class) thật hay không.
9.  **Lưu kết quả**:
    * Sử dụng magic command `%store` của IPython để lưu tất cả các biến quan trọng vào bộ nhớ đệm của Jupyter.

## 4. Kết quả đầu ra (Outputs)

Sau khi chạy notebook này, các biến sau sẽ được lưu và sẵn sàng để sử dụng ở các notebook khác (ví dụ: `model_training.ipynb`) bằng lệnh `%store -r <tên_biến>`:

* **Dữ liệu HOG đã chuẩn hóa (Dùng cho hầu hết mô hình):**
    * `X_train_scaled`
    * `X_test_scaled`
* **Dữ liệu PCA (Dùng để trực quan hóa hoặc mô hình nhẹ):**
    * `X_train_pca_6 (trực quan theo yêu cầu project)`
    * `X_test_pca_6 (trực quan theo yêu cầu project)`
    * `X_train_pca (giữ lại 90% thông tin để chạy mô hình dự đoán)`
    * `X_test_pca (giữ lại 90% thông tin để chạy mô hình dự đoán)`
* **Dữ liệu LDA (Dùng để trực quan hóa hoặc mô hình phân biệt lớp):**
    * `X_train_lda`
    * `X_test_lda`
* **Nhãn đã mã hóa:**
    * `y_train_encoded`
    * `y_test_encoded`

## 5. Thư viện yêu cầu

Để chạy notebook này, bạn cần cài đặt các thư viện Python sau:

* `numpy`
* `pandas`
* `opencv-python` (cho `cv2`)
* `matplotlib`
* `seaborn`
* `scikit-learn` (cho `LabelEncoder`, `StandardScaler`, `PCA`, `LDA`, `KMeans`, metrics)
* `scikit-image` (cho `hog`)
* `tqdm` (để theo dõi tiến độ vòng lặp)
* `jupyter` (hoặc `ipython`)