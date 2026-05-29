# Brain Tumor MRI Classification

Dự án Machine Learning dùng ảnh MRI não để phân loại 4 lớp: `glioma`, `meningioma`, `pituitary` và `notumor`. Pipeline chính nằm trong hai notebook:

- `preprocessing_data.ipynb`: làm sạch ảnh, chuẩn hóa, trích xuất đặc trưng HOG, giảm chiều bằng PCA/LDA và trực quan hóa dữ liệu.
- `classifier.ipynb`: huấn luyện, đánh giá các mô hình phân loại và chạy thêm một thí nghiệm hồi quy phụ.

> Lưu ý: Đây là dự án học tập/nghiên cứu, không dùng như công cụ chẩn đoán y tế.

## Cấu trúc dự án

```text
project_ml/
├── classifier.ipynb              # Huấn luyện, đánh giá mô hình phân loại và hồi quy
├── preprocessing_data.ipynb      # Tiền xử lý dữ liệu, HOG, PCA, LDA, clustering
├── PREPROCESSING.MD              # Tài liệu riêng cho bước tiền xử lý
└── data/
    ├── README.MD                 # Mô tả bộ dữ liệu
    ├── raw/
    │   ├── Training/
    │   └── Testing/
    └── cleaned/
        ├── Training/
        └── Testing/
```

## Bộ dữ liệu

Dữ liệu gốc lấy từ Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

Repo hiện có cả ảnh thô trong `data/raw` và ảnh đã xử lý trong `data/cleaned`. Mỗi phần có cùng số lượng ảnh:

| Tập | glioma | meningioma | notumor | pituitary | Tổng |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 1321 | 1339 | 1595 | 1457 | 5712 |
| Testing | 300 | 306 | 405 | 300 | 1311 |
| Tổng | 1621 | 1645 | 2000 | 1757 | 7023 |

Ảnh sau làm sạch được resize về `128x128x3`. Notebook ghi nhận ảnh raw ban đầu có kích thước trung bình khoảng `445.9x448.3` px.

## Pipeline tiền xử lý

`preprocessing_data.ipynb` thực hiện các bước chính:

1. Đọc dữ liệu trong `data/raw/Training` và `data/raw/Testing`.
2. Làm sạch ảnh bằng cách cắt vùng não chính:
   - chuyển ảnh sang grayscale,
   - Gaussian blur,
   - threshold,
   - erosion/dilation,
   - tìm contour lớn nhất,
   - crop theo vùng chính.
3. Resize ảnh về `128x128` và lưu vào `data/cleaned`.
4. Đọc ảnh cleaned, mã hóa nhãn bằng `LabelEncoder` và `OneHotEncoder`.
5. Chuẩn hóa pixel về khoảng `[0, 1]`.
6. Trích xuất HOG từ ảnh grayscale với cấu hình:
   - `orientations=9`
   - `pixels_per_cell=(8, 8)`
   - `cells_per_block=(2, 2)`
   - `block_norm='L2-Hys'`
   - `transform_sqrt=True`
7. Chuẩn hóa đặc trưng HOG bằng `StandardScaler`.
8. Giảm chiều:
   - PCA giữ 90% phương sai, từ `8100` chiều xuống `1213` chiều.
   - LDA giảm xuống `3` chiều vì bài toán có 4 lớp.
9. Chia train/validation theo các tỷ lệ `4:1`, `7:3`, `6:4`.
10. Trực quan hóa PCA/LDA và thử K-Means với `k=4`.

Kích thước đặc trưng sau HOG:

| Tập | Shape |
| --- | --- |
| `X_train_hog` | `(5712, 8100)` |
| `X_test_hog` | `(1311, 8100)` |
| `X_train_pca` | `(5712, 1213)` |
| `X_test_pca` | `(1311, 1213)` |
| `X_train_lda` | `(5712, 3)` |
| `X_test_lda` | `(1311, 3)` |

## Mô hình

`classifier.ipynb` so sánh hai mô hình phân loại:

- `GaussianNB`
- `LogisticRegression` softmax với `multi_class='multinomial'`, `solver='lbfgs'`, `max_iter=1000`

Dữ liệu đầu vào được so sánh theo ba dạng:

- HOG gốc
- HOG sau PCA
- HOG sau LDA

Notebook cũng có bước thử regularization cho Logistic Regression trên dữ liệu PCA để giảm overfitting, và một thí nghiệm hồi quy phụ dự đoán xác suất thuộc lớp `notumor`.

## Kết quả chính

Kết quả cuối cùng đang lưu trong notebook trên tập test:

| Loại dữ liệu | Naive Bayes | Logistic Regression (Softmax) |
| --- | ---: | ---: |
| HOG gốc | 0.6171 | 0.8635 |
| Sau PCA | 0.5774 | 0.8360 |
| Sau LDA | 0.7239 | 0.7285 |

Mô hình tốt nhất theo accuracy test đang là Logistic Regression trên HOG gốc với accuracy `0.8635`. Tuy nhiên mô hình này có dấu hiệu overfitting vì accuracy train đạt `1.0000`, test đạt `0.8635`.

Thử nghiệm regularization cho Logistic Regression trên PCA cho kết quả đang lưu:

- Train accuracy: `0.9884`
- Test accuracy: `0.8596`
- Chênh lệch train-test: `0.1288`

## Thí nghiệm hồi quy phụ

Notebook chuyển bài toán sang dạng hồi quy bằng cách:

1. Chọn lớp mục tiêu `notumor`.
2. Huấn luyện Logistic Regression để lấy xác suất softmax.
3. Dùng xác suất thuộc lớp `notumor` làm nhãn hồi quy.
4. So sánh `LinearRegression` và `MLPRegressor` trên HOG gốc và dữ liệu PCA giảm còn 1/3 số chiều.

Kết quả đang lưu:

| Dữ liệu đầu vào | Mô hình hồi quy | MSE | R-squared |
| --- | --- | ---: | ---: |
| HOG nguyên bản | Linear Regression | 0.050389 | 0.749976 |
| HOG nguyên bản | MLPRegressor | 0.013150 | 0.934750 |
| Giảm chiều 1/3 | Linear Regression | 0.026210 | 0.869950 |
| Giảm chiều 1/3 | MLPRegressor | 0.011653 | 0.942177 |

## Cài đặt môi trường

Dự án chưa có `requirements.txt`. Có thể tạo môi trường và cài các thư viện cần thiết như sau:

```powershell
cd C:\project\project_ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pandas opencv-python imutils pillow matplotlib seaborn scikit-learn scikit-image tqdm jupyter
```

Nếu dùng Jupyter Notebook:

```powershell
jupyter notebook
```

## Cách chạy

Chạy notebook từ thư mục gốc của dự án:

```powershell
cd C:\project_ml
```

Thứ tự khuyến nghị:

1. Mở và chạy toàn bộ `preprocessing_data.ipynb`.
2. Mở và chạy toàn bộ `classifier.ipynb`.

`classifier.ipynb` có dùng `%run preprocessing_data.ipynb`, vì vậy khi chạy classifier notebook, toàn bộ pipeline tiền xử lý sẽ được chạy lại trong cùng kernel.

## Lưu ý kỹ thuật

- Dataset được lưu cả raw và cleaned nên dung lượng repo lớn và có dữ liệu trùng lặp sau xử lý.
- Một số cell chia train/validation cho PCA/LDA chưa đặt `random_state`, nên kết quả validation có thể thay đổi khi chạy lại.