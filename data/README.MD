# Bộ Dữ Liệu Ảnh Phân Loại Khối U Não

## 1️ Giới thiệu
Bộ dữ liệu này được dùng để **phân loại ảnh MRI não** thành các nhóm khác nhau (ví dụ: có u, không u).  
Dữ liệu đã được **làm sạch** và **chuẩn hóa kích thước ảnh** về **256×256 pixel**.

---

## 2️ Cấu trúc thư mục


---

## 3️ Thông tin dữ liệu

| Tập dữ liệu | Số ảnh      | Kích thước ảnh | Số lớp | Định dạng ảnh |
|--------------|-------------|----------------|---------|----------------|
| Training | ≈ 5.700 ảnh | 256×256×3 (RGB) | 4 | `.jpg`, `.png` |
| Testing | ≈ 1.300 ảnh | 256×256×3 (RGB) | 4 | `.jpg`, `.png` |

**Tổng số mẫu:** khoảng **7.000 ảnh**

---

## 4️ Các lớp (labels)

| Tên lớp | Mô tả |
|----------|--------|
| `glioma_tumor` | Ảnh MRI có u não dạng Glioma |
| `meningioma_tumor` | Ảnh MRI có u não dạng Meningioma |
| `pituitary_tumor` | Ảnh MRI có u não dạng Pituitary |
| `no_tumor` | Ảnh MRI bình thường, không có u |

---

## 5️ Tiền xử lý và chuẩn hóa
Trước khi đưa vào mô hình, dữ liệu đã được:
- **Làm sạch**: loại bỏ ảnh lỗi, sai kích thước, định dạng không hợp lệ.
- **Resize**: tất cả ảnh được chuyển về kích thước **256×256 pixel**.
- **Mã hóa nhãn**:
  - Dùng **LabelEncoder** → nhãn số (0–3)
  - Dùng **OneHotEncoder** → vector nhị phân (cho mô hình softmax hoặc neural network)
- **Chuẩn hóa giá trị pixel**: chia cho 255 → đưa về khoảng `[0, 1]`.

---

## 6 Mục đích sử dụng
Bộ dữ liệu được sử dụng để **huấn luyện và đánh giá mô hình phân loại ảnh**, cụ thể:
- **Naive Bayes**
- **Logistic Regression**

Mục tiêu là so sánh độ chính xác giữa các phương pháp phân loại cơ bản trên cùng tập dữ liệu.

---

## 7️ Ghi chú
- Mỗi lớp được lưu trong thư mục riêng, tương ứng với tên lớp.
- Ảnh có định dạng RGB.
- Nên giữ nguyên cấu trúc thư mục để đảm bảo code đọc đúng dữ liệu.
- Nếu thêm dữ liệu mới, cần đảm bảo cùng kích thước và format.

---

**Nguồn dữ liệu gốc:** [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

*Ngày cập nhật:* 2025-10-21
