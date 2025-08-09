# 📄 README – Quy trình thực hiện tiểu luận
## Ứng dụng Stacking Classifier trong Hệ thống Đề xuất Cây trồng Thông minh cho Nông dân Thành phố Huế

---

## 🎯 Mục tiêu
Xây dựng một hệ thống khuyến nghị cây trồng thông minh dựa trên điều kiện đất đai, khí hậu, giúp nông dân tại Thành phố Huế lựa chọn cây trồng phù hợp.  
Hệ thống sử dụng **Stacking Classifier** để kết hợp nhiều mô hình học máy, nâng cao độ chính xác dự đoán.

---

## 🔍 Quy trình thực hiện

### 1. **Nghiên cứu & Khảo sát**
- Tìm hiểu các bài toán tương tự về hệ thống gợi ý cây trồng.
- Khảo sát điều kiện nông nghiệp tại khu vực Huế.
- Xác định bộ đặc trưng quan trọng: loại đất, pH, lượng mưa, nhiệt độ, độ ẩm…

---

### 2. **Thu thập & Tiền xử lý dữ liệu**
- **Nguồn dữ liệu**: Bộ dữ liệu cây trồng từ nguồn công khai (Kaggle).
- **Tiền xử lý**:
  - Làm sạch dữ liệu: loại bỏ giá trị thiếu, ngoại lai.
  - Chuẩn hóa và mã hóa dữ liệu.
  - Chia tập dữ liệu thành train/test theo tỷ lệ 80/20.

---

### 3. **Xây dựng & Huấn luyện mô hình**
- Huấn luyện các mô hình cơ sở:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - Support Vector Machine (SVM)
- Tối ưu siêu tham số bằng **Grid Search** và **Random Search**.
- Kết hợp các mô hình bằng **Stacking Classifier**.

---

### 4. **Đánh giá mô hình**
- Sử dụng **K-Fold Cross-Validation** để kiểm tra tính ổn định.
- So sánh kết quả độ chính xác, precision, recall, F1-score giữa các mô hình.
- Kết quả: **Stacking Classifier đạt ~93% accuracy**, cao hơn các mô hình đơn lẻ.

---

### 5. **Giải thích mô hình**
- Sử dụng **LIME (Local Interpretable Model-agnostic Explanations)** để:
  - Hiển thị các đặc trưng quan trọng ảnh hưởng đến quyết định gợi ý.
  - Tăng tính minh bạch của hệ thống.

---

### 6. **Triển khai ứng dụng**
- Giao diện web được xây dựng bằng **Streamlit**.
- Chức năng chính:
  - Nhập thông tin điều kiện đất/khí hậu.
  - Hiển thị cây trồng gợi ý.
  - Giải thích lý do gợi ý.
- Ứng dụng có thể chạy trên **máy cục bộ** hoặc triển khai **Streamlit Cloud**.

---

## 🚀 Kết quả & Ý nghĩa
- Mô hình Stacking cho kết quả dự đoán chính xác hơn so với mô hình đơn.
- Hệ thống có thể hỗ trợ nông dân ra quyết định canh tác hợp lý.
- Mở ra khả năng mở rộng:
  - Tích hợp dữ liệu thời tiết thời gian thực.
  - Phân loại đa nhãn để gợi ý nhiều loại cây phù hợp.

---

## 📚 Tác giả & Liên hệ
- **Sinh viên**: Hoàng Kim Thiên – 22T1020444  
- **Giáo viên hướng dẫn**: TS. Lê Quang Chiến  
- **Trường**: Đại học Khoa học – Đại học Huế
