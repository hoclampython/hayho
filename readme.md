# 📊 Hệ Thống Phân Tích & Dự Báo Gian Lận Báo Cáo Tài Chính (Beneish M-Score & AI)

Ứng dụng Web tương tác được xây dựng bằng **Python** và **Streamlit**, ứng dụng phương pháp định lượng **Beneish M-Score (1999)** kết hợp thuật toán học máy **Hồi quy Logistic (Logistic Regression)** nhằm sàng lọc, phát hiện và cảnh báo rủi ro thao túng số liệu Báo cáo tài chính (BCTC) của doanh nghiệp.

Ứng dụng phục vụ đắc lực cho:
- 🔍 **Kiểm toán viên & Thanh tra tài chính**: Nhận diện sớm các khoản mục bất thường và rủi ro gian lận.
- 🏦 **Chuyên viên Thẩm định Tín dụng & Ngân hàng**: Đánh giá độ tin cậy của hồ sơ vay vốn doanh nghiệp.
- 📈 **Nhà đầu tư Chứng khoán**: Tránh các "cạm bẫy lợi nhuận ảo", doanh nghiệp "xào nấu" số liệu trước khi niêm yết hoặc phát hành cổ phiếu.

---

## 🌟 Tính Năng Nổi Bật Của Web App

1. **Khám phá & Trực quan hóa Dữ liệu (EDA)**:
   - Thống kê tỷ lệ mẫu an toàn / thao túng trong tập dữ liệu.
   - Trực quan hóa phân phối từng chỉ số tài chính thông qua Boxplot và Histogram tương tác.
2. **Huấn luyện & Đánh giá Mô hình Machine Learning**:
   - Chia tập Train/Test linh hoạt, tùy chỉnh tham số mô hình.
   - Bảng hệ số hồi quy $\beta$ và Tỷ số chênh (Odds Ratio) kèm diễn giải nghiệp vụ kế toán chuyên sâu.
   - Đánh giá toàn diện: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
   - Ma trận nhầm lẫn (Confusion Matrix Heatmap) và Biểu đồ đường cong ROC tương tác cao (Plotly).
3. **Chẩn đoán Rủi ro Doanh nghiệp Đơn lẻ (Single Company Diagnosis)**:
   - Nhập 8 chỉ số tài chính hoặc tải kịch bản mẫu (DN lành mạnh, DN rủi ro cao, DN ranh giới).
   - Hiển thị song song: **Xác suất Gian lận Machine Learning** (Đồng hồ đo rủi ro Gauge Chart) và **Điểm Beneish M-Score gốc** (so sánh ngưỡng $-1.78$).
   - Phân tích đóng góp chi tiết (Feature Contribution): Chỉ rõ chỉ số nào đang làm tăng rủi ro BCTC nhiều nhất.
4. **Dự báo Hàng Loạt (Batch Prediction & Screening)**:
   - Tải lên danh sách nhiều công ty qua file CSV (có sẵn file mẫu).
   - Tự động chấm điểm, phân loại 3 cấp độ rủi ro (Đỏ, Vàng, Xanh) và xuất báo cáo kết quả ra file CSV.
5. **Cẩm nang Chỉ số & Phương pháp luận BCTC**:
   - Hướng dẫn chi tiết công thức và bản chất kế toán của 8 chỉ số Beneish M-Score.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py              # Mã nguồn chính của ứng dụng Streamlit
├── MScore_data.csv     # Tập dữ liệu mẫu 8 chỉ số M-Score và cờ gian lận (FRAUD_FLAG)
├── requirements.txt    # Danh sách thư viện Python phụ thuộc
├── test.py             # Script notebook huấn luyện mô hình gốc
└── readme.md           # Tài liệu hướng dẫn sử dụng và triển khai
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Localhost)

### Bước 1: Chuẩn bị môi trường Python
Khuyến nghị sử dụng Python phiên bản `3.9` đến `3.11`.

### Bước 2: Mở Terminal / PowerShell tại thư mục dự án
Cài đặt toàn bộ các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi chạy lệnh trên, trình duyệt web sẽ tự động mở ứng dụng tại địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Đẩy Code Lên GitHub & Triển Khai Trên Streamlit Cloud

### 1. Đẩy mã nguồn lên GitHub:
1. Tạo một tài khoản trên [GitHub](https://github.com) (nếu chưa có).
2. Tạo một Repository mới (ví dụ đặt tên: `financial-fraud-detection-app`), chọn chế độ **Public**.
3. Mở Terminal tại thư mục này trên máy của bạn và chạy các lệnh sau:
   ```bash
   git init
   git add .
   git commit -m "Khoi tao web app phan tich gian lan BCTC Beneish M-Score"
   git branch -M main
   git remote add origin https://github.com/<TEN_GITHUB_CUA_BAN>/financial-fraud-detection-app.git
   git push -u origin main
   ```

### 2. Triển khai miễn phí trên Streamlit Cloud (1-Click Deploy):
1. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm nút **"Create app"** hoặc **"New app"**.
3. Chọn Repository vừa tạo: `financial-fraud-detection-app`.
4. Điền các trường cấu hình:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Bấm nút **"Deploy!"**. 
6. Chỉ sau 1-2 phút, Streamlit Cloud sẽ cài đặt `requirements.txt` và khởi chạy web app trực tuyến với đường link công khai (ví dụ: `https://financial-fraud-detection.streamlit.app`) để bạn chia sẻ cho đồng nghiệp, giảng viên hoặc đối tác.

---

## 📖 Bảng Giải Thích 8 Chỉ Số Beneish M-Score

| Chỉ số | Tên đầy đủ | Ý nghĩa phát hiện gian lận | Chiều hướng rủi ro |
| :--- | :--- | :--- | :--- |
| **DSRI** | Days Sales in Receivables Index | Số ngày thu tiền bình quân | DSRI > 1: Tăng rủi ro ghi nhận doanh thu ảo / ghi nhận non |
| **GMI** | Gross Margin Index | Chỉ số tỷ lệ lãi gộp | GMI > 1: Biên lãi gộp suy giảm, tăng áp lực thao túng số liệu |
| **AQI** | Asset Quality Index | Chỉ số chất lượng tài sản | AQI > 1: Tăng rủi ro vốn hóa chi phí vào tài sản thay vì hạch toán lỗ |
| **SGI** | Sales Growth Index | Tăng trưởng doanh thu | SGI cao: Doanh nghiệp tăng trưởng nóng chịu áp lực duy trì tăng trưởng |
| **DEPI**| Depreciation Index | Tỷ lệ khấu hao tài sản cố định | DEPI > 1: Tỷ lệ khấu hao giảm, kéo dài tuổi thọ TSCĐ để tăng lợi nhuận |
| **SGAI**| Sales, General & Admin Expense Index | Tỷ lệ chi phí BH & QLDN | SGAI > 1: Hiệu quả vận hành suy giảm, áp lực làm đẹp lợi nhuận |
| **TATA**| Total Accruals to Total Assets | Tổng biến động dồn tích / Tổng tài sản | TATA cao: Lợi nhuận không có tiền thật bảo chứng (chất lượng lợi nhuận thấp) |
| **LVGI**| Leverage Index | Chỉ số đòn bẩy tài chính | LVGI > 1: Nợ vay tăng cao, tăng nguy cơ vi phạm cam kết tín dụng |

### ⚖️ Công thức M-Score chuẩn gốc (Beneish, 1999):
$$M\text{-Score} = -4.84 + 0.920 \times DSRI + 0.528 \times GMI + 0.404 \times AQI + 0.892 \times SGI + 0.115 \times DEPI - 0.172 \times SGAI + 4.037 \times TATA + 0.0327 \times LVGI$$

- Nếu **$M\text{-Score} > -1.78$**: Doanh nghiệp có khả năng cao đang thao túng Báo cáo tài chính.
- Nếu **$M\text{-Score} \le -1.78$**: Doanh nghiệp ở mức an toàn, ít có dấu hiệu thao túng.

---

## 🛠️ Công Nghệ Sử Dụng
- **Ngôn ngữ:** Python 3.9+
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Xử lý Dữ liệu & Tính toán:** Pandas, NumPy
- **Học máy (Machine Learning):** Scikit-Learn (Logistic Regression, Train-Test Split, Metrics)
- **Trực quan hóa Dữ liệu:** Plotly Interactive, Seaborn, Matplotlib

---
*Dự án được thiết kế chuyên nghiệp, sẵn sàng cho việc nghiên cứu học thuật, phân tích thực tế và triển khai sản phẩm.*
