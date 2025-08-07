import streamlit as st
import numpy as np
import joblib

# Tải mô hình và scaler đã huấn luyện
scaler = joblib.load("scaler.pkl")
model = joblib.load("stacking_model.pkl")

# Giao diện chính
st.title("🌱 Hệ thống đề xuất cây trồng bằng Stacking Classifier")
st.markdown("### Nhập các thông số đầu vào:")

# Các biến đầu vào từ giao diện
n = st.slider("N - Nitơ", 0, 140, 90)
p = st.slider("P - Phốt pho", 0, 140, 42)
k = st.slider("K - Kali", 0, 200, 43)
ph = st.number_input("pH đất", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
temperature = st.number_input("Nhiệt độ (°C)", value=25.0)
humidity = st.number_input("Độ ẩm (%)", value=80.0)
rainfall = st.number_input("Lượng mưa (mm)", value=100.0)

# Dự đoán khi người dùng nhấn nút
if st.button("🌾 Dự đoán cây trồng phù hợp"):
    # Chuẩn bị dữ liệu đầu vào dưới dạng mảng 2 chiều
    input_data = np.array([[n, p, k, ph, temperature, humidity, rainfall]])

    # Chuẩn hóa đầu vào giống lúc training
    input_scaled = scaler.transform(input_data)

    # Dự đoán cây trồng
    prediction = model.predict(input_scaled)

    # Hiển thị kết quả
    st.success(f"✅ Cây trồng được đề xuất: **{prediction[0]}**")
