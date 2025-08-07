import streamlit as st
import numpy as np
import joblib
import os

# Lấy đường dẫn tuyệt đối tới thư mục hiện tại (nơi chứa app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load scaler và mô hình với đường dẫn tuyệt đối
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(BASE_DIR, "stacking_model.pkl"))

# Giao diện Streamlit
st.title("🌱 Hệ thống đề xuất cây trồng bằng Stacking Classifier")
st.markdown("### Nhập các thông số đầu vào:")

n = st.slider("N - Nitơ", 0, 140, 90)
p = st.slider("P - Phốt pho", 0, 140, 42)
k = st.slider("K - Kali", 0, 200, 43)
ph = st.number_input("pH đất", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
temperature = st.number_input("Nhiệt độ (°C)", value=25.0)
humidity = st.number_input("Độ ẩm (%)", value=80.0)
rainfall = st.number_input("Lượng mưa (mm)", value=100.0)

if st.button("🌾 Dự đoán cây trồng phù hợp"):
    input_data = np.array([[n, p, k, ph, temperature, humidity, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"✅ Cây trồng được đề xuất: **{prediction[0]}**")
