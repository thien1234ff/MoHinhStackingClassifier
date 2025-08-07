import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
import lime
import lime.lime_tabular
import webbrowser
import sys
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Thiết lập encoding cho stdout
sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Define base models
base_models = [
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)),
    ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ('nb', GaussianNB()),
    ('rf', RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)),
    ('svm', SVC(kernel='linear', random_state=42, probability=True))
]

# Define meta-learner
meta_learner = MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42)

# Create stacking classifier
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=10,
    stack_method='auto'
)

# Stratified KFold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
f1_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []

# Lưu X_test_scaled, X_test (chưa chuẩn hóa), và y_test để sử dụng sau
X_test_scaled_all = []
X_test_all = []
y_test_all = []

# Cross-validation loop
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Feature Scaling using StandardScaler
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
    
    # Lưu dữ liệu test để sử dụng sau
    X_test_scaled_all.append(X_test_scaled)
    X_test_all.append(X_test)
    y_test_all.append(y_test)
    
    # Fit stacking model
    stacking_model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = stacking_model.predict(X_test_scaled)
    
    # Calculate metrics
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))

# Print average performance metrics
print("Stacking Classifier Performance:")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")

# Feature importance (dựa trên RandomForestClassifier trong base models)
rf_model = [model for name, model in base_models if name == 'rf'][0]
rf_model.fit(X_train_scaled, y_train)  # Fit lại RandomForest để lấy feature importance
feature_names = ["Đạm (N)", "Lân (P)", "Kali (K)", "pH", "Nhiệt độ", "Độ ẩm", "Lượng mưa"]
feature_importance = rf_model.feature_importances_
feature_importance_dict = dict(zip(feature_names, feature_importance))

# Tạo biểu đồ tầm quan trọng đặc trưng
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance, color='#4CAF50')
plt.title("Tầm quan trọng của đặc trưng (RandomForest)")
plt.xlabel("Đặc trưng")
plt.ylabel("Tầm quan trọng")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight')
feature_importance_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Tạo biểu đồ hiệu suất mô hình
metrics = ['F1-score', 'Accuracy', 'Precision', 'Recall']
values = [np.mean(f1_scores), np.mean(accuracy_scores), np.mean(precision_scores), np.mean(recall_scores)]
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='#2196F3')
# Đổi tiêu đề để tránh lặp lại với section HTML
plt.title("Biểu đồ hiệu suất", fontsize=14, pad=20)
plt.ylabel("Giá trị", fontsize=12)
plt.ylim(0, 1)
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)
plt.tight_layout()
# Thêm khoảng cách phía trên biểu đồ
plt.subplots_adjust(top=0.85)  # Giảm giá trị top để đẩy biểu đồ xuống
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight')
performance_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Tạo LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=np.unique(y_train).tolist(),
    mode='classification'
)

# Giải thích một mẫu
i = 0
X_test_scaled = X_test_scaled_all[-1]  # Lấy dữ liệu test từ fold cuối
X_test = X_test_all[-1]  # Lấy dữ liệu gốc (chưa chuẩn hóa)
y_test = y_test_all[-1]
lime_exp = lime_explainer.explain_instance(
    data_row=X_test_scaled[i],
    predict_fn=stacking_model.predict_proba,
    num_features=7
)
predicted_crop = stacking_model.predict(X_test_scaled[i:i+1])[0]
true_crop = y_test.iloc[i]

# Lấy giá trị đặc trưng gốc của mẫu
sample_features = X_test.iloc[i]
sample_features_text = "\n".join([f"- {name}: {value:.4f}" for name, value in zip(feature_names, sample_features)])

# Lưu biểu đồ LIME dưới dạng base64
lime_exp.as_pyplot_figure()
plt.title(f"Giải thích LIME cho mẫu {i} (Dự đoán: {predicted_crop})")
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png', bbox_inches='tight')
lime_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
plt.close()

# Tạo văn bản giải thích chi tiết bằng tiếng Việt
explanation_text = f"""
- **Cây trồng thực tế**: {true_crop}
- **Cây trồng dự đoán**: {predicted_crop}
- **Dự đoán đúng**: {'Có' if true_crop == predicted_crop else 'Không'}

**Giá trị đặc trưng của mẫu (trước chuẩn hóa):**
{sample_features_text}

**Tầm quan trọng của đặc trưng (Toàn cục, dựa trên RandomForest trong stacking):**
{chr(10).join([f'- {feature}: {importance:.4f}' for feature, importance in feature_importance_dict.items()])}

**Giải thích cục bộ từ LIME:**
Mô hình dự đoán '{predicted_crop}' dựa trên các yếu tố chính sau:
{chr(10).join([f'- {exp[0]}: {exp[1]:.4f}' for exp in lime_exp.as_list()])}
"""

# Define output_path before using it in the HTML
output_path = f'giaithich_du_doan_mau_{i}_stacking_custom.html'

# Tạo HTML đẹp và chi tiết
custom_html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Giải thích dự đoán cây trồng - Mẫu {i}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        html {{ scroll-behavior: smooth; }}
        .sidebar {{
            position: sticky;
            top: 20px;
        }}
        .card {{
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }}
    </style>
</head>
<body class="bg-gray-100 font-sans">
    <!-- Header -->
    <header class="bg-gradient-to-r from-green-500 to-green-700 text-white py-6">
        <div class="container mx-auto px-4 text-center">
            <h1 class="text-3xl md:text-4xl font-bold">Giải thích dự đoán cây trồng - Mẫu {i}</h1>
            <p class="mt-2 text-lg">Mô hình Stacking Classifier với giải thích LIME</p>
        </div>
    </header>

    <!-- Navigation -->
    <nav class="bg-white shadow-md py-4">
        <div class="container mx-auto px-4">
            <ul class="flex flex-wrap gap-4 justify-center">
                <li><a href="#overview" class="text-green-600 hover:underline">Tổng quan</a></li>
                <li><a href="#performance" class="text-green-600 hover:underline">Hiệu suất mô hình</a></li>
                <li><a href="#explanation" class="text-green-600 hover:underline">Giải thích dự đoán</a></li>
                <li><a href="#features" class="text-green-600 hover:underline">Tầm quan trọng đặc trưng</a></li>
                <li><a href="#lime" class="text-green-600 hover:underline">Giải thích LIME</a></li>
            </ul>
        </div>
    </nav>

    <div class="container mx-auto px-4 py-8 flex flex-col md:flex-row gap-8">
        <!-- Sidebar -->
        <aside class="md:w-1/4 sidebar">
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h3 class="text-xl font-semibold mb-4">Hành động</h3>
                <a href="{output_path}" download class="block mb-2 bg-green-500 text-white text-center py-2 rounded hover:bg-green-600">
                    <i class="fas fa-download mr-2"></i>Tải xuống HTML
                </a>
                <a href="https://github.com" target="_blank" class="block bg-gray-500 text-white text-center py-2 rounded hover:bg-gray-600">
                    <i class="fas fa-code mr-2"></i>Xem mã nguồn
                </a>
            </div>
        </aside>

        <!-- Main Content -->
        <main class="md:w-3/4">
            <!-- Overview Section -->
            <section id="overview" class="mb-8 card bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-2xl font-semibold mb-4"><i class="fas fa-info-circle mr-2"></i>Tổng quan mô hình</h2>
                <p class="mb-4">Mô hình Stacking Classifier kết hợp các mô hình cơ sở sau với meta-learner là MLPClassifier:</p>
                <table class="w-full border-collapse mb-4">
                    <thead>
                        <tr class="bg-green-500 text-white">
                            <th class="p-3">Mô hình</th>
                            <th class="p-3">Mô tả</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td class="p-3 border">Decision Tree</td><td class="p-3 border">Cây quyết định với tiêu chí entropy, độ sâu tối đa 10.</td></tr>
                        <tr><td class="p-3 border">Logistic Regression</td><td class="p-3 border">Hồi quy logistic với 2000 lần lặp tối đa.</td></tr>
                        <tr><td class="p-3 border">Naive Bayes</td><td class="p-3 border">Gaussian Naive Bayes cho phân loại xác suất.</td></tr>
                        <tr><td class="p-3 border">Random Forest</td><td class="p-3 border">Rừng ngẫu nhiên với 100 cây, tiêu chí entropy.</td></tr>
                        <tr><td class="p-3 border">SVM</td><td class="p-3 border">Máy vector hỗ trợ với kernel tuyến tính.</td></tr>
                    </tbody>
                </table>
                <p>Tập dữ liệu bao gồm 7 đặc trưng: Đạm (N), Lân (P), Kali (K), pH, Nhiệt độ, Độ ẩm, Lượng mưa, được sử dụng để dự đoán loại cây trồng phù hợp.</p>
            </section>

            <!-- Performance Section -->
            <section id="performance" class="mb-8 card bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-2xl font-semibold mb-4"><i class="fas fa-chart-bar mr-2"></i>Hiệu suất mô hình</h2>
                <p class="mb-4">Hiệu suất trung bình qua 5-fold cross-validation:</p>
                <table class="w-full border-collapse mb-4">
                    <thead>
                        <tr class="bg-green-500 text-white">
                            <th class="p-3">Chỉ số</th>
                            <th class="p-3">Giá trị</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr><td class="p-3 border">F1-score</td><td class="p-3 border">{np.mean(f1_scores):.4f}</td></tr>
                        <tr><td class="p-3 border">Accuracy</td><td class="p-3 border">{np.mean(accuracy_scores):.4f}</td></tr>
                        <tr><td class="p-3 border">Precision</td><td class="p-3 border">{np.mean(precision_scores):.4f}</td></tr>
                        <tr><td class="p-3 border">Recall</td><td class="p-3 border">{np.mean(recall_scores):.4f}</td></tr>
                    </tbody>
                </table>
                <!-- Thêm padding phía trên biểu đồ để tạo khoảng cách -->
                <div class="pt-6">
                    <img src="data:image/png;base64,{performance_img}" alt="Performance Metrics" class="w-full rounded-lg">
                </div>
            </section>

            <!-- Explanation Section -->
            <section id="explanation" class="mb-8 card bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-2xl font-semibold mb-4"><i class="fas fa-seedling mr-2"></i>Giải thích dự đoán cho mẫu {i}</h2>
                <pre class="bg-gray-50 p-4 rounded">{explanation_text}</pre>
            </section>

            <!-- Feature Importance Section -->
            <section id="features" class="mb-8 card bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-2xl font-semibold mb-4"><i class="fas fa-chart-pie mr-2"></i>Tầm quan trọng đặc trưng toàn cục</h2>
                <p class="mb-4">Dựa trên RandomForestClassifier, các đặc trưng được xếp hạng theo mức độ ảnh hưởng đến dự đoán.</p>
                <img src="data:image/png;base64,{feature_importance_img}" alt="Feature Importance" class="w-full rounded-lg">
            </section>

            <!-- LIME Explanation Section -->
            <section id="lime" class="mb-8 card bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-2xl font-semibold mb-4"><i class="fas fa-search mr-2"></i>Giải thích cục bộ từ LIME</h2>
                <p class="mb-4">LIME (Local Interpretable Model-agnostic Explanations) giải thích lý do mô hình dự đoán '{predicted_crop}' cho mẫu này. Biểu đồ dưới đây cho thấy mức độ ảnh hưởng của từng đặc trưng đến dự đoán.</p>
                <img src="data:image/png;base64,{lime_img}" alt="LIME Explanation" class="w-full rounded-lg">
            </section>
        </main>
    </div>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white text-center py-4">
        <p>Được tạo bởi Grok 3, xAI - Ngày {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    </footer>
</body>
</html>
"""

# Lưu HTML tùy chỉnh
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(custom_html)
webbrowser.open(output_path)
