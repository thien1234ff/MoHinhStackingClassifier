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
import sys
import joblib


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

# ... các import và phần chuẩn bị dữ liệu giữ nguyên ...

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
    
    # ==== LẤY META-FEATURES ====
    import pandas as pd
    from sklearn.model_selection import cross_val_predict

    meta_features = []
    meta_feature_names = [name for name, _ in base_models]

    for name, model in base_models:
        preds = cross_val_predict(model, X_train_scaled, y_train, cv=5, method='predict_proba')
        # Lấy xác suất của lớp đầu tiên (hoặc điều chỉnh tùy số lớp)
        meta_features.append(preds[:, 1])

    # Chuyển sang DataFrame
    df_meta = pd.DataFrame(np.array(meta_features).T, columns=meta_feature_names)
    print("\nVí dụ meta-features từ fold này:")
    print(df_meta.head())
    # ===========================
    
    # Predict
    y_pred = stacking_model.predict(X_test_scaled)
    
    # Calculate metrics
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='macro'))
    recall_scores.append(recall_score(y_test, y_pred, average='macro'))

# Print average performance metrics
print("\nStacking Classifier Performance:")
print(f"Average F1-score: {np.mean(f1_scores):.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")

# Sau khi đánh giá xong bằng KFold, huấn luyện lại trên toàn bộ dữ liệu
sc_X_full = StandardScaler()
X_scaled_full = sc_X_full.fit_transform(X)
stacking_model.fit(X_scaled_full, Y)

# Lưu cả model và scaler
joblib.dump(stacking_model, "stacking_model.pkl")
joblib.dump(sc_X_full, "scaler.pkl")
