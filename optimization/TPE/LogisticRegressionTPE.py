import optuna
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load dữ liệu
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hàm tối ưu hóa
def objective(trial):
    # Chọn tham số
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga", "newton-cg"])
    C = trial.suggest_loguniform("C", 1e-4, 1e4)
    max_iter = trial.suggest_int("max_iter", 100, 3000)

    # Điều kiện cần thiết cho elasticnet
    l1_ratio = None
    if penalty == "elasticnet":
        if solver != "saga":
            raise optuna.exceptions.TrialPruned()
        l1_ratio = trial.suggest_uniform("l1_ratio", 0.0, 1.0)

    # Tránh tổ hợp không hợp lệ
    if penalty in ["l1"] and solver not in ["liblinear", "saga"]:
        raise optuna.exceptions.TrialPruned()
    if penalty in ["none", "l2"] and solver == "liblinear":
        raise optuna.exceptions.TrialPruned()

    # Tạo model
    model = LogisticRegression(
        random_state=42
    )

    # Tính điểm trung bình với cross-validation
    score = cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()
    return score

# Tạo study và tối ưu
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100, timeout=300)  # chạy 100 lần hoặc 5 phút

# In kết quả tốt nhất
print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)
