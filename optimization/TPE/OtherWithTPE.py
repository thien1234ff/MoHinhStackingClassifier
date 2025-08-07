import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Load data
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SVM optimization
def objective_svm(trial):
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    return cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()

# Random Forest optimization
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42)
    return cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()

# Naive Bayes (doesn't require tuning but we include dummy search)
def objective_nb(trial):
    var_smoothing = trial.suggest_loguniform("var_smoothing", 1e-11, 1e-7)
    model = GaussianNB(var_smoothing=var_smoothing)
    return cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()

# Run studies
study_svm = optuna.create_study(direction="maximize")
study_svm.optimize(objective_svm, n_trials=30)
print("Best SVM:", study_svm.best_params, "→ Accuracy:", study_svm.best_value)

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=30)
print("Best Random Forest:", study_rf.best_params, "→ Accuracy:", study_rf.best_value)

study_nb = optuna.create_study(direction="maximize")
study_nb.optimize(objective_nb, n_trials=30)
print("Best Naive Bayes:", study_nb.best_params, "→ Accuracy:", study_nb.best_value)
