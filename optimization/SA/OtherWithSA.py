import numpy as np
import pandas as pd
import random
import math
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Load dataset
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]
X_scaled = StandardScaler().fit_transform(X)

# ---------- Simulated Annealing ----------
def simulated_annealing(model_name, init_params, eval_fn, bounds, max_steps=50, temp=1.0, cooling=0.9):
    current = init_params.copy()
    best = current.copy()
    current_score = eval_fn(current)
    best_score = current_score

    for step in range(max_steps):
        # Tạo biến thể mới
        new = current.copy()
        for key in bounds:
            low, high = bounds[key]
            if isinstance(current[key], int):
                new[key] = int(np.clip(current[key] + random.randint(-5, 5), low, high))
            else:
                new[key] = float(np.clip(current[key] * np.exp(random.uniform(-0.3, 0.3)), low, high))
        
        new_score = eval_fn(new)
        delta = new_score - current_score
        if delta > 0 or math.exp(delta / temp) > random.random():
            current = new.copy()
            current_score = new_score
            if new_score > best_score:
                best = new.copy()
                best_score = new_score

        temp *= cooling
        print(f"{model_name} - Step {step+1}: Params = {current}, Accuracy = {current_score:.4f}")

    print(f"🔹 Best {model_name} → Params = {best}, Accuracy = {best_score:.4f}\n")
    return best, best_score

# ---------- Evaluate functions ----------
def evaluate_svm(params):
    model = SVC(C=params['C'], gamma=params['gamma'], kernel='rbf', random_state=42)
    return cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()

def evaluate_rf(params):
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        random_state=42)
    return cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()

def evaluate_nb(params):
    model = GaussianNB(var_smoothing=params['var_smoothing'])
    return cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy').mean()

# ---------- Run SA ----------
# SVM
simulated_annealing(
    model_name="SVM",
    init_params={'C': 1.0, 'gamma': 0.1},
    eval_fn=evaluate_svm,
    bounds={'C': (1e-3, 1e3), 'gamma': (1e-4, 1.0)}
)

# Random Forest
simulated_annealing(
    model_name="RandomForest",
    init_params={'n_estimators': 100, 'max_depth': 10},
    eval_fn=evaluate_rf,
    bounds={'n_estimators': (10, 200), 'max_depth': (3, 50)}
)

# Naive Bayes
simulated_annealing(
    model_name="NaiveBayes",
    init_params={'var_smoothing': 1e-9},
    eval_fn=evaluate_nb,
    bounds={'var_smoothing': (1e-11, 1e-7)}
)
