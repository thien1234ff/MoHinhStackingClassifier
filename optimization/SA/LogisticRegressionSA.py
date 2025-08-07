import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import random
import math

# Load data
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluation function (cross-validated accuracy)
def evaluate(C, max_iter):
    model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', penalty='l2', random_state=42)
    scores = cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy')
    return scores.mean()

# Simulated Annealing
def simulated_annealing(init_C, init_iter, max_steps=100, temp=1.0, cooling=0.95):
    current_C = init_C
    current_iter = init_iter
    current_score = evaluate(current_C, current_iter)
    best_params = (current_C, current_iter)
    best_score = current_score

    for step in range(max_steps):
        # Propose new neighbor (random small change)
        new_C = current_C * np.exp(np.random.uniform(-0.3, 0.3))  # multiplicative perturbation
        new_iter = int(current_iter + np.random.randint(-200, 201))
        new_iter = max(100, min(new_iter, 3000))  # Keep within bounds

        new_score = evaluate(new_C, new_iter)

        # Acceptance probability
        delta = new_score - current_score
        if delta > 0 or math.exp(delta / temp) > random.random():
            current_C = new_C
            current_iter = new_iter
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_params = (new_C, new_iter)

        # Update temperature
        temp *= cooling

        print(f"Step {step+1}: C={current_C:.5f}, iter={current_iter}, score={current_score:.4f}")

    return best_params, best_score

# Khởi tạo
best_params, best_score = simulated_annealing(init_C=1.0, init_iter=1000)
print("\nBest parameters:")
print(f"C = {best_params[0]:.5f}")
print(f"max_iter = {best_params[1]}")
print(f"Best cross-validated accuracy = {best_score:.4f}")
