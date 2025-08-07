import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import random
import math

# Load data
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluation function (cross-validated accuracy)
def evaluate(max_depth, min_samples_split, min_samples_leaf):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, random_state=42)
    scores = cross_val_score(model, X_scaled, Y, cv=5, scoring='accuracy')
    return scores.mean()

# Simulated Annealing function
def simulated_annealing(init_max_depth, init_min_samples_split, init_min_samples_leaf, max_steps=100, temp=1.0, cooling=0.95):
    current_max_depth = init_max_depth
    current_min_samples_split = init_min_samples_split
    current_min_samples_leaf = init_min_samples_leaf
    current_score = evaluate(current_max_depth, current_min_samples_split, current_min_samples_leaf)
    best_params = (current_max_depth, current_min_samples_split, current_min_samples_leaf)
    best_score = current_score

    for step in range(max_steps):
        # Propose new neighbor (random small change)
        new_max_depth = int(current_max_depth + np.random.randint(-2, 3))
        new_min_samples_split = max(2, current_min_samples_split + np.random.randint(-1, 2))  # Avoid <=1
        new_min_samples_leaf = max(1, current_min_samples_leaf + np.random.randint(-1, 2))  # Avoid <=0

        new_score = evaluate(new_max_depth, new_min_samples_split, new_min_samples_leaf)

        # Acceptance probability
        delta = new_score - current_score
        if delta > 0 or math.exp(delta / temp) > random.random():
            current_max_depth = new_max_depth
            current_min_samples_split = new_min_samples_split
            current_min_samples_leaf = new_min_samples_leaf
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_params = (new_max_depth, new_min_samples_split, new_min_samples_leaf)

        # Update temperature
        temp *= cooling

        print(f"Step {step+1}: max_depth={current_max_depth}, min_samples_split={current_min_samples_split}, "
              f"min_samples_leaf={current_min_samples_leaf}, score={current_score:.4f}")

    return best_params, best_score

# Initialize and run Simulated Annealing
best_params, best_score = simulated_annealing(init_max_depth=5, init_min_samples_split=2, init_min_samples_leaf=1)

# Output the best parameters and score
print("\nBest parameters:")
print(f"max_depth = {best_params[0]}")
print(f"min_samples_split = {best_params[1]}")
print(f"min_samples_leaf = {best_params[2]}")
print(f"Best cross-validated accuracy = {best_score:.4f}")
