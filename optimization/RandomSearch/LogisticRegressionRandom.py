import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import uniform

# Load dataset
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the hyperparameter space for RandomizedSearchCV
param_dist = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': uniform(loc=0.0001, scale=4),  # Uniform distribution for C (log scale)
    'solver': ['liblinear', 'lbfgs', 'saga', 'newton-cg'],
    'max_iter': [1000, 2000, 3000],
    'l1_ratio': uniform(loc=0, scale=1),  # Only relevant for 'elasticnet'
}

# Apply RandomizedSearchCV
clf_random = RandomizedSearchCV(LogisticRegression(random_state=42), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', verbose=1, n_jobs=-1, random_state=42)
clf_random.fit(X_scaled, Y)

# Get the best model
best_model_random = clf_random.best_estimator_

# Print the best parameters and score
print("Best parameters found:", clf_random.best_params_)
print("Best cross-validated accuracy:", clf_random.best_score_)
