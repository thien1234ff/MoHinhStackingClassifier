import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Feature Scaling (fit outside cross-validation to avoid data leakage)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define safe param_grid (avoid invalid combinations)
param_grid = [
    {
        'penalty': ['l2', 'none'],
        'C': np.logspace(-4, 4, 5),
        'solver': ['lbfgs', 'sag', 'newton-cg'],
        'max_iter': [1000]
    },
    {
        'penalty': ['l1'],
        'C': np.logspace(-4, 4, 5),
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]
    },
    {
        'penalty': ['elasticnet'],
        'C': np.logspace(-4, 4, 5),
        'solver': ['saga'],
        'l1_ratio': [0.5],  # ElasticNet requires l1_ratio
        'max_iter': [1000]
    }
]

# Apply GridSearchCV
clf = GridSearchCV(LogisticRegression(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
clf.fit(X_scaled, Y)

# Get best model
best_model = clf.best_estimator_
print("Best parameters found:", clf.best_params_)
print("Best cross-validated accuracy:", clf.best_score_)
