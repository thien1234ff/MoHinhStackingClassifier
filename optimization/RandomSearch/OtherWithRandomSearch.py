import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Load and preprocess data
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 1. Random Forest - Random Search
rf_param_dist = {
    'n_estimators': np.arange(50, 201, 10),
    'max_depth': [None] + list(np.arange(5, 30, 5)),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
rf_random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rf_random_search.fit(X_train, y_train)
print("Best Random Forest params:", rf_random_search.best_params_)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_random_search.best_estimator_.predict(X_test)))

# 2. Naive Bayes - Random Search
nb_param_dist = {
    'var_smoothing': np.logspace(-12, -6, 100)
}
nb_random_search = RandomizedSearchCV(
    GaussianNB(),
    nb_param_dist,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)
nb_random_search.fit(X_train, y_train)
print("Best Naive Bayes params:", nb_random_search.best_params_)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_random_search.best_estimator_.predict(X_test)))

# 3. SVM - Random Search
svm_param_dist = {
    'C': np.logspace(-2, 2, 20),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}
svm_random_search = RandomizedSearchCV(
    SVC(random_state=42),
    svm_param_dist,
    n_iter=15,
    cv=5,
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)
svm_random_search.fit(X_train, y_train)
print("Best SVM params:", svm_random_search.best_params_)
print("SVM Accuracy:", accuracy_score(y_test, svm_random_search.best_estimator_.predict(X_test)))
