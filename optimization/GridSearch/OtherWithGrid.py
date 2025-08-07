import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 1. Random Forest - Grid Search
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

print("Best Random Forest parameters found:", rf_grid_search.best_params_)
print("Best Random Forest cross-validated accuracy:", rf_grid_search.best_score_)

# Train and evaluate the best Random Forest model
rf_best_model = rf_grid_search.best_estimator_
y_pred_rf = rf_best_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Test accuracy:", rf_accuracy)

# 2. Naive Bayes - Grid Search
nb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}
nb_grid_search = GridSearchCV(GaussianNB(), nb_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
nb_grid_search.fit(X_train, y_train)

print("Best Naive Bayes parameters found:", nb_grid_search.best_params_)
print("Best Naive Bayes cross-validated accuracy:", nb_grid_search.best_score_)

# Train and evaluate the best Naive Bayes model
nb_best_model = nb_grid_search.best_estimator_
y_pred_nb = nb_best_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Test accuracy:", nb_accuracy)

# 3. SVM - Grid Search
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_grid_search = GridSearchCV(SVC(random_state=42), svm_param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
svm_grid_search.fit(X_train, y_train)

print("Best SVM parameters found:", svm_grid_search.best_params_)
print("Best SVM cross-validated accuracy:", svm_grid_search.best_score_)

# Train and evaluate the best SVM model
svm_best_model = svm_grid_search.best_estimator_
y_pred_svm = svm_best_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM Test accuracy:", svm_accuracy)
