import pandas as pd
import numpy as np
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    # Create and train the model
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Return the negative accuracy because Optuna minimizes the objective function
    return 1 - accuracy  # Minimize the negative accuracy

# Create an Optuna study
study = optuna.create_study(direction='minimize')  # Minimize the loss function (1 - accuracy)
study.optimize(objective, n_trials=50)  # 50 trials

# Output the best parameters and score
print("Best parameters found:", study.best_params)
print("Best accuracy (1 - loss):", 1 - study.best_value)

# Train the final model with the best parameters
best_clf = DecisionTreeClassifier(
    criterion=study.best_params['criterion'],
    max_depth=study.best_params['max_depth'],
    min_samples_split=study.best_params['min_samples_split'],
    min_samples_leaf=study.best_params['min_samples_leaf'],
    random_state=42
)

best_clf.fit(X_train, y_train)

# Evaluate the final model
y_pred = best_clf.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final accuracy on test set: {final_accuracy}")
