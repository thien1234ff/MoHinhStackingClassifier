import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph","temperature","humidity","rainfall"]]

# Splitting the dataset into the Training set and Test set 
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize Logistic Regression classifier
classifier = LogisticRegression(max_iter=200, random_state=0)

# Variables to store evaluation metrics
f1 = []
accur = []
precisions = []
recalls = []

# Initialize confusion matrix accumulator
average_cm = np.zeros((len(Y.unique()), len(Y.unique())))

# Loop through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Feature Scaling
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
    
    # Fit classifier and make predictions
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Add current confusion matrix to the accumulator
    average_cm += cm
    
    # Evaluate with F1-score
    f_error = f1_score(y_test, y_pred, average='weighted')
    f1.append(f_error)
    
    # Evaluate with accuracy
    acc = accuracy_score(y_test, y_pred)
    accur.append(acc)
    
    # Evaluate with precision and recall
    pre = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    precisions.append(pre)
    recalls.append(rec)

# Calculate the average confusion matrix
average_cm /= kf.get_n_splits()

# Display the average confusion matrix
labels = sorted(Y.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=average_cm, display_labels=labels)

# Use '0.0f' to format the values as floating point numbers
disp.plot(cmap='Blues', values_format='.0f')  # or '.2f' if you want 2 decimal places
plt.xlabel("Predicted label")  # Cột: nhãn dự đoán
plt.ylabel("True label")       # Hàng: nhãn thật
plt.title("Average Confusion Matrix")
plt.xticks(rotation=45, ha='right')
plt.show()

# Print the average evaluation metrics
print("Average F1-score: ", np.mean(f1))
print("Average Accuracy: ", np.mean(accur))
print("Average Precision: ", np.mean(precisions))
print("Average Recall: ", np.mean(recalls))
