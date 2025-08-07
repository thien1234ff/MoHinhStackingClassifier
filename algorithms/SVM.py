import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Đọc dữ liệu
df = pd.read_csv('data/Agriculture_dataset.csv', usecols=lambda col: "Unnamed" not in col)
Y = df["label"]
X = df[["N", "P", "K", "ph", "temperature", "humidity", "rainfall"]]

# Khởi tạo KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Khởi tạo SVM với kernel tuyến tính (có thể đổi sang 'rbf' nếu cần)
classifier = SVC(kernel='linear', random_state=0)

# Các biến lưu trữ kết quả
f1 = []
accur = []
precisions = []
recalls = []

# Ma trận nhầm lẫn trung bình
average_cm = np.zeros((len(Y.unique()), len(Y.unique())))

# Vòng lặp cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Chuẩn hóa dữ liệu
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)
    
    # Huấn luyện và dự đoán
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    
    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    average_cm += cm
    
    # Đánh giá
    f1.append(f1_score(y_test, y_pred, average='weighted'))
    accur.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='macro'))
    recalls.append(recall_score(y_test, y_pred, average='macro'))

# Tính ma trận nhầm lẫn trung bình
average_cm /= kf.get_n_splits()

# Hiển thị ma trận nhầm lẫn trung bình
labels = sorted(Y.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=average_cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='.0f')
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Average Confusion Matrix (SVM)")
plt.xticks(rotation=45, ha='right')
plt.show()

# In các chỉ số trung bình
print("Average F1-score: ", np.mean(f1))
print("Average Accuracy: ", np.mean(accur))
print("Average Precision: ", np.mean(precisions))
print("Average Recall: ", np.mean(recalls))
