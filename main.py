
import pandas as pd
import matplotlib.pyplot as plt
# 1. Load the Dataset
df = pd.read_csv("data.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
print(df.info())
# 2. Data Cleaning
# Remove unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'])
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Check for duplicate rows
print("\nDuplicates:", df.duplicated().sum())
# 3. Encode the Target Variable
# M = Malignant = 1
# B = Benign = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
print("\nDiagnosis Distribution:")
print(df['diagnosis'].value_counts())
# 4. Separate Features and Target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
# 5. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# 6. Feature Selection
# Feature selection is performed ONLY on
# the training data to avoid data leakage.
train_data = X_train.copy()
train_data['diagnosis'] = y_train
correlation = train_data.corr()['diagnosis'].drop('diagnosis')
top_features = (
    correlation.abs()
    .sort_values(ascending=False)
    .head(10)
)
selected_features = top_features.index
print("\nTop 10 Selected Features:")
print(top_features)
# Apply the selected features to both datasets
X_train = X_train[selected_features]
X_test = X_test[selected_features]
print("\nSelected Training Features:", X_train.shape)
print("Selected Testing Features:", X_test.shape)
# 7. Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\nScaling completed successfully.")
# 8. Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nLogistic Regression completed successfully.")

#  (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN completed successfully.")

# 10. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree completed successfully.")

# 11. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest completed successfully.")

# 12. Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------- Logistic Regression ----------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------- KNN ----------
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
print("\n--- KNN ---")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1-Score:", f1_knn)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# ---------- Decision Tree ----------
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-Score:", f1_dt)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# ---------- Random Forest ----------
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-Score:", f1_rf)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# 13. Final Results Comparison
results = pd.DataFrame({
    'Algorithm': [
        'Logistic Regression',
        'KNN',
        'Decision Tree',
        'Random Forest'
    ],
    'Accuracy': [
        accuracy,
        accuracy_knn,
        accuracy_dt,
        accuracy_rf
    ],
    'Precision': [
        precision,
        precision_knn,
        precision_dt,
        precision_rf
    ],
    'Recall': [
        recall,
        recall_knn,
        recall_dt,
        recall_rf
    ],
    'F1-Score': [
        f1,
        f1_knn,
        f1_dt,
        f1_rf
    ]
})

print("\n========== FINAL RESULTS ==========")
print(results)

# 14. Accuracy Comparison Visualization
models = [
    'Logistic Regression',
    'KNN',
    'Decision Tree',
    'Random Forest'
]

accuracies = [
    accuracy,
    accuracy_knn,
    accuracy_dt,
    accuracy_rf
]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

# 15. Confusion Matrix - Best Model
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred
)
plt.title("Logistic Regression - Confusion Matrix")

plt.show()

