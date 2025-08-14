import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load data
csv_path = 'Cyber-RF_Anomaly_Detector_Challenge_Dataset_TrainingSet_80.csv'
df = pd.read_csv(csv_path)

# Normalize last column: 0 = normal, 1 = anomalous
last_col = df.columns[-1]
df[last_col] = df[last_col].apply(lambda x: 0 if str(x).lower() == 'normal' or str(x) == '0' else 1)

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
FP = cm[0][1]
TN = cm[0][0]
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'False Positive Rate (FPR): {fpr:.4f}')
