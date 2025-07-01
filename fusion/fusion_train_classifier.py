import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# === Load Fusion Training Dataset ===
csv_path = "fusion_training_dataset.csv"
print(f"📄 Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# === Features and Labels ===
X = df[['deep_score', 'micro_score']].values
y = (df['fusion_label'] == 'LIVE').astype(int).values  # 1 = LIVE, 0 = SPOOF

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Standardize Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Classifier ===
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

# === Evaluate Model ===
y_pred = clf.predict(X_test_scaled)
report = classification_report(y_test, y_pred, target_names=["SPOOF", "LIVE"])
print("\n📊 Classification Report:\n", report)

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
labels = ["SPOOF", "LIVE"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Fusion Classifier - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# === Save Confusion Matrix ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
plot_dir = os.path.join("fusion", "plots")
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, f"fusion_confusion_matrix_{timestamp}.png")
plt.savefig(plot_path)
print(f"📊 Confusion matrix saved to: {plot_path}")

# === Save Model and Scaler ===
import joblib
model_path = os.path.join("fusion", f"fusion_classifier_model_{timestamp}.joblib")
scaler_path = os.path.join("fusion", f"fusion_scaler_{timestamp}.joblib")
joblib.dump(clf, model_path)
joblib.dump(scaler, scaler_path)
print(f"✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
