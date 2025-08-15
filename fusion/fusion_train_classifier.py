"""
fusion_train_classifier.py (Updated to train with all micro-expression probability scores)
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

dataset_path = "fusion_training_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ {dataset_path} not found! Please run fusion_dataset_builder.py first.")

print(f"📄 Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

# Features: deep_score + all micro-expression probabilities
micro_prob_cols = [col for col in df.columns if col.startswith("micro_score_")]
feature_cols = ["deep_score"] + micro_prob_cols

X = df[feature_cols].values
y = (df['fusion_label'] == 'LIVE').astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("🖥 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_s, y_train)

print("🖥 Training XGBoost...")
xgb_clf = xgb.XGBClassifier(
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train_s, y_train)

rf_pred = rf.predict(X_test_s)
xgb_pred = xgb_clf.predict(X_test_s)
vote_pred = np.round((rf_pred + xgb_pred) / 2).astype(int)

print("\n📊 RF Report:\n", classification_report(y_test, rf_pred, target_names=["SPOOF", "LIVE"]))
print("\n📊 XGB Report:\n", classification_report(y_test, xgb_pred, target_names=["SPOOF", "LIVE"]))
print("\n📊 Voting Report:\n", classification_report(y_test, vote_pred, target_names=["SPOOF", "LIVE"]))

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
joblib_dir = os.path.join("fusion", "joblib")
plots_dir = os.path.join("fusion", "plots")
os.makedirs(joblib_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

joblib.dump(rf, os.path.join(joblib_dir, f"fusion_rf_model_{timestamp}.joblib"))
joblib.dump(xgb_clf, os.path.join(joblib_dir, f"fusion_xgb_model_{timestamp}.joblib"))
joblib.dump(scaler, os.path.join(joblib_dir, f"fusion_scaler_{timestamp}.joblib"))

print(f"\n✅ Models and scaler saved with timestamp: {timestamp}")

cm = confusion_matrix(y_test, vote_pred)
labels = ["SPOOF", "LIVE"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Fusion Voting Confusion Matrix ({timestamp})")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(plots_dir, f"fusion_voting_confusion_{timestamp}.png")
plt.savefig(cm_path)
plt.close()

print(f"📈 Confusion matrix plot saved to: {cm_path}")
