"""
📄 evaluate_microexpression_model.py
────────────────────────────────────────────
Evaluate a saved micro-expression model (.keras)
and generate a classification report and confusion matrix.

Usage:
    Place this file inside the `microexpression/` directory.
    Run with:
        python evaluate_microexpression_model.py

Requirements:
    - numpy, seaborn, matplotlib, scikit-learn, tensorflow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from datetime import datetime as dt

# === Configuration ===
MODEL_PATH = "microexpression_model_20250622_2034.keras"  # Update to test other models
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'microexpression_processed'))
X_PATH = os.path.join(DATASET_DIR, 'X_sequences.npy')
Y_PATH = os.path.join(DATASET_DIR, 'y_labels.npy')
CM_DIR = os.path.join(os.path.dirname(__file__), 'confusion-matrix')
os.makedirs(CM_DIR, exist_ok=True)

# === Load Data ===
print("📥 Loading data...")
X = np.load(X_PATH)
y = np.load(Y_PATH)
y_cat = to_categorical(y, num_classes=3)

# === Load Model ===
print(f"📦 Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# === Predict ===
print("🔍 Running predictions...")
y_pred = model.predict(X, verbose=1)
y_true = np.argmax(y_cat, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

# === Classification Report ===
print("\n📊 Classification Report:")
class_names = ['positive', 'negative', 'surprise']
report = classification_report(y_true, y_pred_class, target_names=class_names)
print(report)

# === Confusion Matrix ===
def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    timestamp = dt.now().strftime("%Y%m%d_%H%M")
    fig_path = os.path.join(output_path, f"confusion_matrix_eval_{timestamp}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"🔲 Confusion matrix saved to: {fig_path}")

plot_confusion_matrix(y_true, y_pred_class, class_names, CM_DIR)
print("✅ Evaluation complete.")
