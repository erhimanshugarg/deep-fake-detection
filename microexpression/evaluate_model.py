"""
📊 Micro-Expression Model Evaluation Script
───────────────────────────────────────────────────────────────────────────────

This script performs a comprehensive evaluation of a trained micro-expression 
recognition model on the full dataset. It generates detailed performance metrics
and visualizations to assess model quality and identify potential weaknesses.

📋 FUNCTIONALITY:
- Loads a trained micro-expression model from the models directory
- Processes the complete dataset (not just validation split)
- Generates predictions for all sequences
- Creates a detailed classification report with precision, recall, and F1-score
- Visualizes model performance with a confusion matrix
- Saves the confusion matrix as an image for documentation

🧠 MODEL EVALUATION:
- Evaluates on the entire dataset to get comprehensive performance metrics
- Focuses on per-class performance to identify class-specific weaknesses
- Uses standard metrics (precision, recall, F1) for classification assessment
- Visualizes error patterns through confusion matrix heatmap

📊 INPUT/OUTPUT:
- Input:
  - Trained model: models/microexpression_model_*.keras
  - Dataset: X_sequences.npy and y_labels.npy from dataset/microexpression_processed/
- Output:
  - Console: Detailed classification report with precision, recall, F1-score
  - Image: Confusion matrix visualization saved to confusion-matrix/
  - File path: confusion-matrix/confusion_matrix_eval_<timestamp>.png

🔍 USAGE:
- Update MODEL_PATH variable to point to the specific model you want to evaluate
- Run script: python evaluate_model.py
- Review the classification report in console output
- Examine the confusion matrix image for visual performance assessment
- Use results to identify model strengths and weaknesses

📝 NOTES:
- Unlike evaluate_checkpoint.py, this script evaluates on the full dataset
- The confusion matrix helps identify which classes are commonly confused
- Timestamp in output filenames helps track multiple evaluation runs
- This evaluation is useful for final model assessment before deployment
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
MODEL_PATH = "models/microexpression_model_20250622_2034.keras"  # Update to test other models
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
