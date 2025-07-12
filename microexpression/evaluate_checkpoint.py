"""
🔍 Micro-Expression Model Checkpoint Evaluation
───────────────────────────────────────────────────────────────────────────────

This script evaluates a specific model checkpoint saved during the training process
to assess its performance on the validation set. It's useful for verifying the 
quality of intermediate models and comparing different training checkpoints.

📋 FUNCTIONALITY:
- Loads the preprocessed sequence data used for training
- Recreates the exact same validation split used during training
- Loads a specific model checkpoint from the checkpoints directory
- Generates predictions on the validation set
- Evaluates model performance with a detailed classification report
- Displays precision, recall, and F1-score for each emotion class

🧠 MODEL EVALUATION:
- Uses the same validation data split as the training script
- Maintains consistent evaluation methodology for fair comparison
- Focuses on classification metrics rather than loss values
- Provides class-specific performance metrics

📊 INPUT/OUTPUT:
- Input:
  - X_sequences.npy and y_labels.npy from dataset/microexpression_processed/
  - Saved model checkpoint from checkpoints/best_model_*.keras
- Output:
  - Console: Detailed classification report with precision, recall, F1-score
  - No files are saved by this script

🔍 USAGE:
- Update the checkpoint path to the specific model you want to evaluate
- Run script: python evaluate_checkpoint.py
- Review the classification report in console output
- Compare results with other checkpoints or the final model

📝 NOTES:
- The script uses a fixed random seed (42) to ensure the same validation split
- The checkpoint filename contains the epoch number and validation loss
- This evaluation is useful for identifying the best checkpoint for deployment
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# === Paths ===
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'microexpression_processed'))
X_PATH = os.path.join(DATASET_DIR, 'X_sequences.npy')
Y_PATH = os.path.join(DATASET_DIR, 'y_labels.npy')

# === Load Data ===
X = np.load(X_PATH)
y = np.load(Y_PATH)

# === Recreate the same validation split used earlier ===
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

y_cat = to_categorical(y, num_classes=3)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# === Load Checkpoint Model ===
model = load_model("checkpoints/best_model_24_0.9389.keras")
print("✅ Loaded model from checkpoint.")

# === Evaluate ===
y_pred = model.predict(X_val)
y_true = np.argmax(y_val, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

# === Print Classification Report ===
print("\n📊 Classification Report (Checkpoint Model):")
print(classification_report(
    y_true, y_pred_class,
    target_names=['positive', 'negative', 'surprise']
))
