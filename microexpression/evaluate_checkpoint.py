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
