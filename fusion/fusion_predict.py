"""
🔗 fusion_predict.py
────────────────────────────────────────────
Combines predictions from:
1. Deepfake Detection Model
2. Micro-Expression Recognition Model

Applies rule-based late fusion logic to return a final
binary liveness decision: LIVE ✅ or SPOOF ❌

Assumes:
- Models already trained and stored as:
    mobilenet_deepfake_model_finetuned.keras
    microexpression_model_20250622_2034.keras
- Input: Preprocessed numpy arrays for each model

Note: This version expects demo inputs to be
saved as .npy arrays: X_deepfake.npy and X_microexpression.npy
"""

import os
import numpy as np
from tensorflow.keras.models import load_model

# === Load Models ===
print("📦 Loading models...")
# Load models from subfolders
# Assuming structure: fusion/, deepfake/model/, microexpression/models/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

deepfake_model_path = os.path.join(project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_fine_tuned.keras')
microexp_model_path = os.path.join(project_root, 'microexpression', 'models', 'microexpression_model_20250622_2034.keras')

deepfake_model = load_model(deepfake_model_path)
microexp_model = load_model(microexp_model_path)

# === Load Inputs ===
print("📥 Loading input arrays...")
X_deep = np.load("X_deepfake.npy")           # Shape: (1, 224, 224, 3)
X_micro = np.load("X_microexpression.npy")  # Shape: (1, 16, 224, 224, 3)

# === Get Predictions ===
deep_pred = deepfake_model.predict(X_deep)[0]          # e.g., [0.1, 0.9]
micro_pred = microexp_model.predict(X_micro)[0]        # e.g., [0.3, 0.5, 0.2]

# === Interpret Predictions ===
deep_class = np.argmax(deep_pred)  # 0 = fake, 1 = real
micro_class = np.argmax(micro_pred)  # 0 = positive, 1 = negative, 2 = surprise

# === Rule-Based Fusion Logic ===
def fusion_decision(deep, micro):
    if deep == 1 and micro in [0, 1, 2]:
        return "✅ LIVE (Real + Valid Emotion)"
    elif deep == 0 and micro in [0, 1, 2]:
        return "❌ SPOOF (Fake + Emotion)"
    elif deep == 1 and micro not in [0, 1, 2]:
        return "⚠️ Inconclusive (Real + Unknown Emotion)"
    else:
        return "❌ SPOOF"


# === Final Decision ===
final_result = fusion_decision(deep_class, micro_class)

# === Output Results ===
print("\n🔍 Deepfake Prediction:", deep_pred, "> Class:", "Real" if deep_class == 1 else "Fake")
print("🔍 Micro-Expression Prediction:", micro_pred, "> Class:", ["Positive", "Negative", "Surprise"][micro_class])
print("\n🎯 Final Liveness Decision:", final_result)
