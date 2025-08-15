"""
fusion_predict.py (Final – All 4 Fusion Improvements)
----------------------------------------------------
✓ RF + XGB fusion model (replaces logistic regression)
✓ Weighted probability voting + confidence-based thresholding
✓ Rule-based overrides for extreme base-model scores
✓ SHAP explainability for feature contributions
"""

import os
import numpy as np
import torch
from torchvision import transforms
from tensorflow.keras.models import load_model
import joblib
import shap

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from microexpression.resnet18_au_model import ResNet18AU

# ===== CONFIG =====
WEIGHT_RF  = 0.98
WEIGHT_XGB = 0.96
DECISION_THRESHOLD = 0.5
TIER_HIGH = 0.85
TIER_MED  = 0.65
emotion_labels = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

# ===== PATHS =====
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fusion_dir   = os.path.join(project_root, 'fusion')
inputs_dir   = os.path.join(fusion_dir, 'inputs')
joblib_dir   = os.path.join(fusion_dir, 'joblib')

deep_input_path  = os.path.join(inputs_dir, 'X_deepfake.npy')
micro_input_path = os.path.join(inputs_dir, 'X_microexpression.npy')

deepfake_model_path = os.path.join(
    project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_fine_tuned.keras'
)
microexp_model_path = os.path.join(
    project_root, 'microexpression', 'models', 'resnet18_microexpr_v9_best.pth'
)

# ===== LOAD MODELS =====
print("📦 Loading Deepfake model...")
deepfake_model = load_model(deepfake_model_path)

print("📦 Loading Micro-expression model (PyTorch, 6-class)...")
micro_model = ResNet18AU(num_classes=len(emotion_labels), au_feature_dim=12)
micro_model.load_state_dict(torch.load(microexp_model_path, map_location='cpu'))
micro_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== LOAD FUSION CLASSIFIERS & SCALER =====
def latest_file(dir_path, prefix):
    files = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.joblib')]
    return os.path.join(dir_path, sorted(files)[-1]) if files else None

rf_model_path = latest_file(joblib_dir, "fusion_rf_model_")
xgb_model_path = latest_file(joblib_dir, "fusion_xgb_model_")
scaler_path    = latest_file(joblib_dir, "fusion_scaler_")

if not rf_model_path or not xgb_model_path or not scaler_path:
    raise FileNotFoundError("❌ Missing RF/XGB/scaler joblib files in fusion/joblib/")

rf_clf = joblib.load(rf_model_path)
xgb_clf = joblib.load(xgb_model_path)
scaler  = joblib.load(scaler_path)

print(f"📥 Loaded Fusion Models:\n RF: {os.path.basename(rf_model_path)}\n XGB: {os.path.basename(xgb_model_path)}\n Scaler: {os.path.basename(scaler_path)}")

# ===== LOAD INPUTS =====
if not os.path.exists(deep_input_path) or not os.path.exists(micro_input_path):
    raise FileNotFoundError("❌ Input .npy files not found. Run prepare_fusion_inputs.py first.")

X_deep  = np.load(deep_input_path)
X_micro = np.load(micro_input_path)

# ===== BASE MODEL PREDICTIONS =====
deep_pred_raw = deepfake_model.predict(X_deep, verbose=0)[0]
if len(deep_pred_raw) == 1:
    prob_real = float(deep_pred_raw[0])
    prob_fake = 1.0 - prob_real
    deep_score = prob_real
    deep_class = 1 if prob_real > 0.5 else 0
    deep_pred_probs = [prob_fake, prob_real]
else:
    deep_pred_probs = deep_pred_raw.tolist()
    deep_class = int(np.argmax(deep_pred_probs))
    deep_score = deep_pred_probs[1]
deep_class_str = "Real" if deep_class == 1 else "Fake"

frames_tensor = torch.stack([
    transform(X_micro[0, i]) for i in range(X_micro.shape[1])
])
seq_tensor = frames_tensor.unsqueeze(0)

with torch.no_grad():
    micro_logits = micro_model(seq_tensor)
    micro_probs  = torch.softmax(micro_logits, dim=1)[0].cpu().numpy()
micro_class = int(np.argmax(micro_probs))
micro_class_str = emotion_labels[micro_class]

# ===== RULE-BASED OVERRIDES =====
override_label = None
override_reason = None

if deep_pred_probs[0] >= 0.99:
    override_label = "SPOOF"
    override_reason = "Deepfake model extremely confident (≥99% Fake)"
elif all(p < 0.20 for p in micro_probs):
    override_label = "INCONCLUSIVE"
    override_reason = "Low micro-expression confidence (<20% all classes)"
elif deep_pred_probs[1] >= 0.99 and max(micro_probs) >= 0.80:
    override_label = "LIVE"
    override_reason = "High deepfake Real score & strong emotion confidence"

# ===== ML FUSION =====
features = np.array([deep_score] + micro_probs.tolist()).reshape(1, -1)
features_scaled = scaler.transform(features)

rf_proba  = rf_clf.predict_proba(features_scaled)[0]
xgb_proba = xgb_clf.predict_proba(features_scaled)[0]

weighted_proba = (
    rf_proba * WEIGHT_RF +
    xgb_proba * WEIGHT_XGB
) / (WEIGHT_RF + WEIGHT_XGB)

final_pred = 1 if weighted_proba[1] >= DECISION_THRESHOLD else 0
fusion_label_ml = "LIVE" if final_pred == 1 else "SPOOF"

# ===== FINAL DECISION =====
if override_label:
    fusion_label = override_label
    fusion_conf  = None
    conf_tier    = "N/A"
else:
    fusion_label = fusion_label_ml
    conf_score = weighted_proba[final_pred]
    if conf_score >= TIER_HIGH:
        conf_tier = "High Confidence"
    elif conf_score >= TIER_MED:
        conf_tier = "Medium Confidence"
    else:
        conf_tier = "Low Confidence"
    fusion_conf = conf_score

# ===== OUTPUT =====
print("\n🔍 Deepfake Prediction:")
print(f"    Raw scores: [Fake={deep_pred_probs[0]:.4f}, Real={deep_pred_probs[1]:.4f}]")
print(f"    Class: {deep_class_str}")

print("\n🔍 Micro-Expression Prediction Probabilities:")
for idx, emo in enumerate(emotion_labels):
    print(f"    {emo.capitalize():<10}: {micro_probs[idx]:.4f}")
print(f"    Predicted Emotion: {micro_class_str}")

if override_label:
    print("\n⚠️ Rule-Based Override Triggered:")
    print(f"    Reason: {override_reason}")
    print(f"    Final Decision: {fusion_label}")
else:
    print("\n🧠 ML Fusion Decision (Weighted Probability Voting):")
    print(f"    RF Probabilities:   [SPOOF={rf_proba[0]:.4f}, LIVE={rf_proba[1]:.4f}]")
    print(f"    XGB Probabilities:  [SPOOF={xgb_proba[0]:.4f}, LIVE={xgb_proba[1]:.4f}]")
    print(f"    Weighted Probabilities: [SPOOF={weighted_proba[0]:.4f}, LIVE={weighted_proba[1]:.4f}]")
    print(f"    Final Decision: {fusion_label} ({conf_tier}, {fusion_conf:.2%} confidence)")

# ===== SHAP EXPLAINABILITY =====
print("\n📊 SHAP Feature Contributions (RF model):")
explainer = shap.Explainer(rf_clf, feature_names=["deep_score"] + emotion_labels)
shap_values = explainer(features_scaled)
for name, val in zip(["deep_score"] + emotion_labels, shap_values.values[0]):
    if isinstance(val, (np.ndarray, list)):
        # If val has more than one element, choose the value related to the positive class or average
        val_to_print = val.item() if val.size == 1 else val[1] if len(val) > 1 else val[0]
    else:
        val_to_print = val

    print(f"    {name:<15}: {val_to_print:.4f}")

