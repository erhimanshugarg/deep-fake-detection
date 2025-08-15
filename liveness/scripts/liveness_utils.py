# liveness_utils.py
import cv2
import numpy as np
import pandas as pd
import torch
torch.classes.__path__ = []
from torchvision import transforms
from tensorflow.keras.models import load_model
import joblib

import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent            # .../deepfake-detection/liveness/scripts
project_root = current_dir.parents[1]                    # .../deepfake-detection
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Optional sanity check (can be removed)
# print(f"[INFO] Project root added to sys.path: {project_root}")

# Import ResNet18AU from microexpression package
from microexpression.resnet18_au_model import ResNet18AU

# ===== Configuration =====
WEIGHT_RF = 0.98
WEIGHT_XGB = 0.96
DECISION_THRESHOLD = 0.5
TIER_HIGH = 0.85
TIER_MED = 0.65

MODEL_EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

# ===== Paths =====
LOG_DIR = os.path.join(project_root, 'liveness', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

DEEPFAKE_MODEL_PATH = os.path.join(project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_v3.keras')
MICROEXP_MODEL_PATH = os.path.join(project_root, 'microexpression', 'models', 'resnet18_microexpr_v9_best.pth')

FUSION_JOBLIB_DIR = os.path.join(project_root, 'fusion', 'joblib')

# ===== Load Models =====
def load_all_models():
    # Load deepfake model
    deepfake_model = load_model(DEEPFAKE_MODEL_PATH)

    # Load micro-expression AU model
    micro_model = ResNet18AU(num_classes=len(MODEL_EMOTIONS), au_feature_dim=12)
    micro_model.load_state_dict(torch.load(MICROEXP_MODEL_PATH, map_location='cpu'))
    micro_model.eval()

    # Load fusion models and scaler
    rf_model_path = max((os.path.join(FUSION_JOBLIB_DIR, f)
                         for f in os.listdir(FUSION_JOBLIB_DIR) if f.startswith("fusion_rf_model_")),
                         key=os.path.getctime)
    xgb_model_path = max((os.path.join(FUSION_JOBLIB_DIR, f)
                          for f in os.listdir(FUSION_JOBLIB_DIR) if f.startswith("fusion_xgb_model_")),
                          key=os.path.getctime)
    scaler_path = max((os.path.join(FUSION_JOBLIB_DIR, f)
                       for f in os.listdir(FUSION_JOBLIB_DIR) if f.startswith("fusion_scaler_")),
                       key=os.path.getctime)

    rf_clf = joblib.load(rf_model_path)
    xgb_clf = joblib.load(xgb_model_path)
    scaler = joblib.load(scaler_path)

    return deepfake_model, micro_model, rf_clf, xgb_clf, scaler

# ===== Image Transform for micro-expression input =====
transform_me = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ===== Prepare Micro-expression Frames =====
def prepare_microexpr_frames(frames):
    if len(frames) < 16:
        frames += [frames[-1]] * (16 - len(frames))
    elif len(frames) > 16:
        frames = frames[-16:]
    processed = [transform_me(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    return torch.stack(processed).unsqueeze(0)  # Shape: (1, 16, 3, 224, 224)

# ===== Run Fusion Inference =====
def run_inference(deepfake_model, micro_model, rf_clf, xgb_clf, scaler, frames):
    if not frames:
        raise ValueError("Empty frame sequence received for inference.")
    mid_frame = frames[len(frames) // 2]
    deep_img = cv2.resize(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB), (224, 224))
    deep_input = np.expand_dims(deep_img.astype(np.float32) / 255.0, axis=0)

    deep_pred = deepfake_model.predict(deep_input, verbose=0)[0]
    if len(deep_pred) == 1:
        deep_score = float(deep_pred[0])
        deep_pred_probs = [1 - deep_score, deep_score]
    else:
        deep_pred_probs = deep_pred.tolist()
        deep_score = deep_pred_probs[1]

    seq_tensor = prepare_microexpr_frames(frames)
    with torch.no_grad():
        micro_logits = micro_model(seq_tensor)
        micro_probs = torch.softmax(micro_logits, dim=1)[0].cpu().numpy()

    features = np.array([deep_score] + micro_probs.tolist()).reshape(1, -1)
    features_scaled = scaler.transform(features)
    rf_proba = rf_clf.predict_proba(features_scaled)[0]
    xgb_proba = xgb_clf.predict_proba(features_scaled)[0]
    weighted_proba = (rf_proba * WEIGHT_RF + xgb_proba * WEIGHT_XGB) / (WEIGHT_RF + WEIGHT_XGB)
    final_pred = 1 if weighted_proba[1] >= DECISION_THRESHOLD else 0
    fusion_label = "LIVE" if final_pred == 1 else "SPOOF"

    conf_score = weighted_proba[final_pred]
    if conf_score >= TIER_HIGH:
        conf_tier = "High Confidence"
    elif conf_score >= TIER_MED:
        conf_tier = "Medium Confidence"
    else:
        conf_tier = "Low Confidence"

    return {
        "fusion_label": fusion_label,
        "confidence_tier": conf_tier,
        "confidence_score": conf_score,
        "deep_score": deep_score,
        "deep_pred_probs": deep_pred_probs,
        "micro_probs": micro_probs,
        "weighted_proba": weighted_proba
    }

# ===== Log Session =====
def log_session(timestamp, fusion_label, conf_tier, conf_score, deep_score, micro_probs):
    log_file = os.path.join(LOG_DIR, "session_log.csv")
    row = {
        "timestamp": timestamp,
        "fusion_label": fusion_label,
        "confidence_tier": conf_tier,
        "confidence_score": conf_score,
        "deep_score": deep_score,
        **{MODEL_EMOTIONS[i]: micro_probs[i] for i in range(len(MODEL_EMOTIONS))}
    }
    df = pd.DataFrame([row])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, index=False, mode='a', header=False)
