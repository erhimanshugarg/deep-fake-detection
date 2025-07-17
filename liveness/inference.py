# liveness/inference.py

import os
import numpy as np
import cv2
from glob import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from joblib import load as joblib_load

# === Paths ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEP_MODEL_PATH = os.path.join(ROOT, "deepfake", "model", "mobilenet_deepfake_model_v3.keras")
MICRO_MODEL_PATH = os.path.join(ROOT, "microexpression", "models", "microexpression_model_20250622_2034.keras")
FUSION_MODEL_PATH = os.path.join(ROOT, "fusion", "fusion", "fusion_classifier_model_20250701_2237.joblib")
SCALER_PATH = os.path.join(ROOT, "fusion", "fusion", "fusion_scaler_20250701_2237.joblib")

# === Load models once ===
deep_model = load_model(DEEP_MODEL_PATH)
micro_model = load_model(MICRO_MODEL_PATH)
fusion_model = joblib_load(FUSION_MODEL_PATH)
scaler = joblib_load(SCALER_PATH)

def preprocess_video(video_path, output_dir, max_frames=60, interval=5):
    cap = cv2.VideoCapture(video_path)
    saved = 0
    frame_id = 0
    os.makedirs(output_dir, exist_ok=True)
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        frame_id += 1
    cap.release()

def prepare_inputs(frames_path):
    jpgs = sorted(glob(os.path.join(frames_path, '*.jpg')))
    if len(jpgs) < 16:
        raise ValueError("Need at least 16 frames for inference.")

    # Deepfake input: use middle frame
    img = cv2.imread(jpgs[len(jpgs) // 2])
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    deep_input = preprocess_input(img.astype(np.float32))
    deep_input = np.expand_dims(deep_input, axis=0)

    # Microexpression input: first 16 frames
    sequence = []
    for f in jpgs[:16]:
        img = cv2.imread(f)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        sequence.append(img)
    micro_input = np.expand_dims(np.array(sequence), axis=0)

    return deep_input, micro_input

def predict_liveness_from_frames(frames_path):
    print("🧪 Preparing inputs...")
    X_deep, X_micro = prepare_inputs(frames_path)

    print("🔍 Running deepfake model...")
    deep_pred = deep_model.predict(X_deep, verbose=0)[0]
    if deep_pred.shape == ():  # scalar
        deep_pred = np.array([1 - deep_pred, deep_pred])
    elif deep_pred.shape[0] == 1:
        deep_pred = np.array([1 - deep_pred[0], deep_pred[0]])
    deep_class = int(deep_pred[1] > 0.5)

    print(f"🔢 Raw deepfake output: {deep_pred}")
    print(f"🎭 Deepfake: {deep_pred} → class: {['REAL', 'FAKE'][deep_class]}")

    print("🔍 Running microexpression model...")
    micro_pred = micro_model.predict(X_micro, verbose=0)[0]
    micro_class = np.argmax(micro_pred)
    print(f"😐 Micro-expression: {micro_pred} → class: {['Positive', 'Negative', 'Surprise'][micro_class]}")

    # Fusion input
    fusion_input = np.array([[deep_pred[deep_class], micro_pred[micro_class]]])
    input_scaled = scaler.transform(fusion_input)
    fusion_result = fusion_model.predict(input_scaled)[0]
    fusion_proba = fusion_model.predict_proba(input_scaled)[0]
    print(f"🔗 Fusion result: {fusion_result}")
    print(f"📊 Fusion probas: {fusion_proba}")

    return {
        "deep_class": "REAL" if deep_class == 0 else "FAKE",
        "deep_confidence": float(deep_pred[deep_class]),
        "micro_class": ["Positive", "Negative", "Surprise"][micro_class],
        "micro_confidence": float(micro_pred[micro_class]),
        "fusion_result": "LIVE" if fusion_result == 1 else "SPOOF",
        "confidence": float(fusion_proba[int(fusion_result)])
    }
