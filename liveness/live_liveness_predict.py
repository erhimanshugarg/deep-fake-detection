"""
🎯 Unified Liveness Prediction Script
───────────────────────────────────────────────────────────────────────────────

This script provides real-time liveness detection by combining deepfake detection
and microexpression analysis. It's designed to distinguish between live human faces
and various spoofing attempts (photos, videos, deepfakes) in security applications.

📋 FUNCTIONALITY:
- Captures input from webcam or video file
- Extracts frames at regular intervals
- Processes frames for both deepfake and microexpression models
- Combines model predictions using a trained fusion classifier
- Applies confidence-based override rules for edge cases
- Outputs final LIVE/SPOOF decision with confidence scores

🧠 MODELS & ARCHITECTURE:
- Deepfake Detection: MobileNetV2-based model
  - Input: Single face frame (224×224 RGB)
  - Output: Binary classification (real/fake) with confidence score

- Microexpression Analysis: CNN+LSTM temporal model
  - Input: Sequence of 16 face frames (224×224 RGB)
  - Output: 3-class emotion prediction (positive/negative/surprise)

- Fusion Strategy: Late fusion with Logistic Regression
  - Input: Confidence scores from both models
  - Output: Final binary decision (LIVE/SPOOF)
  - Override rules for high-confidence cases

📊 INPUT/OUTPUT:
- Input:
  - Webcam stream (default camera)
  - Video file (MP4, AVI, etc.)
- Output:
  - Console: Detailed prediction scores and final decision
  - Visual feedback: Webcam preview window (when using webcam)
  - Temporary files: Extracted frames in dataset/temp_input/

🔍 USAGE:
- Run script: python live_liveness_predict.py
- When prompted, enter "webcam" or a video file path
- For webcam mode: Press 'q' to stop capture after sufficient frames
- Review prediction results in console output
- Final decision appears as "✅ LIVE" or "❌ SPOOF"

📝 IMPLEMENTATION NOTES:
- Requires pre-trained models (deepfake, microexpression, fusion)
- Temporary frames are saved to disk and cleaned up between runs
- Confidence-based override provides fallback for edge cases
- Processing time depends on hardware capabilities
"""

import os
import cv2
import numpy as np
import tempfile
import shutil
from glob import glob
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from joblib import load as joblib_load

# === Paths ===
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_INPUT = os.path.join(ROOT, 'dataset', 'temp_input')
DEEP_MODEL_PATH = os.path.join(ROOT, 'deepfake', 'model', 'mobilenet_deepfake_model_v3.keras')
MICRO_MODEL_PATH = os.path.join(ROOT, 'microexpression', 'models', 'microexpression_model_20250622_2034.keras')
FUSION_MODEL_PATH = os.path.join(ROOT, 'fusion', 'fusion/fusion_classifier_model_20250701_2237.joblib')
SCALER_PATH = os.path.join(ROOT, 'fusion', 'fusion/fusion_scaler_20250701_2237.joblib')

# === Load Models ===
print("📦 Loading models...")
deep_model = load_model(DEEP_MODEL_PATH)
micro_model = load_model(MICRO_MODEL_PATH)
fusion_model = joblib_load(FUSION_MODEL_PATH)
scaler = joblib_load(SCALER_PATH)

# === Utility Functions ===

def extract_frames_from_video(video_path, output_dir, max_frames=60, interval=5):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    os.makedirs(output_dir, exist_ok=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved >= max_frames:
            break
        if count % interval == 0:
            path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        count += 1
    cap.release()

def capture_from_webcam(output_dir, frame_count=60):
    cap = cv2.VideoCapture(0)
    os.makedirs(output_dir, exist_ok=True)
    print("📷 Capturing from webcam... Press 'q' to stop.")
    saved = 0
    while cap.isOpened() and saved < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam - Press q to stop", frame)
        path = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def prepare_deepfake_input(frames_path):
    jpgs = sorted(glob(os.path.join(frames_path, '*.jpg')))
    if not jpgs:
        raise Exception("No frames found for deepfake input.")
    img = cv2.imread(jpgs[len(jpgs) // 2])
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)

def prepare_microexpression_input(frames_path):
    jpgs = sorted(glob(os.path.join(frames_path, '*.jpg')))[:16]
    if len(jpgs) < 16:
        raise Exception("Not enough frames for microexpression input.")
    sequence = []
    for f in jpgs:
        img = cv2.imread(f)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        sequence.append(img)
    return np.expand_dims(np.array(sequence), axis=0)

def predict_liveness(X_deep, X_micro):
    deep_pred = deep_model.predict(X_deep, verbose=0)[0]  # sigmoid or softmax
    micro_pred = micro_model.predict(X_micro, verbose=0)[0]  # softmax

    # Handle binary sigmoid output
    if deep_pred.shape == ():  # scalar
        deep_pred = np.array([1 - deep_pred, deep_pred])  # [Real, Fake]
    elif deep_pred.shape[0] == 1:
        deep_pred = np.array([1 - deep_pred[0], deep_pred[0]])

    deep_class = int(deep_pred[1] > 0.5)
    micro_class = np.argmax(micro_pred)

    # Use only top class probabilities (2 features)
    fusion_input = np.array([[deep_pred[deep_class], micro_pred[micro_class]]])
    input_scaled = scaler.transform(fusion_input)
    fusion_result = fusion_model.predict(input_scaled)[0]

    print(f"\n🎭 Deepfake Score: {deep_pred} → Class: {['Real', 'Fake'][deep_class]}")
    print(f"😐 Micro-expression Score: {micro_pred} → Class: {['Positive', 'Negative', 'Surprise'][micro_class]}")
    print(f"🔗 Fusion Model Decision: {'LIVE' if fusion_result == 'LIVE' else 'SPOOF'}")

    # === Confidence-based override ===
    deep_conf_fake = deep_pred[1]
    micro_conf_negative = micro_pred[1]

    if deep_conf_fake < 0.4 and micro_conf_negative < 0.4:
        print("🧠 Override: Both models suggest LIVE → ✅ LIVE")
    else:
        print(f"\n🎯 Final Liveness Decision: {'✅ LIVE' if fusion_result == 'LIVE' else '❌ SPOOF'}")


# === MAIN EXECUTION ===

if __name__ == "__main__":
    input_type = input("Choose input type - webcam or video path: ").strip()

    if os.path.exists(TEMP_INPUT):
        shutil.rmtree(TEMP_INPUT)
    os.makedirs(TEMP_INPUT, exist_ok=True)

    if input_type.lower() == "webcam":
        capture_from_webcam(TEMP_INPUT)
    elif os.path.exists(input_type):
        extract_frames_from_video(input_type, TEMP_INPUT)
    else:
        print("❌ Invalid input. Exiting.")
        exit()

    try:
        X_deep = prepare_deepfake_input(TEMP_INPUT)
        X_micro = prepare_microexpression_input(TEMP_INPUT)
        predict_liveness(X_deep, X_micro)
    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")
