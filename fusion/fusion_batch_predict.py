"""
🔁 Fusion Batch Predict Script
───────────────────────────────────────────────
Performs batch fusion predictions on:
- 100 real + 100 fake deepfake frames
- Matching 200 micro-expression sequences

Outputs:
- fusion/batch_results/fusion_batch_results_<timestamp>.csv
- fusion/batch_results/fusion_batch_results_<timestamp>.docx
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Paths ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')
fusion_dir = os.path.join(project_root, 'fusion')
batch_result_dir = os.path.join(fusion_dir, 'batch_results')
os.makedirs(batch_result_dir, exist_ok=True)

# === Load Models ===
print("📦 Loading models...")
deep_model_path = os.path.join(project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_fine_tuned.keras')
micro_model_path = os.path.join(project_root, 'microexpression', 'models', 'microexpression_model_20250622_2034.keras')

deep_model = load_model(deep_model_path)
micro_model = load_model(micro_model_path)

# === Collect Deepfake Frames ===
real_frames = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'real', '*', '*.jpg')))[:100]
fake_frames = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'fake', '*', '*.jpg')))[:100]
deep_frames = real_frames + fake_frames
deep_labels = ["real"] * len(real_frames) + ["fake"] * len(fake_frames)

# === Collect Microexpression Sequences ===
sequence_dirs = []
micro_root = os.path.join(dataset_dir, 'microexpression_processed')

for emotion in os.listdir(micro_root):
    emotion_path = os.path.join(micro_root, emotion)
    if os.path.isdir(emotion_path):
        for video_dir in os.listdir(emotion_path):
            video_path = os.path.join(emotion_path, video_dir)
            if os.path.isdir(video_path):
                frames = glob(os.path.join(video_path, '*.jpg'))
                if len(frames) >= 16:
                    sequence_dirs.append((emotion, video_path))

# === Match Count
limit = min(len(deep_frames), len(sequence_dirs))
print(f"🔄 Running {limit} fusion predictions (1-to-1)...")

results = []

for i in range(limit):
    # --- Deepfake Frame ---
    df_img = cv2.imread(deep_frames[i])
    df_img = cv2.resize(df_img, (224, 224))
    df_img = cv2.cvtColor(df_img, cv2.COLOR_BGR2RGB)
    df_img = preprocess_input(df_img.astype(np.float32))
    X_deep = np.expand_dims(df_img, axis=0)

    # --- Microexpression Sequence ---
    frames = sorted(glob(os.path.join(sequence_dirs[i][1], '*.jpg')))[:16]
    sequence = []
    for f in frames:
        img = cv2.imread(f)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        sequence.append(img)
    X_micro = np.expand_dims(np.array(sequence), axis=0)

    # --- Predictions ---
    deep_pred = deep_model.predict(X_deep, verbose=0)[0]
    micro_pred = micro_model.predict(X_micro, verbose=0)[0]

    deep_class = np.argmax(deep_pred)
    micro_class = np.argmax(micro_pred)

    # Fusion rule
    if deep_class == 1 and micro_class in [0, 1, 2]:
        fusion = "LIVE"
    else:
        fusion = "SPOOF"

    results.append({
        "index": i + 1,
        "deep_label": deep_labels[i],
        "deep_pred": "real" if deep_class == 1 else "fake",
        "deep_score": round(float(deep_pred[deep_class]), 4),
        "micro_emotion": ["positive", "negative", "surprise"][micro_class],
        "micro_score": round(float(micro_pred[micro_class]), 4),
        "fusion_result": fusion
    })

# === Save Result CSV
df = pd.DataFrame(results)
dt_str = datetime.now().strftime("%Y%m%d_%H%M")
csv_path = os.path.join(batch_result_dir, f"fusion_batch_results_{dt_str}.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Saved fusion batch results: {csv_path}")
