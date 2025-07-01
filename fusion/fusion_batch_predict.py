"""
🔁 fusion_batch_predict.py
────────────────────────────────────────────
Performs batch-level fusion predictions using 1-to-1 matching between:
- Deepfake face frames
- Micro-expression sequences

📁 Expected Folder Structure:
dataset/
├── processed_data/
│   ├── real/subject_id/frame.jpg
│   └── fake/subject_id/frame.jpg
├── microexpression_processed/
│   └── emotion/video_id/frame.jpg

Outputs:
- fusion/fusion_batch_results_<timestamp>.csv
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
os.makedirs(fusion_dir, exist_ok=True)

# === Load Models ===
print("📦 Loading models...")
deep_model = load_model(os.path.join(project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_v3.keras'))
micro_model = load_model(os.path.join(project_root, 'microexpression', 'models', 'microexpression_model_20250622_2034.keras'))

# === Collect Deepfake Frames ===
real_frames = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'real', '*', '*.jpg')))[:50]
fake_frames = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'fake', '*', '*.jpg')))[:50]
deep_frames = real_frames + fake_frames
deep_labels = ["real"] * len(real_frames) + ["fake"] * len(fake_frames)

# === Collect Microexpression Sequences ===
sequence_dirs = []
micro_processed_path = os.path.join(dataset_dir, 'microexpression_processed')

for emotion in os.listdir(micro_processed_path):
    emotion_path = os.path.join(micro_processed_path, emotion)
    if os.path.isdir(emotion_path):
        for video_dir in os.listdir(emotion_path):
            full_path = os.path.join(emotion_path, video_dir)
            if os.path.isdir(full_path):
                frames = glob(os.path.join(full_path, '*.jpg'))
                if len(frames) >= 16:
                    sequence_dirs.append((emotion, full_path))

# === Limit to shortest set for 1-to-1 matching ===
limit = min(len(deep_frames), len(sequence_dirs))
results = []

print(f"🔄 Running {limit} fusion predictions (1-to-1)...")

# === Smart Fusion Rule ===
def smart_fusion_decision(deep_score, micro_class, threshold=0.8):
    if deep_score < threshold:
        return "LIVE"
    elif micro_class in [0, 1, 2]:
        return "SPOOF"
    else:
        return "INCONCLUSIVE"

# === Prediction Loop ===
for i in range(limit):
    # --- Deepfake Frame ---
    df_img = cv2.imread(deep_frames[i])
    df_img = cv2.resize(df_img, (224, 224))
    df_img = cv2.cvtColor(df_img, cv2.COLOR_BGR2RGB)
    df_img = preprocess_input(df_img.astype(np.float32))
    X_deep = np.expand_dims(df_img, axis=0)

    # --- Microexpression Sequence ---
    frames = sorted(glob(os.path.join(sequence_dirs[i][1], '*.jpg')))[:16]
    seq = []
    for f in frames:
        img = cv2.imread(f)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        seq.append(img)
    X_micro = np.expand_dims(np.array(seq), axis=0)

    # === Predictions ===
    deep_pred = deep_model.predict(X_deep, verbose=0)[0]
    micro_pred = micro_model.predict(X_micro, verbose=0)[0]

    deep_class = np.argmax(deep_pred)  # 0=fake, 1=real
    micro_class = np.argmax(micro_pred)

    # === Apply Smart Fusion Rule ===
    fusion = smart_fusion_decision(deep_score=deep_pred[0], micro_class=micro_class)

    results.append({
        "index": i + 1,
        "deep_label": deep_labels[i],
        "deep_pred": "real" if deep_class == 1 else "fake",
        "deep_score": round(float(deep_pred[deep_class]), 4),
        "micro_emotion": ["positive", "negative", "surprise"][micro_class],
        "micro_score": round(float(micro_pred[micro_class]), 4),
        "fusion_result": fusion
    })

# === Save CSV ===
df = pd.DataFrame(results)
dt_str = datetime.now().strftime("%Y%m%d_%H%M")
csv_path = os.path.join(fusion_dir, f"fusion_batch_results_{dt_str}.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Saved fusion batch results: {csv_path}")
