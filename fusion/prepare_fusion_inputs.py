"""
prepare_fusion_inputs.py (Updated for 6-class micro-expression model)

- Extracts one representative face frame from deepfake real dataset
- Extracts one 16-frame micro-expression sequence from microexpression dataset
- Preprocesses images (resize, color convert, normalize) consistent with batch pipeline
- Saves numpy arrays as X_deepfake.npy and X_microexpression.npy in fusion/inputs/
"""

import os
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Paths ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')
fusion_input_dir = os.path.join(project_root, 'fusion', 'inputs')
os.makedirs(fusion_input_dir, exist_ok=True)

# === Deepfake Input: pick one real frame ===
real_frame_paths = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'real', '*', '*.jpg')))
if len(real_frame_paths) == 0:
    raise Exception("No real frames found in processed_data/real/")
deep_img_path = real_frame_paths[0]

deep_img = cv2.imread(deep_img_path)
deep_img = cv2.resize(deep_img, (224, 224))
deep_img = cv2.cvtColor(deep_img, cv2.COLOR_BGR2RGB)
deep_img = preprocess_input(deep_img.astype(np.float32))
deep_img = np.expand_dims(deep_img, axis=0)  # shape (1, 224, 224, 3)

np.save(os.path.join(fusion_input_dir, 'X_deepfake.npy'), deep_img)
print(f"✅ Saved deepfake input: {os.path.join(fusion_input_dir, 'X_deepfake.npy')}")

# === Micro-expression Input: find one sequence with at least 16 frames ===
microexpression_root = os.path.join(dataset_dir, 'microexpression_processed')
emotion_dirs = sorted([d for d in glob(os.path.join(microexpression_root, '*')) if os.path.isdir(d)])

sequence = []
found_sequence = False

for emotion_dir in emotion_dirs:
    video_dirs = sorted([v for v in glob(os.path.join(emotion_dir, '*')) if os.path.isdir(v)])
    for video_dir in video_dirs:
        frame_paths = sorted(glob(os.path.join(video_dir, '*.jpg')))
        if len(frame_paths) >= 16:
            for f in frame_paths[:16]:
                img = cv2.imread(f)
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0  # scale as [0, 1]
                sequence.append(img)
            found_sequence = True
            break
    if found_sequence:
        break

if not found_sequence:
    raise Exception("❌ Could not find a 16-frame micro-expression sequence")

X_micro = np.expand_dims(np.array(sequence), axis=0)  # shape (1, 16, 224, 224, 3)
np.save(os.path.join(fusion_input_dir, 'X_microexpression.npy'), X_micro)
print(f"✅ Saved micro-expression input: {os.path.join(fusion_input_dir, 'X_microexpression.npy')}")

print("🎉 Fusion input preparation complete.")
