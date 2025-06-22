"""
🎯 prepare_fusion_inputs.py (Updated)
────────────────────────────────────────────
Creates demo input files for fusion prediction using preprocessed images.

✔️ Extracts:
- A face frame from deepfake/processed_data/real/
- A 16-frame sequence from microexpression_processed/

📁 Expected Structure:
dataset/
├── processed_data/
│   ├── real/video_id/frame.jpg
│   └── fake/video_id/frame.jpg
├── microexpression_processed/
│   └── emotion_class/video_id/*.jpg

🔽 Output:
- fusion/X_deepfake.npy
- fusion/X_microexpression.npy
"""

import os
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Paths ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')

# === Deepfake Input ===
real_frame_path = glob(os.path.join(dataset_dir, 'processed_data', 'real', '*', '*.jpg'))[0]
deep_img = cv2.imread(real_frame_path)
deep_img = cv2.resize(deep_img, (224, 224))
deep_img = cv2.cvtColor(deep_img, cv2.COLOR_BGR2RGB)
deep_img = preprocess_input(deep_img.astype(np.float32))
deep_img = np.expand_dims(deep_img, axis=0)  # (1, 224, 224, 3)

# Save
fusion_dir = os.path.join(project_root, 'fusion')
os.makedirs(fusion_dir, exist_ok=True)
np.save(os.path.join(fusion_dir, 'X_deepfake.npy'), deep_img)
print("✅ Saved: X_deepfake.npy")

# === Microexpression Input ===
# Find one sequence with at least 16 frames
micro_seq_root = os.path.join(dataset_dir, 'microexpression_processed')
class_dirs = [d for d in glob(os.path.join(micro_seq_root, '*')) if os.path.isdir(d)]

sequence = []
found = False
for class_dir in class_dirs:
    video_dirs = [v for v in glob(os.path.join(class_dir, '*')) if os.path.isdir(v)]
    for video_dir in video_dirs:
        frames = sorted(glob(os.path.join(video_dir, '*.jpg')))
        if len(frames) >= 16:
            for f in frames[:16]:
                img = cv2.imread(f)
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                sequence.append(img)
            found = True
            break
    if found:
        break

if not found:
    raise Exception("❌ Could not find a 16-frame microexpression sequence")

X_seq = np.expand_dims(np.array(sequence), axis=0)  # (1, 16, 224, 224, 3)
np.save(os.path.join(fusion_dir, 'X_microexpression.npy'), X_seq)
print("✅ Saved: X_microexpression.npy")

print("🎉 Fusion input preparation complete.")
