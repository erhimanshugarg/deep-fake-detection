"""
fusion_batch_predict.py (Updated for 6-class microexpression model)
- Loads deepfake (Keras) + microexpression (PyTorch) models
- Runs predictions on 100 real + 100 fake frame/microexpression pairs
- Saves results CSV in fusion/batch_results with ground-truth fusion labels for training

fusion_batch_predict.py (Updated to output all 6 micro-expression probabilities)
"""

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import torch
from torchvision import transforms

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from microexpression.resnet18_au_model import ResNet18AU  # your PyTorch model

# === Paths ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset')
fusion_dir = os.path.join(project_root, 'fusion')
batch_result_dir = os.path.join(fusion_dir, 'batch_results')
os.makedirs(batch_result_dir, exist_ok=True)

# === Load Models ===
print("📦 Loading deepfake model...")
deep_model_path = os.path.join(project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_fine_tuned.keras')
deep_model = load_model(deep_model_path)

print("📦 Loading microexpression model...")
micro_model_path = os.path.join(project_root, 'microexpression', 'models', 'resnet18_microexpr_v9_best.pth')
micro_model = ResNet18AU(num_classes=6, au_feature_dim=12)
micro_model.load_state_dict(torch.load(micro_model_path, map_location='cpu'))
micro_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

emotion_labels = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# === Collect Frames ===
real_frames = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'real', '*', '*.jpg')))[:100]
fake_frames = sorted(glob(os.path.join(dataset_dir, 'processed_data', 'fake', '*', '*.jpg')))[:100]
deep_frames = real_frames + fake_frames
deep_labels = ["real"] * len(real_frames) + ["fake"] * len(fake_frames)

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

limit = min(len(deep_frames), len(sequence_dirs))
print(f"🔄 Running {limit} fusion predictions (1-to-1)...")

results = []
for i in range(limit):
    df_img = cv2.imread(deep_frames[i])
    df_img = cv2.resize(df_img, (224, 224))
    df_img = cv2.cvtColor(df_img, cv2.COLOR_BGR2RGB)
    df_img = preprocess_input(df_img.astype(np.float32))
    X_deep = np.expand_dims(df_img, axis=0)
    deep_pred = deep_model.predict(X_deep, verbose=0)[0]
    deep_class = int(np.argmax(deep_pred))

    frames = sorted(glob(os.path.join(sequence_dirs[i][1], '*.jpg')))[:16]
    seq_tensor = torch.stack([transform(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)) for f in frames])
    seq_tensor = seq_tensor.unsqueeze(0)
    with torch.no_grad():
        micro_logits = micro_model(seq_tensor)
        micro_probs = torch.softmax(micro_logits, dim=1)[0].cpu().numpy()
    micro_class = int(np.argmax(micro_probs))

    fusion_label = "LIVE" if deep_labels[i] == "real" else "SPOOF"

    result = {
        "index": i + 1,
        "deep_label": deep_labels[i],
        "deep_pred": "real" if deep_class == 1 else "fake",
        "deep_score": round(float(deep_pred[deep_class]), 4),
        "micro_emotion": emotion_labels[micro_class],
        "micro_score": round(float(micro_probs[micro_class]), 4),
        "fusion_result": fusion_label
    }

    # Add all micro-expression class probabilities
    for idx, emotion in enumerate(emotion_labels):
        result[f"micro_score_{emotion}"] = float(round(micro_probs[idx], 6))

    results.append(result)

df = pd.DataFrame(results)
dt_str = datetime.now().strftime("%Y%m%d_%H%M")
csv_path = os.path.join(batch_result_dir, f"fusion_batch_results_{dt_str}.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Saved fusion batch results: {csv_path}")
