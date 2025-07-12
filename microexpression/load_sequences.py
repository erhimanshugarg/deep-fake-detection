"""
🔄 Micro-Expression Sequence Preparation Script
───────────────────────────────────────────────────────────────────────────────

This script converts preprocessed micro-expression video frames into fixed-length
sequences suitable for temporal deep learning models. It's a critical bridge between
raw video preprocessing and model training in the micro-expression recognition pipeline.

📋 FUNCTIONALITY:
- Loads preprocessed face images from the CASME2 dataset
- Organizes frames into fixed-length sequences (16 frames per sequence)
- Handles variable-length videos through padding or truncation
- Converts emotion class labels to numerical format for training
- Saves the prepared sequences and labels as NumPy arrays
- Filters out corrupted frames and videos with insufficient frames

🧠 ARCHITECTURE:
- Sequence Standardization: Fixed-length sequence creation (16 frames)
- Frame Processing: Loading, resizing, and normalization of image data
- Label Encoding: Categorical mapping (positive → 0, negative → 1, surprise → 2)
- Error Handling: Skips corrupted frames and insufficient sequences

📊 INPUT/OUTPUT:
- Input:
  - Preprocessed face images from preprocess.py
  - Structure: dataset/microexpression_processed/{positive,negative,surprise}/video_name/*.jpg
  - Format: 224×224 RGB images
- Output:
  - X_sequences.npy: NumPy array with shape (num_samples, 16, 224, 224, 3)
  - y_labels.npy: NumPy array with shape (num_samples,) containing class indices
  - Saved to: dataset/microexpression_processed/

🔍 USAGE:
- Run after completing preprocess.py
- Execute: python load_sequences.py
- Monitor progress with built-in progress bars
- The output files are used directly by train_model_v4.py for model training

📝 NOTES:
- Videos with fewer than 16 frames are padded by repeating the last frame
- Videos with more than 16 frames are truncated to the first 16 frames
- Videos with fewer than 4 frames or corrupted frames are skipped entirely
- The sequence length (16) is optimized for micro-expression temporal patterns
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# === Configuration ===
processed_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'microexpression_processed'))
output_x_path = os.path.join(processed_data_dir, 'X_sequences.npy')
output_y_path = os.path.join(processed_data_dir, 'y_labels.npy')

emotion_classes = ['positive', 'negative', 'surprise']
emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotion_classes)}

sequence_length = 16
image_size = (224, 224)

all_sequences = []
all_labels = []

# === Helper: Load and resize a single frame ===
def load_and_resize_frame(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.resize(image, image_size)

# === Loop through each emotion class ===
for class_name in emotion_classes:
    class_path = os.path.join(processed_data_dir, class_name)
    if not os.path.isdir(class_path):
        print(f"⚠️ Skipping missing emotion class folder: {class_name}")
        continue

    for video_folder in tqdm(os.listdir(class_path), desc=f"Loading '{class_name}' videos"):
        video_path = os.path.join(class_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        frame_file_paths = sorted([
            os.path.join(video_path, file)
            for file in os.listdir(video_path) if file.endswith('.jpg')
        ])

        # Skip videos with very few frames
        if len(frame_file_paths) < 4:
            continue

        # Trim or pad to fixed sequence length
        if len(frame_file_paths) < sequence_length:
            frame_file_paths += [frame_file_paths[-1]] * (sequence_length - len(frame_file_paths))
        else:
            frame_file_paths = frame_file_paths[:sequence_length]

        frame_sequence = [load_and_resize_frame(fp) for fp in frame_file_paths]
        frame_sequence = [f for f in frame_sequence if f is not None]

        if len(frame_sequence) != sequence_length:
            print(f"⚠️ Skipping {video_folder} (corrupt or unreadable frames)")
            continue

        all_sequences.append(np.stack(frame_sequence))  # (sequence_length, H, W, C)
        all_labels.append(emotion_to_label[class_name])

# === Convert to NumPy arrays and save ===
X = np.array(all_sequences)
y = np.array(all_labels)

np.save(output_x_path, X)
np.save(output_y_path, y)

print("✅ Sequence preparation complete.")
print(f"📦 X shape: {X.shape}, y shape: {y.shape}")
print(f"💾 Saved to:\n  {output_x_path}\n  {output_y_path}")
