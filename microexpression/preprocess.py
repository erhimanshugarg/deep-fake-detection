"""
🎬 Micro-Expression Video Preprocessing Script
───────────────────────────────────────────────────────────────────────────────

This script preprocesses the CASME2 micro-expression dataset by extracting face 
regions from video frames and organizing them into an emotion-based directory 
structure for deep learning model training.

📋 FUNCTIONALITY:
- Loads video files from the CASME2 dataset
- Processes emotion labels from an Excel metadata file
- Maps raw emotions to 3 standardized categories (positive, negative, surprise)
- Extracts face regions from video frames using OpenCV's Haar Cascade detector
- Resizes face images to a consistent format (224×224 pixels)
- Saves processed images in an organized directory structure by emotion class

🧠 ARCHITECTURE:
- Face Detection: OpenCV Haar Cascade (lightweight, CPU-based detection)
- Emotion Mapping: Custom mapping function for CASME2 emotion categories
- Frame Processing: Sequential extraction and saving of video frames
- Error Handling: Skips frames where face detection fails

📊 INPUT/OUTPUT:
- Input:
  - CASME2 video files (.avi format)
  - CASME2 coding Excel file with emotion labels
  - Expected structure: dataset/CASME2/sub*/video_name.avi
- Output:
  - Processed face images organized by emotion
  - Structure: dataset/microexpression_processed/{positive,negative,surprise}/video_name/*.jpg
  - Format: 224×224 RGB images

🔍 USAGE:
- Ensure CASME2 dataset is downloaded and organized in expected structure
- Run script: python preprocess.py
- Monitor progress with built-in progress bars and status messages
- Resulting dataset is used by load_sequences.py to create training data

📝 NOTES:
- The script maps raw CASME2 emotions to 3 categories for simplified classification
- Only processes videos with valid emotion labels (happiness, disgust, repression, surprise)
- Uses a simple face detection method that may fail on challenging frames
"""

import os
import cv2
import pandas as pd
from tqdm import tqdm

# === PATH CONFIGURATION ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
VIDEO_DIR = os.path.join(DATASET_DIR, 'CASME2')
LABEL_FILE = os.path.join(DATASET_DIR, 'CASME2-coding-20140508.xlsx')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'microexpression_processed')

# === Make sure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load and normalize label file ===
df = pd.read_excel(LABEL_FILE)
df = df.dropna(subset=['Estimated Emotion'])
df['Filename'] = df['Filename'].str.strip()
df['Estimated Emotion'] = df['Estimated Emotion'].str.lower().str.strip()

# === Map raw labels to 3-class categories ===
def map_emotion(raw):
    if raw == 'happiness':
        return 'positive'
    elif raw in ['disgust', 'repression']:
        return 'negative'
    elif raw == 'surprise':
        return 'surprise'
    else:
        return None  # skip 'others', unknown

df['Emotion'] = df['Estimated Emotion'].apply(map_emotion)
df = df.dropna(subset=['Emotion'])  # filter out undefined mappings

# === Lightweight face detection using OpenCV ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces):
        x, y, w, h = faces[0]
        x, y = max(x, 0), max(y, 0)
        face = frame[y:y+h, x:x+w]
        return cv2.resize(face, (224, 224))
    return None

# === Process only first 2 valid videos ===
processed_count = 0

for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing CASME2 videos"):
    # if processed_count >= 10:
    #     break

    video_name = row['Filename']
    emotion = row['Emotion']
    subject = row['Subject']

    video_file = video_name if video_name.endswith('.avi') else video_name + '.avi'
    video_path = os.path.join(VIDEO_DIR, f"sub{int(subject):02d}", video_file)

    if not os.path.exists(video_path):
        print(f"⚠️ Skipped (missing): {video_path}")
        continue

    save_dir = os.path.join(OUTPUT_DIR, emotion, video_name.split('.')[0])
    os.makedirs(save_dir, exist_ok=True)

    print(f"📂 Processing: {video_path} → {save_dir}")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"🖼️ Frame {frame_idx}")

        face = extract_face(frame)

        if face is not None:
            save_path = os.path.join(save_dir, f"{frame_idx:04d}.jpg")
            cv2.imwrite(save_path, face)
            print(f"✅ Saved: {save_path}")
        else:
            print(f"❌ No face detected in frame {frame_idx}")

        frame_idx += 1

    cap.release()
    processed_count += 1

print("✅ Preprocessing complete.")
print(f"📁 Output saved to: {OUTPUT_DIR}")
