import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

# --------------------------
# CONFIGURABLE PARAMETERS
# --------------------------
INPUT_DIR = "../dataset/FFPP"  # Folder with 'real/' and 'fake/' subfolders
OUTPUT_DIR = "../dataset/processed_data"
IMG_SIZE = (224, 224)
FRAME_INTERVAL = 3               # Sample every 3rd frame
MAX_VIDEOS_PER_CLASS = 10        # Baby step: process only first 3 videos per class

# --------------------------
# FACE DETECTOR INIT
# --------------------------
detector = MTCNN()

def extract_faces_from_video(video_path, label, video_name):
    save_dir = os.path.join(OUTPUT_DIR, label, video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_id % FRAME_INTERVAL == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)
            if faces:
                x, y, w, h = faces[0]['box']
                x, y = max(0, x), max(0, y)
                cropped = rgb[y:y+h, x:x+w]
                resized = cv2.resize(cropped, IMG_SIZE)
                out_path = os.path.join(save_dir, f"{saved_count:05d}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
                saved_count += 1

        frame_id += 1

    cap.release()

def process_all_videos(max_videos_per_class=MAX_VIDEOS_PER_CLASS):
    for label in ['real', 'fake']:
        input_label_path = os.path.join(INPUT_DIR, label)
        if not os.path.exists(input_label_path):
            print(f"Directory not found: {input_label_path}")
            continue

        video_files = sorted([f for f in os.listdir(input_label_path) if f.endswith(".mp4")])
        video_files = video_files[:max_videos_per_class]

        for video_file in tqdm(video_files, desc=f"Processing {label} videos"):
            video_path = os.path.join(input_label_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            extract_faces_from_video(video_path, label, video_name)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_all_videos()
    print("✅ Preprocessing complete.")
