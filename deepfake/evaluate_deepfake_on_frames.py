"""
🧪 evaluate_deepfake_on_frames.py
────────────────────────────────────────────
Evaluates the deepfake detection model on real test frames used in fusion.
Helps verify if the model is misclassifying real inputs.

📁 Expected Folder Structure:
dataset/
└── processed_data/
    ├── real/*/*.jpg
    └── fake/*/*.jpg

✅ Output:
- Classification Report
- Confusion Matrix
"""

import os
import cv2
import numpy as np
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# === Paths ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, 'dataset', 'processed_data')
model_path = os.path.join(project_root, 'deepfake', 'model', 'mobilenet_deepfake_model_fine_tuned.keras')

# === Load Model ===
model = load_model(model_path)

# === Load Frames ===
def load_images_from_folder(folder, label):
    image_paths = sorted(glob(os.path.join(folder, '*', '*.jpg')))
    images = []
    labels = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32))
        images.append(img)
        labels.append(label)
    return images, labels

real_imgs, real_labels = load_images_from_folder(os.path.join(dataset_dir, 'real'), label=1)
fake_imgs, fake_labels = load_images_from_folder(os.path.join(dataset_dir, 'fake'), label=0)

X = np.array(real_imgs + fake_imgs)
y_true = np.array(real_labels + fake_labels)

# === Predict ===
y_pred_probs = model.predict(X, batch_size=16)
y_pred = np.argmax(y_pred_probs, axis=1)

# === Report ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Deepfake Model Confusion Matrix")
plt.tight_layout()

# Save plot
plot_path = os.path.join(project_root, 'deepfake', 'evaluation_confusion_matrix.png')
plt.savefig(plot_path)
print(f"\n✅ Confusion matrix saved to: {plot_path}")
