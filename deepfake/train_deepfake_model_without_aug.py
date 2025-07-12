"""
🔍 Deepfake Detection Model Training (Minimal Augmentation)
───────────────────────────────────────────────────────────────────────────────

This script trains a lightweight deepfake detection model based on MobileNetV2
with minimal data augmentation. It's designed for quick training cycles and
baseline model creation.

📋 FUNCTIONALITY:
- Loads processed face images from the dataset directory
- Applies minimal data augmentation (only rotation and horizontal flip)
- Builds a MobileNetV2-based model with frozen base layers (feature extraction only)
- Trains the model for a short period (10 epochs)
- Saves the trained model in .keras format
- Generates and saves training accuracy plots

🧠 MODEL ARCHITECTURE:
- Base: Pre-trained MobileNetV2 (frozen weights from ImageNet)
- Feature Extraction: GlobalAveragePooling2D
- Classification Head: 128-unit dense layer with ReLU + 30% dropout
- Output: Single sigmoid unit for binary classification (real/fake)
- Optimizer: Adam with 1e-4 learning rate
- Loss: Binary Cross-Entropy

📊 INPUT/OUTPUT:
- Input: Processed face images in ../dataset/processed_data/{real,fake}/
- Output:
  - Trained model: model/mobilenet_deepfake_model.keras
  - Training plot: plots/training_plot_<timestamp>.png
  - Console output: Training progress and file paths

🔍 USAGE:
- Ensure processed dataset exists in the expected directory
- Run script: python train_deepfake_model_without_aug.py
- Monitor training progress in console output
- Review accuracy plot after training completes

📝 COMPARISON TO FULL TRAINING:
This script differs from train_deepfake_model.py in several key ways:
1. Uses minimal data augmentation (vs. extensive augmentation)
2. Keeps base model frozen (vs. fine-tuning all layers)
3. Uses higher learning rate (1e-4 vs. 1e-5)
4. Trains for fewer epochs (10 vs. 20)
5. Doesn't use class weighting for imbalanced data
6. Doesn't generate confusion matrix for evaluation

This version is ideal for quick iterations, baseline comparisons, or when
computational resources are limited.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "../dataset/processed_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "model/mobilenet_deepfake_model.keras"  # ✅ Save in .keras format

# -----------------------------
# DATA LOADING
# -----------------------------
print("🔄 Loading data...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# -----------------------------
# MODEL BUILDING
# -----------------------------
print("🧠 Building model...")
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze base layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAINING
# -----------------------------
print("🚀 Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(MODEL_PATH)  # ✅ Saves as .keras directory
print(f"✅ Model saved to {MODEL_PATH}")




# -----------------------------
# PLOT HISTORY
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Create plots directory if it doesn't exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the plot in the 'plots' directory
plot_filename = f"training_plot_{timestamp}.png"
plot_path = os.path.join(plots_dir, plot_filename)
plt.savefig(plot_path)
print(f"📊 Plot saved as {plot_path}")
plt.show()
