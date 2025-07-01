"""
🎯 Purpose: Train Deepfake Detection Model v3 (MobileNetV2)
🧠 Uses:
- Strong augmentations
- Class weighting
- Confusion Matrix + Accuracy plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === CONFIG
DATA_DIR = "../dataset/processed_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "model/mobilenet_deepfake_model_v3.keras"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# === Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"📊 Class Weights: {class_weights_dict}")

# === Model Architecture
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Train Model
print("🚀 Training MobileNetV2...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict
)

# === Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")

# === Plot Accuracy
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(PLOTS_DIR, f"deepfake_acc_plot_{timestamp}.png")

plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("📈 Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
print(f"📊 Accuracy plot saved to: {plot_path}")

# === Confusion Matrix
print("📊 Evaluating on validation set...")
val_gen.reset()
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

report = classification_report(y_true, y_pred, target_names=["Fake", "Real"])
print("\n📊 Classification Report:\n", report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path)
print(f"🔲 Confusion matrix saved as: {cm_path}")
