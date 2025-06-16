import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import class_weight
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "processed_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "mobilenet_deepfake_model_finetuned.keras"

# -----------------------------
# DATA GENERATORS WITH AUGMENTATION
# -----------------------------
print("🔄 Loading data with augmentations...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
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

# -----------------------------
# CLASS WEIGHTS (optional but helps balance)
# -----------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"📊 Class weights: {class_weights_dict}")

# -----------------------------
# MODEL SETUP + FINE-TUNING
# -----------------------------
print("🧠 Building fine-tuned MobileNetV2 model...")
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = True  # ✅ Fine-tune all layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# TRAINING
# -----------------------------
print("🚀 Starting fine-tuned training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Fine-tuned model saved as {MODEL_PATH}")

# Create 'plots' directory if it doesn't exist
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------
# PLOT HISTORY
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_filename = f"finetuned_training_plot_{timestamp}.png"
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Fine-tuned Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(plots_dir, plot_filename)
plt.savefig(plot_path)
print(f"📊 Plot saved as {plot_path}")
print(f"📊 Accuracy plot saved as {plot_filename}")
plt.show()
