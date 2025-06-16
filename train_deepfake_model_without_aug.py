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
DATA_DIR = "processed_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "mobilenet_deepfake_model.keras"  # ✅ Save in .keras format

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
