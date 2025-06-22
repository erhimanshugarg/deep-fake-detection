import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model/mobilenet_deepfake_model.keras"
DATA_DIR = "../dataset/processed_data"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# Load model
# -----------------------------
print("📦 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Prepare validation data
# -----------------------------
print("🔄 Preparing validation data...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# -----------------------------
# Predict on validation set
# -----------------------------
print("🔍 Predicting on validation set...")
pred_probs = model.predict(val_gen)
pred_labels = (pred_probs > 0.5).astype(int).flatten()

true_labels = val_gen.classes
class_names = list(val_gen.class_indices.keys())

# -----------------------------
# Metrics & Evaluation
# -----------------------------
print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save plot with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cm_filename = f"confusion_matrix_{timestamp}.png"
plt.tight_layout()
plt.savefig(cm_filename)
print(f"\n✅ Confusion matrix saved as {cm_filename}")
plt.show()
