"""
────────────────────────────────────────────────────────────────────
🎯 Purpose: Train Micro-Expression Classifier (CNN + LSTM)
with TensorBoard Logging, Model Checkpointing, Plots, and Confusion Matrix
────────────────────────────────────────────────────────────────────

Loads:
- X_sequences.npy: shape (num_samples, seq_len, 224, 224, 3)
- y_labels.npy: shape (num_samples) with values {0,1,2}

Trains a deep learning model to classify micro-expressions
as positive, negative, or surprise using CNN + LSTM architecture.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from datetime import datetime as dt

# === Configuration ===
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'microexpression_processed'))
X_PATH = os.path.join(DATASET_DIR, 'X_sequences.npy')
Y_PATH = os.path.join(DATASET_DIR, 'y_labels.npy')

SEQ_LENGTH = 16
IMG_SHAPE = (224, 224, 3)
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 30

# === Output Folders ===
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
CM_DIR = os.path.join(os.path.dirname(__file__), 'confusion-matrix')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# === Load Data ===
X = np.load(X_PATH)
y = np.load(Y_PATH)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# Print class distribution in validation
val_labels = np.argmax(y_val, axis=1)
unique, counts = np.unique(val_labels, return_counts=True)
class_names = ['positive', 'negative', 'surprise']
print("\n📊 Validation Set Class Distribution:")
for label, count in zip(unique, counts):
    print(f"{class_names[label]:>9}: {count} samples")

# === Compute Class Weights ===
class_weights_array = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights_array))

# === Build CNN + LSTM Model ===
base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE, pooling='avg')
for layer in base_cnn.layers:
    layer.trainable = False  # Freeze base CNN layers

input_layer = Input(shape=(SEQ_LENGTH, *IMG_SHAPE))
x = TimeDistributed(base_cnn)(input_layer)
x = LSTM(128)(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===

# EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# TensorBoard
log_dir = os.path.join("logs", "fit", dt.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# ModelCheckpoint
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "best_model_{epoch:02d}_{val_loss:.4f}.keras")

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# === Train Model ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[early_stop, tensorboard_callback, checkpoint_callback]
)

# === Plot Training Curves ===
def plot_training_curves(history, output_path):
    plt.figure(figsize=(10, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    timestamp = dt.now().strftime("%Y%m%d_%H%M")
    fig_path = os.path.join(output_path, f"training_curves_{timestamp}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"📉 Training curves saved to: {fig_path}")

plot_training_curves(history, PLOT_DIR)

# === Evaluate Model ===
y_pred = model.predict(X_val)
y_true = np.argmax(y_val, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_class, target_names=class_names))

# === Confusion Matrix Plot ===
def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    timestamp = dt.now().strftime("%Y%m%d_%H%M")
    fig_path = os.path.join(output_path, f"confusion_matrix_{timestamp}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"🔲 Confusion matrix saved to: {fig_path}")

plot_confusion_matrix(y_true, y_pred_class, class_names, CM_DIR)

# Ensure the models directory exists
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

# === Save Final Model ===
timestamp = dt.now().strftime("%Y%m%d_%H%M")
model_filename = f"microexpression_model_{timestamp}.keras"
model_path = os.path.join(models_dir, model_filename)
model.save(model_filename)
print(f"✅ Final model saved to: {model_filename}")
