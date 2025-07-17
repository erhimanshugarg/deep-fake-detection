#!/usr/bin/env python3
"""
train_xception_augmented.py
───────────────────────────
• Trains a robust REAL vs FAKE detector using Xception
• Heavy augmentation *before* the backbone + CutMix
• Progressive fine‑tuning (frozen → unfrozen backbone)
Tested with TensorFlow 2.16 / Python 3.12
"""

import os, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers, models, callbacks, optimizers, losses, metrics
from tensorflow.keras.applications import Xception
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ───────── CONFIG ────────────────────────────────────────────────
DATA_DIR     = r"E:\deepfake-detection\dataset\processed_data"   # ← adjust
IMG_SIZE     = (299, 299)
BATCH        = 24
WARM_EPOCHS  = 3
FT_EPOCHS    = 15
LR_WARM      = 1e-4
LR_FT        = 3e-5
LABEL_SMOOTH = 0.05
OUT_ROOT     = "deepfake"
SEED         = 42
# ─────────────────────────────────────────────────────────────────

tf.keras.utils.set_random_seed(SEED)
os.makedirs(f"{OUT_ROOT}/model",  exist_ok=True)
os.makedirs(f"{OUT_ROOT}/plots",  exist_ok=True)

# ───────── Data Generators (rescale only) ───────────────────────
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255., validation_split=0.2
)
train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    subset='training',  class_mode='binary', shuffle=True, seed=SEED
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    subset='validation', class_mode='binary', shuffle=False
)

# ───────── tf.data wrappers + augmentation ──────────────────────
def gen_to_dataset(gen, training=True):
    ds = tf.data.Dataset.from_generator(
        lambda: gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,),            dtype=tf.float32)
        )
    )
    if training:
        ds = ds.map(lambda x,y: (augment(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(cutmix_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# ───────── Pre‑backbone augmentation stack ──────────────────────
class RandomBrightness(layers.Layer):
    def __init__(self, delta=0.15): super().__init__(); self.d=delta
    def call(self, x): return tf.image.random_brightness(x, max_delta=self.d)

augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.2),
    RandomBrightness(0.15),
], name="augment")

# ───────── CutMix (safe for graph mode) ─────────────────────────
@tf.function
def cutmix_batch(images, labels, alpha=1.0, prob=0.5):
    """Apply CutMix to an entire batch of (N,H,W,C) images."""
    # skip with probability (1‑prob)
    if tf.random.uniform([]) > prob:
        return images, labels

    batch_size = tf.shape(images)[0]
    # Sample lambda from Beta
    lam = tf.random.uniform([], 0, 1)
    lam = tf.maximum(lam, 1 - lam)           # ensure >= 0.5
    r_x = tf.random.uniform([], 0, IMG_SIZE[1], tf.int32)
    r_y = tf.random.uniform([], 0, IMG_SIZE[0], tf.int32)
    r_w = tf.cast(IMG_SIZE[1] * tf.math.sqrt(1 - lam), tf.int32)
    r_h = tf.cast(IMG_SIZE[0] * tf.math.sqrt(1 - lam), tf.int32)

    x1 = tf.clip_by_value(r_x - r_w // 2, 0, IMG_SIZE[1])
    y1 = tf.clip_by_value(r_y - r_h // 2, 0, IMG_SIZE[0])
    x2 = tf.clip_by_value(r_x + r_w // 2, 0, IMG_SIZE[1])
    y2 = tf.clip_by_value(r_y + r_h // 2, 0, IMG_SIZE[0])

    # Shuffle indices for pairing
    idx = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, idx)
    shuffled_labels = tf.gather(labels, idx)

    # Apply CutMix
    mask = tf.ones((y2 - y1, x2 - x1, 3))
    pad_top    = y1
    pad_bottom = IMG_SIZE[0] - y2
    pad_left   = x1
    pad_right  = IMG_SIZE[1] - x2
    mask = tf.pad(mask, [[pad_top, pad_bottom],
                         [pad_left, pad_right], [0,0]])
    mask = tf.cast(mask, images.dtype)

    images = images * (1 - mask) + shuffled_images * mask
    labels = lam * labels + (1 - lam) * shuffled_labels
    return images, labels

# ───────── Build tf.data pipelines ──────────────────────────────
train_ds = gen_to_dataset(train_gen, training=True)
val_ds   = gen_to_dataset(val_gen,   training=False)

# ───────── Class weights to address imbalance ───────────────────
cw = compute_class_weight(class_weight='balanced',
                          classes=np.unique(train_gen.classes),
                          y=train_gen.classes)
class_w = {0: cw[0], 1: cw[1]}
print("Class weights:", class_w)

# ───────── Model Definition ─────────────────────────────────────
base = Xception(include_top=False, weights='imagenet',
                input_shape=IMG_SIZE+(3,), pooling='avg')
base.trainable = False   # warm‑up frozen

inputs = layers.Input(shape=IMG_SIZE+(3,))
x = augment(inputs)                                   # augmentation
x = tf.keras.applications.xception.preprocess_input(x)
x = base(x, training=False)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(LR_WARM),
    loss=losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy', metrics.AUC(name='auc', curve='ROC')]
)

stamp = datetime.now().strftime("%Y%m%d_%H%M")
ckpt_path = f"{OUT_ROOT}/model/xception_best_{stamp}.keras"

cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max',
                              save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max',
                                factor=0.3, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_auc', mode='max',
                            patience=6, restore_best_weights=True, verbose=1)
]

# ───────── Stage 1: Warm‑up ─────────────────────────────────────
print("\n🚀 Stage 1 – warm‑up (frozen backbone)")
model.fit(train_ds, validation_data=val_ds,
          epochs=WARM_EPOCHS, class_weight=class_w, callbacks=cbs, verbose=2)

# ───────── Stage 2: Fine‑tune ───────────────────────────────────
base.trainable = True
model.compile(
    optimizer=optimizers.Adam(LR_FT),
    loss=losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['accuracy', metrics.AUC(name='auc', curve='ROC')]
)
print("\n🚀 Stage 2 – fine‑tune (unfrozen backbone)")
model.fit(train_ds, validation_data=val_ds,
          epochs=FT_EPOCHS, class_weight=class_w, callbacks=cbs, verbose=2)

# ───────── Evaluation @best checkpoint ──────────────────────────
best = tf.keras.models.load_model(ckpt_path, compile=False)
probs  = best.predict(val_ds, verbose=0).ravel()
labels = np.concatenate([y for _,y in val_ds])

fpr, tpr, thr = roc_curve(labels, probs)
best_th = thr[np.argmax(tpr - fpr)]
print(f"\n🔧 Suggested threshold ≈ {best_th:.3f}")

pred = (probs >= best_th).astype(int)
print("\n📊 Classification report @best_th:")
print(classification_report(labels, pred, target_names=['Fake','Real'], digits=4))

cm = confusion_matrix(labels, pred)
plt.figure(figsize=(4,3.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
plt.xlabel('Pred'); plt.ylabel('Actual')
plt.title('Xception + Augmentation')
plt.tight_layout()
cm_path = f"{OUT_ROOT}/plots/xception_confmat_{stamp}.png"
plt.savefig(cm_path)
print("\n✅ Confusion matrix →", cm_path)
print("✅ Best model →", ckpt_path)
