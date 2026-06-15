"""
train_model.py
==============
Klasifikasi Nominal Uang Kertas Rupiah menggunakan MobileNetV2 Transfer Learning
Dataset: dataset/train/ (split otomatis 80% train, 20% val+test)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ─── KONFIGURASI ────────────────────────────────────────────────────────────────
TRAIN_DIR     = "dataset/train"
MODEL_PATH    = "model/rupiah_cnn.h5"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.0001

CLASS_NAMES = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
NUM_CLASSES = len(CLASS_NAMES)

os.makedirs("model", exist_ok=True)

# ─── DATA GENERATOR (split 80% train, 20% val) ──────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"\n[INFO] Class indices : {train_generator.class_indices}")
print(f"[INFO] Train samples : {train_generator.samples}")
print(f"[INFO] Val samples   : {val_generator.samples}\n")

# Simpan class indices untuk Flask
with open("model/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

# ─── MODEL MobileNetV2 ────────────────────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─── CALLBACKS ───────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, verbose=1, min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, verbose=1
    )
]

# ─── PHASE 1: TRAINING (base frozen) ─────────────────────────────────────────────
print("\n[PHASE 1] Training head layers...\n")
history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─── PHASE 2: FINE-TUNING ────────────────────────────────────────────────────────
print("\n[PHASE 2] Fine-tuning...\n")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)

# ─── EVALUASI ────────────────────────────────────────────────────────────────────
print("\n[INFO] Evaluasi pada data validasi...")
val_loss, val_acc = model.evaluate(val_generator)
print(f"\nAkurasi Validasi : {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Loss Validasi    : {val_loss:.4f}")

# Classification Report
y_pred = np.argmax(model.predict(val_generator), axis=1)
y_true = val_generator.classes
idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
labels = [idx_to_class[i] for i in sorted(idx_to_class)]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# ─── CONFUSION MATRIX ────────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Prediksi")
plt.ylabel("Label Asli")
plt.title("Confusion Matrix - Klasifikasi Uang Kertas Rupiah")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
plt.show()

# ─── GRAFIK TRAINING ─────────────────────────────────────────────────────────────
acc     = history1.history["accuracy"]     + history2.history["accuracy"]
val_acc_h = history1.history["val_accuracy"] + history2.history["val_accuracy"]
loss    = history1.history["loss"]         + history2.history["loss"]
val_loss_h = history1.history["val_loss"]  + history2.history["val_loss"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(acc, label="Train Accuracy")
axes[0].plot(val_acc_h, label="Val Accuracy")
axes[0].set_title("Akurasi Model")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(loss, label="Train Loss")
axes[1].plot(val_loss_h, label="Val Loss")
axes[1].set_title("Loss Model")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("model/training_history.png", dpi=150)
plt.show()

print(f"\n[DONE] Model disimpan di: {MODEL_PATH}")