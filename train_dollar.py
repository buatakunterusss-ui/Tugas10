"""
train_dollar.py
===============
Klasifikasi Nominal Uang Kertas Dollar USD menggunakan MobileNetV2
Dataset: dataset_dollar/train/
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

TRAIN_DIR  = "dataset_dollar/train"
MODEL_PATH = "model/dollar_cnn.h5"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 0.0001

os.makedirs("model", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training", shuffle=True
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation", shuffle=False
)

print(f"\n[INFO] Classes  : {train_gen.class_indices}")
print(f"[INFO] Train    : {train_gen.samples}")
print(f"[INFO] Val      : {val_gen.samples}\n")

with open("model/dollar_class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

NUM_CLASSES = len(train_gen.class_indices)

base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(LR),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7),
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

print("\n[PHASE 1] Training head layers...\n")
h1 = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

print("\n[PHASE 2] Fine-tuning...\n")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(LR/10),
              loss="categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=callbacks)

val_loss, val_acc = model.evaluate(val_gen)
print(f"\nAkurasi Dollar Model: {val_acc*100:.2f}%")

y_pred = np.argmax(model.predict(val_gen), axis=1)
y_true = val_gen.classes
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
labels = [idx_to_class[i] for i in sorted(idx_to_class)]
print(classification_report(y_true, y_pred, target_names=labels))

print(f"\n[DONE] Model disimpan di: {MODEL_PATH}")
