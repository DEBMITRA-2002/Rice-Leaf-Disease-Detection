
from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# -----------------------------
# 1. Configuration
# -----------------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

TRAIN_DIR = Path("data/processed/train")
VAL_DIR = Path("data/processed/val")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

for folder in [TRAIN_DIR, VAL_DIR]:
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder.resolve()}")

# -----------------------------
# 2. Load image datasets
# -----------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_data = val_datagen.flow_from_directory(
    str(VAL_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

if train_data.class_indices != val_data.class_indices:
    raise ValueError("Training and validation class mappings do not match.")

num_classes = train_data.num_classes
class_names = list(train_data.class_indices.keys())

print("\nClasses:", class_names)
print("Number of classes:", num_classes)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)

with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# -----------------------------
# 3. Load pretrained MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# Freeze pretrained layers for initial training
base_model.trainable = False

# -----------------------------
# 4. Build classification head
# -----------------------------
inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="predictions",
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 5. Callbacks
# -----------------------------
callbacks = [
    ModelCheckpoint(
        str(MODEL_DIR / "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# -----------------------------
# 6. Train classification head
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save final model as well
model.save(MODEL_DIR / "final_model.keras")

# -----------------------------
# 7. Plot training history
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
plt.close()

print("\nTraining complete!")
print("Best model:", MODEL_DIR / "best_model.keras")
print("Final model:", MODEL_DIR / "final_model.keras")
print("Class mapping:", MODEL_DIR / "class_names.json")
print("Training graph:", OUTPUT_DIR / "training_history.png")
