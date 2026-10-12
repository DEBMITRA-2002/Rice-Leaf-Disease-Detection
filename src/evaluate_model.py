from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

TEST_DIR = Path("data/processed/test")
MODEL_PATH = Path("models/best_model.keras")
CLASS_PATH = Path("models/class_names.json")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for path in [TEST_DIR, MODEL_PATH, CLASS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file/folder missing: {path}")

with open(CLASS_PATH) as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=16,
    label_mode="int",
    shuffle=False,
)

if test_data.class_names != class_names:
    raise ValueError(
        f"Class mismatch: model={class_names}, test={test_data.class_names}"
    )

# MobileNetV2 expects pixel values scaled to [-1, 1].
# This is equivalent to applications.mobilenet_v2.preprocess_input.
test_data = test_data.map(
    lambda images, labels: (
        tf.keras.applications.mobilenet_v2.preprocess_input(images),
        labels,
    )
)

y_true = np.concatenate([labels.numpy() for _, labels in test_data])
probabilities = model.predict(test_data, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\n===== TEST RESULTS =====")
print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title("Rice Leaf Disease - Confusion Matrix")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")
