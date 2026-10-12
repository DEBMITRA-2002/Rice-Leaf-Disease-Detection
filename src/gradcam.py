
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/best_model.keras")
CLASS_PATH = Path("models/class_names.json")
OUTPUT_DIR = Path("outputs")

IMG_SIZE = (224, 224)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD MODEL AND CLASS NAMES
# ==========================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

if not CLASS_PATH.is_file():
    raise FileNotFoundError(
        f"Class names not found: {CLASS_PATH.resolve()}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully.")
print("Classes:", class_names)


# ==========================================
# 3. FIND MOBILENETV2 FEATURE EXTRACTOR
# ==========================================

def find_mobilenet(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "mobilenetv2" in layer.name.lower():
                return layer

    raise ValueError(
        "Nested MobileNetV2 was not found in the saved model."
    )


# ==========================================
# 4. GENERATE GRAD-CAM
# ==========================================

def make_gradcam_heatmap(image_tensor, model):

    base_model = find_mobilenet(model)

    # Build a feature extractor from MobileNetV2's own graph
    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer("out_relu").output,
        name="gradcam_feature_extractor",
    )

    # Locate the trained classification head
    gap_layer = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    )

    dropout_layer = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    )

    prediction_layer = model.get_layer("predictions")

    # GradientTape must record the operations from feature maps
    # through the trained classification head.
    with tf.GradientTape() as tape:

        feature_maps = feature_extractor(
            image_tensor,
            training=False,
        )

        tape.watch(feature_maps)

        x = gap_layer(feature_maps)
        x = dropout_layer(x, training=False)
        predictions = prediction_layer(x)

        class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. Cannot generate Grad-CAM."
        )

    # Average gradients over the height and width
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    # Weighted sum of feature maps
    heatmap = tf.reduce_sum(
        feature_maps[0]
        * channel_weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    # Keep positive activations
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize the heatmap
    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / (
            maximum + tf.keras.backend.epsilon()
        ),
        tf.zeros_like(heatmap),
    )

    return (
        heatmap.numpy(),
        int(class_index.numpy()),
        predictions.numpy()[0],
    )


# ==========================================
# 5. LOAD AND PREPROCESS INPUT IMAGE
# ==========================================

def load_image(image_path):

    path = Path(image_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path.resolve()}"
        )

    with Image.open(path) as img:
        original_image = img.convert("RGB")
        resized_image = original_image.resize(IMG_SIZE)

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    processed_image = preprocess_input(
        image_array.copy()
    )

    image_tensor = np.expand_dims(
        processed_image,
        axis=0,
    )

    return original_image, image_tensor


# ==========================================
# 6. PREDICTION AND VISUALIZATION
# ==========================================

def main():

    image_path = input(
        "\nEnter the path of a rice leaf image: "
    ).strip().strip('"').strip("'")

    original_image, image_tensor = load_image(image_path)

    heatmap, predicted_index, probabilities = (
        make_gradcam_heatmap(image_tensor, model)
    )

    predicted_label = class_names[predicted_index]
    confidence_score = float(probabilities[predicted_index])

    print("\n========== PREDICTION ==========")
    print("Predicted class:", predicted_label)
    print(f"Model score: {confidence_score:.2%}")
    print(
        "Note: This score is not a calibrated probability "
        "of disease."
    )

    # Resize heatmap to the image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE,
    ).numpy().squeeze()

    # Create the three-panel visualization
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap_resized,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(
        original_image.resize(IMG_SIZE)
    )
    axes[2].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Prediction: {predicted_label}"
    )
    axes[2].axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "gradcam_result.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print("\nGrad-CAM image saved successfully.")
    print("Output:", output_path.resolve())


# ==========================================
# 7. RUN
# ==========================================

if __name__ == "__main__":
    main()
