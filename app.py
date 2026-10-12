
from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================
# CONFIGURATION
# =====================================

MODEL_PATH = Path("models/best_model.keras")
CLASS_PATH = Path("models/class_names.json")
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Rice Leaf AI",
    page_icon="🌾",
    layout="wide",
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def load_class_names():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

# =====================================
# GRAD-CAM
# =====================================

def find_mobilenet(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "mobilenetv2" in layer.name.lower():
                return layer
    raise ValueError("MobileNetV2 model not found.")

def generate_gradcam(image_tensor, model):
    base_model = find_mobilenet(model)

    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=base_model.get_layer("out_relu").output,
    )

    gap_layer = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    )
    dropout_layer = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dropout)
    )
    prediction_layer = model.get_layer("predictions")

    with tf.GradientTape() as tape:
        features = feature_extractor(image_tensor, training=False)
        tape.watch(features)

        x = gap_layer(features)
        x = dropout_layer(x, training=False)
        predictions = prediction_layer(x)

        class_index = tf.argmax(predictions[0])
        score = predictions[:, class_index]

    gradients = tape.gradient(score, features)

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients unavailable.")

    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        features[0] * weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], IMG_SIZE
    ).numpy().squeeze()

    return heatmap, int(class_index.numpy()), predictions.numpy()[0]


# =====================================
# USER INTERFACE
# =====================================

st.title("🌾 Rice Leaf Disease Detection")
st.caption(
    "Deep Learning • MobileNetV2 • Transfer Learning • Grad-CAM"
)

st.info(
    "Educational image-classification demo. "
    "Predictions are not a substitute for expert agricultural advice."
)

if not MODEL_PATH.is_file() or not CLASS_PATH.is_file():
    st.error(
        "Model files not found. Please train the model first "
        "and check the models/ folder."
    )
    st.stop()

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(
            original_image,
            caption="Input rice leaf image",
            width=400,
        )

        resized = original_image.resize(IMG_SIZE)
        image_array = np.asarray(resized, dtype=np.float32)
        image_tensor = preprocess_input(image_array.copy())
        image_tensor = np.expand_dims(image_tensor, axis=0)

        with st.spinner("Analyzing leaf image..."):
            heatmap, predicted_index, probabilities = generate_gradcam(
                image_tensor, model
            )

        predicted_label = class_names[predicted_index]
        score = float(probabilities[predicted_index])

        st.divider()
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Class", predicted_label)

        with col2:
            st.metric("Model Score", f"{score:.2%}")

        st.caption(
            "The model score is not a calibrated probability of disease."
        )

        st.subheader("Class Scores")
        score_order = np.argsort(probabilities)[::-1]

        for index in score_order:
            st.write(
                f"{class_names[index]}: {probabilities[index]:.2%}"
            )
            st.progress(
                float(np.clip(probabilities[index], 0.0, 1.0))
            )

        st.divider()
        st.subheader("Explainable AI — Grad-CAM")

        heatmap_col, overlay_col = st.columns(2)

        with heatmap_col:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
            ax.set_title("Grad-CAM Heatmap")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with overlay_col:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(original_image.resize(IMG_SIZE))
            ax.imshow(
                heatmap,
                cmap="jet",
                alpha=0.4,
                vmin=0,
                vmax=1,
            )
            ax.set_title("Heatmap Overlay")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        st.warning(
            "Grad-CAM highlights image regions influencing the model. "
            "It does not prove that the model has identified the actual "
            "disease symptoms."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
