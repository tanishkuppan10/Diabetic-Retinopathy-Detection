"""
utils.py — Helper functions for the Diabetic Retinopathy Detection Dashboard
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_PATH    = "best_efficientnet_model.keras"
IMG_SIZE      = 224
CLASS_NAMES   = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# Alphabetical order matches Keras flow_from_dataframe default
CLASS_LABELS  = {
    0: "Mild",
    1: "Moderate",
    2: "No_DR",
    3: "Proliferate_DR",
    4: "Severe",
}

SEVERITY_MAP = {
    "No_DR":          {"level": 0, "color": "#22c55e", "icon": "✅", "desc": "No Diabetic Retinopathy — Healthy retina."},
    "Mild":           {"level": 1, "color": "#84cc16", "icon": "🟡", "desc": "Mild NPDR — Microaneurysms only."},
    "Moderate":       {"level": 2, "color": "#f59e0b", "icon": "🟠", "desc": "Moderate NPDR — More than microaneurysms, less than severe."},
    "Severe":         {"level": 3, "color": "#ef4444", "icon": "🔴", "desc": "Severe NPDR — No PDR; extensive intraretinal abnormalities."},
    "Proliferate_DR": {"level": 4, "color": "#7c3aed", "icon": "🚨", "desc": "Proliferative DR — Neovascularisation present. Immediate care needed."},
}

DATASET_COUNTS = {
    "No_DR":          1805,
    "Moderate":       999,
    "Mild":           370,
    "Proliferate_DR": 295,
    "Severe":         193,
}

# Saved classification report values from results.txt
CLASSIFICATION_REPORT = {
    "Class":     ["Mild",  "Moderate", "No_DR", "Proliferate_DR", "Severe"],
    "Precision": [0.67,    0.64,       0.93,    0.50,              0.44],
    "Recall":    [0.05,    0.85,       0.97,    0.57,              0.21],
    "F1-Score":  [0.10,    0.73,       0.95,    0.53,              0.29],
    "Support":   [37,      100,        181,     30,                19],
}

# Confusion matrix from results.txt (true × predicted)
# Order: Mild, Moderate, No_DR, Proliferate_DR, Severe
CONFUSION_MATRIX = np.array([
    [2,   17,  15,  3,   0],   # Mild
    [0,   85,  12,  3,   0],   # Moderate
    [0,   6,   175, 0,   0],   # No_DR
    [0,   7,   6,   17,  0],   # Proliferate_DR
    [0,   9,   3,   3,   4],   # Severe
])


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained EfficientNetB3 model (cached for performance)."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Resize, convert to BGR, apply CLAHE, convert back.
    Matches the preprocessing pipeline from image_preprocessing.ipynb.
    Returns a (1, IMG_SIZE, IMG_SIZE, 3) numpy array ready for inference.
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # CLAHE on L-channel (same as preprocessing notebook)
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge((l, a, b))
    img     = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img.astype(np.float32), axis=0)
    return img_array


def pil_to_display(pil_image: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Return a display-ready RGB numpy array (no CLAHE) from a PIL image."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (size, size))
    return img


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(model, pil_image: Image.Image):
    """
    Run inference on a PIL image.
    Returns: (predicted_class_name: str, confidence: float, all_probs: np.ndarray)
    """
    img_array = preprocess_image(pil_image)
    preds     = model.predict(img_array, verbose=0)[0]          # shape: (5,)
    idx       = int(np.argmax(preds))
    class_name = CLASS_LABELS[idx]
    confidence = float(preds[idx])
    return class_name, confidence, preds


# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────
def make_gradcam_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str = "top_conv") -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the top predicted class.
    Returns a (IMG_SIZE, IMG_SIZE) heatmap array (values 0-1).
    """
    # Build a model that outputs (feature maps, final predictions)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs         = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        pred_index     = tf.argmax(predictions[0])
        class_channel  = predictions[:, pred_index]

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap).numpy()
    heatmap      = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap


def overlay_gradcam(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Superimpose the Grad-CAM heatmap over the original image.
    Returns an RGB numpy array.
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    jet_colormap    = cm.get_cmap("jet")
    jet_heatmap     = jet_colormap(heatmap_uint8)
    jet_heatmap     = np.uint8(jet_heatmap * 255)[:, :, :3]

    superimposed = cv2.addWeighted(img, 1 - alpha, jet_heatmap, alpha, 0)
    return superimposed


def get_gradcam_overlay(model, pil_image: Image.Image):
    """
    Full Grad-CAM pipeline.
    Returns the superimposed RGB image as a numpy array, or None on failure.
    """
    try:
        img_array = preprocess_image(pil_image)
        heatmap   = make_gradcam_heatmap(model, img_array)
        overlay   = overlay_gradcam(pil_image, heatmap)
        return overlay
    except Exception:
        return None


# ─────────────────────────────────────────────
# Sample images helper
# ─────────────────────────────────────────────
def get_sample_images(class_name: str, n: int = 4, base_dir: str = "colored_images") -> list:
    """Return up to n PIL Images from a given class folder."""
    folder = os.path.join(base_dir, class_name)
    if not os.path.exists(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files = files[:n]
    images = []
    for f in files:
        try:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images
