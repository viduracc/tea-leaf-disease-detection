import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
import os
import json
import scipy.stats
from pathlib import Path
from tensorflow.keras.applications.efficientnet import preprocess_input

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "v5.1.keras")
BACKBONE_PATH = os.path.join(PROJECT_ROOT, "models", "backbone_extractor.keras")
CLASS_INDICES_PATH = os.path.join(PROJECT_ROOT, "models", "class_indices.json")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "app_data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
IMG_SIZE = (300, 300)
EXPAND_FACTOR = 1.4
MIN_LEAF_RATIO = 0.10       # reject if leaf < 10% of image
ENTROPY_THRESH = 0.60       # reject if normalised entropy > 0.60
HIGH_CONF_THRESH = 0.85     # green
MED_CONF_THRESH = 0.50      # amber (below = yellow/low)

DEFAULT_CLASS_NAMES = [
    "Algal Leaf", "Healthy", "Red Leaf Spot", "White Spot"
]

REJECTION_MESSAGES = {
    "no_leaf": ("No tea leaf detected. Please upload a clear image of a "
                "single tea leaf against a plain background."),
    "uncertain": ("The model is too uncertain to make a prediction. Please "
                  "retake the photo in better lighting with the leaf clearly "
                  "visible."),
    "low_conf": ("Low confidence prediction. Consider retaking the photo in "
                 "better lighting or consulting an agricultural expert."),
    "medium_conf": "Moderate confidence prediction.",
}

DISCLAIMER = (
    "Disclaimer: This system is a research tool. "
    "Always consult a qualified expert for treatment decisions."
)


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma=2.0, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_t = y_true * (1 - self.label_smoothing) + self.label_smoothing / n
        y_p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_t * tf.math.log(y_p)
        pt = tf.reduce_sum(y_t * y_p, axis=-1, keepdims=True)
        return tf.reduce_mean(
            tf.pow(1 - pt, self.gamma) * tf.reduce_sum(ce, axis=-1))

    def get_config(self):
        c = super().get_config()
        c.update({"gamma": self.gamma, "label_smoothing": self.label_smoothing})
        return c



def segment_leaf(img_bgr, expand=EXPAND_FACTOR, min_area=500):
    """
    HSV heuristic leaf segmentation.
    Returns (bbox, crop, mask, leaf_ratio).
    leaf_ratio = fraction of image covered by detected leaf.
    Returns (None, full_img, blank_mask, 0.0) if no leaf found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([90, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([30, 255, 255]))
    mask = cv2.bitwise_or(mask, mask2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img_bgr.copy(), np.zeros(img_bgr.shape[:2], np.uint8), 0.0

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, img_bgr.copy(), np.zeros(img_bgr.shape[:2], np.uint8), 0.0

    ratio = float((mask > 0).sum()) / mask.size
    if ratio > 0.85:
        # Likely uniform background leaked into HSV range — use Otsu
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k2 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, img_bgr.copy(), mask, 0.0
        c = max(contours, key=cv2.contourArea)

    leaf_ratio = float(cv2.contourArea(c)) / (img_bgr.shape[0] * img_bgr.shape[1])

    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w / 2, y + h / 2
    x1 = int(max(0, cx - w * expand / 2))
    y1 = int(max(0, cy - h * expand / 2))
    x2 = int(min(img_bgr.shape[1], cx + w * expand / 2))
    y2 = int(min(img_bgr.shape[0], cy + h * expand / 2))

    crop = img_bgr[y1:y2, x1:x2].copy()
    return (x1, y1, x2, y2), crop, mask, leaf_ratio


def neutralize_background(img_bgr, mask):
    """Set background pixels (mask==0) to gray 128."""
    result = img_bgr.copy()
    result[mask == 0] = 128
    return result



def preprocess_image(image_bytes: bytes):
    """
    Full v5.1 preprocessing from raw image bytes.
    Returns (preprocessed_array, crop_bgr, leaf_mask, leaf_ratio).
    preprocessed_array: shape (300,300,3) float32, ready for model.
    crop_bgr: the neutralized crop shown to the model (for display).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (600, 600))

    bbox, crop, mask, leaf_ratio = segment_leaf(img_bgr)

    crop_mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    crop_neutral = neutralize_background(crop, crop_mask)

    crop_resized = cv2.resize(crop_neutral, IMG_SIZE[::-1])
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    preprocessed = preprocess_input(crop_rgb.astype(np.float32))

    leaf_mask_model = cv2.resize(
        crop_mask, IMG_SIZE[::-1], interpolation=cv2.INTER_NEAREST)

    return preprocessed, crop_resized, leaf_mask_model, leaf_ratio


def get_model_input_for_display(image_bytes: bytes) -> np.ndarray:
    """
    Returns the preprocessed crop as RGB uint8 (no preprocess_input),
    suitable for display.
    """
    preprocessed, crop_bgr, _, _ = preprocess_image(image_bytes)
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)



def interpret_confidence(probs, classes):
    """
    Returns a result dict with status, prediction, confidence and message.

    Status tiers:
      high     (>=0.85) — show result + GradCAM
      medium   (>=0.50) — show result + GradCAM + top-2 bar chart
      low      (<0.50)  — show result + GradCAM + all-class bar chart
      rejected          — show message only, no GradCAM
    """
    n_classes = len(classes)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    entropy = float(scipy.stats.entropy(probs)) / np.log(n_classes)

    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}

    if entropy > ENTROPY_THRESH:
        return {
            "status": "rejected",
            "prediction": None,
            "confidence": None,
            "all_probs": None,
            "message": REJECTION_MESSAGES["uncertain"],
            "disclaimer": DISCLAIMER,
        }

    if conf >= HIGH_CONF_THRESH:
        return {
            "status": "high",
            "prediction": classes[idx],
            "confidence": conf,
            "all_probs": None,
            "message": f"Detected: {classes[idx]}",
            "disclaimer": DISCLAIMER,
        }

    if conf >= MED_CONF_THRESH:
        top2 = dict(sorted(all_probs.items(),
                           key=lambda x: x[1], reverse=True)[:2])
        return {
            "status": "medium",
            "prediction": classes[idx],
            "confidence": conf,
            "all_probs": top2,
            "message": REJECTION_MESSAGES["medium_conf"],
            "disclaimer": DISCLAIMER,
        }

    return {
        "status": "low",
        "prediction": classes[idx],
        "confidence": conf,
        "all_probs": all_probs,
        "message": REJECTION_MESSAGES["low_conf"],
        "disclaimer": DISCLAIMER,
    }



def get_gradcam_masked(backbone_ext, head_layers, img_pre,
                       crop_bgr, leaf_mask, class_idx):
    """
    GradCAM with background activations zeroed out.
    Returns (overlay_rgb, heatmap_normalized) or (crop_rgb, blank) on failure.
    """
    img_t = tf.cast(img_pre[np.newaxis], tf.float32)

    with tf.GradientTape() as tape:
        conv_out, bb_out = backbone_ext(img_t, training=False)
        tape.watch(conv_out)
        x = bb_out
        for layer in head_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        loss = x[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        h, w = crop_bgr.shape[:2]
        blank = np.zeros((h, w), np.float32)
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), blank

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
    heatmap = np.maximum(heatmap, 0)

    h, w = crop_bgr.shape[:2]
    hmap_r = cv2.resize(heatmap, (w, h))

    # Zero out background activations so Grad-CAM only highlights the leaf
    hmap_r[leaf_mask == 0] = 0
    if hmap_r.max() > 0:
        hmap_r /= hmap_r.max()

    jet = cv2.applyColorMap(np.uint8(255 * hmap_r), cv2.COLORMAP_JET)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(crop_rgb, 0.6,
                              cv2.cvtColor(jet, cv2.COLOR_BGR2RGB), 0.4, 0)
    return overlay, hmap_r


def load_class_names() -> list[str]:
    """Load sorted class names from class_indices.json or use defaults."""
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
            return sorted(class_indices, key=class_indices.get)
    else:
        print(f"class_indices.json not found at {CLASS_INDICES_PATH}, "
              "using defaults.")
        return DEFAULT_CLASS_NAMES



class MLModel:
    """Singleton ML model for tea leaf disease prediction."""
    _instance = None
    _model = None
    _backbone = None
    _head_layers = None
    _classes = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModel, cls).__new__(cls)
            cls._instance._classes = load_class_names()
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        if self._model is None:
            print(f"Loading classifier from {MODEL_PATH}...")
            try:
                self._model = tf.keras.models.load_model(
                    MODEL_PATH,
                    custom_objects={"FocalLoss": FocalLoss}
                )
                print("Classifier loaded successfully.")
            except Exception as e:
                print(f"Error loading classifier: {e}")

            if os.path.exists(BACKBONE_PATH):
                print(f"Loading backbone extractor from {BACKBONE_PATH}...")
                try:
                    self._backbone = tf.keras.models.load_model(BACKBONE_PATH)
                    self._head_layers = [
                        l for l in self._model.layers
                        if l.__class__.__name__
                        not in ("InputLayer", "Functional")
                    ]
                    print("Backbone extractor loaded successfully.")
                except Exception as e:
                    print(f"Error loading backbone extractor: {e}")
                    self._backbone = None
            else:
                print(f"backbone_extractor.keras not found at {BACKBONE_PATH} "
                      "— Grad-CAM will be unavailable.")

    def predict(self, image_bytes: bytes) -> dict:
        """
        Full inference pipeline with OOD detection, confidence tiers,
        and masked GradCAM.

        Returns dict with keys:
          status, prediction, confidence, message, disclaimer,
          all_probs, gradcam_overlay, crop_display
        """
        if self._model is None:
            raise ValueError("Model not loaded")


        try:
            img_pre, crop_bgr, leaf_mask, leaf_ratio = \
                preprocess_image(image_bytes)
        except Exception as e:
            return {
                "status": "rejected",
                "prediction": None,
                "confidence": None,
                "all_probs": None,
                "message": f"Could not process image: {e}",
                "disclaimer": DISCLAIMER,
                "gradcam_overlay": None,
                "crop_display": None,
            }

        if leaf_ratio < MIN_LEAF_RATIO:
            return {
                "status": "rejected",
                "prediction": None,
                "confidence": None,
                "all_probs": None,
                "message": REJECTION_MESSAGES["no_leaf"],
                "disclaimer": DISCLAIMER,
                "gradcam_overlay": None,
                "crop_display": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
            }

        probs = self._model.predict(img_pre[np.newaxis], verbose=0)[0]
        result = interpret_confidence(probs, self._classes)

        result["crop_display"] = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        if result["status"] != "rejected" and self._backbone is not None:
            class_idx = self._classes.index(result["prediction"])
            overlay, _ = get_gradcam_masked(
                self._backbone, self._head_layers, img_pre,
                crop_bgr, leaf_mask, class_idx)
            result["gradcam_overlay"] = overlay
        else:
            result["gradcam_overlay"] = None

        return result

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from index."""
        if class_idx < len(self._classes):
            return self._classes[class_idx]
        return f"Class {class_idx}"


ml_model = MLModel()
