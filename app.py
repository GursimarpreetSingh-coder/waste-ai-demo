from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "waste_model_quant.tflite"
LABELS = ("Dry Waste", "Wet Waste")  # Model output order: [dry, wet]
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
inference_lock = threading.Lock()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode an image and apply the MobileNetV2 preprocessing used in training."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Please upload a valid image file.") from exc

    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.float32)
    return np.expand_dims((pixels / 127.5) - 1.0, axis=0)


def predict(image_bytes: bytes) -> tuple[str, float, dict[str, float]]:
    input_data = preprocess_image(image_bytes)
    with inference_lock:
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details["index"])[0].astype(float)

    # Convert model output into a mutually exclusive Dry/Wet probability pair.
    # Some exported binary models return two sigmoid scores, which can total a
    # little more (or less) than 1. The API must always return 100% in total.
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid prediction values.")
    score_total = float(scores.sum())
    if np.any(scores < 0) or score_total <= 0:
        shifted = scores - np.max(scores)
        scores = np.exp(shifted) / np.exp(shifted).sum()
    else:
        scores = scores / score_total

    index = int(np.argmax(scores))
    dry_percent = round(float(scores[0]) * 100, 2)
    probabilities = {
        LABELS[0]: dry_percent,
        LABELS[1]: round(100.0 - dry_percent, 2),
    }
    return LABELS[index], probabilities[LABELS[index]], probabilities


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/background.png")
def brand_logo():
    """Serve the local WasteVision brand image used in the page header."""
    return send_from_directory(BASE_DIR, "background.png")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH.name, "labels": LABELS})


@app.post("/classify")
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image received. Send it using the 'image' field."}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "The image file is empty."}), 400

    try:
        label, confidence, probabilities = predict(image_bytes)
        return jsonify({
            "label": label,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "color": "green" if label == "Dry Waste" else "red",
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Classification failed")
        return jsonify({"error": "Classification failed. Please try another image."}), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image is too large. Maximum size is 8 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
