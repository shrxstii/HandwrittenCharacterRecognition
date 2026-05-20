"""
Flask Web App — Handwritten Digit Recognition
CodeAlpha Internship Project
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import re
import os
import sys
from PIL import Image
import io

app = Flask(__name__)

# ── Load model once at startup ──────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'handwriting_cnn_final.h5')

# Try loading model
model = None
try:
    import tensorflow as tf
    from tensorflow import keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")
    else:
        # Try same directory
        alt_path = 'handwriting_cnn_final.h5'
        if os.path.exists(alt_path):
            model = keras.models.load_model(alt_path)
            print(f"[OK] Model loaded from {alt_path}")
        else:
            print(f"[WARNING] Model file not found. Using demo mode.")
except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    print("[INFO] Running in demo mode")


def preprocess_canvas(image_data_url):
    """Convert base64 canvas image → 28x28 numpy array ready for model."""
    # Strip header: "data:image/png;base64,..."
    header, encoded = image_data_url.split(',', 1)
    image_bytes = base64.b64decode(encoded)

    # Open image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

    # White background
    background = Image.new('RGB', img.size, (0, 0, 0))  # black background
    background.paste(img, mask=img.split()[3])            # alpha channel as mask
    img = background.convert('L')                          # grayscale

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    img_arr = np.array(img).astype('float32')

    # Normalize
    img_arr = img_arr / 255.0

    # Add batch + channel dims → (1, 28, 28, 1)
    img_arr = img_arr.reshape(1, 28, 28, 1)
    return img_arr


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image data received'}), 400

        img_arr = preprocess_canvas(image_data)

        if model is not None:
            # Real prediction
            probs = model.predict(img_arr, verbose=0)[0]
            predicted = int(np.argmax(probs))
            confidence = float(probs[predicted]) * 100
            all_probs = [round(float(p) * 100, 2) for p in probs]
        else:
            # Demo mode — random for testing UI
            probs = np.random.dirichlet(np.ones(10))
            predicted = int(np.argmax(probs))
            confidence = float(probs[predicted]) * 100
            all_probs = [round(float(p) * 100, 2) for p in probs]

        return jsonify({
            'predicted': predicted,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs,
            'top3': sorted(
                [{'digit': i, 'prob': round(float(p)*100, 2)} for i, p in enumerate(probs)],
                key=lambda x: x['prob'], reverse=True
            )[:3]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Handwritten Digit Recognition — Web App")
    print("  CodeAlpha Internship")
    print("="*50)
    print("  Open in browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)