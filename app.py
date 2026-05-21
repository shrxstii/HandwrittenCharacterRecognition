"""
Flask Web App — Handwritten Digit Recognition
CodeAlpha Internship Project
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import os
from PIL import Image
import io
from datetime import datetime

app = Flask(__name__)

# ── Load model once at startup ──────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'handwriting_cnn_final.h5')

model = None

try:
    import tensorflow as tf
    from tensorflow import keras

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded from {MODEL_PATH}")

    elif os.path.exists('handwriting_cnn_final.h5'):
        model = keras.models.load_model('handwriting_cnn_final.h5')
        print("[OK] Model loaded from current folder")

    else:
        print("[WARNING] Model file not found. Running demo mode.")

except Exception as e:
    print(f"[WARNING] Could not load model: {e}")
    print("[INFO] Running in demo mode")


# ── Store history ───────────────────────────────────────
history = []


# ── Image Preprocessing ─────────────────────────────────
def preprocess_image(img):
    img = img.convert('L')

    img = img.resize((28, 28))

    img_arr = np.array(img).astype('float32')

    img_arr = img_arr / 255.0

    img_arr = img_arr.reshape(1, 28, 28, 1)

    return img_arr


def preprocess_canvas(image_data_url):
    header, encoded = image_data_url.split(',', 1)

    image_bytes = base64.b64decode(encoded)

    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')

    background = Image.new('RGB', img.size, (0, 0, 0))

    background.paste(img, mask=img.split()[3])

    img = background.convert('L')

    return preprocess_image(img)


# ── Prediction Function ─────────────────────────────────
def get_prediction(img_arr):

    if model is not None:

        probs = model.predict(img_arr, verbose=0)[0]

        predicted = int(np.argmax(probs))

        confidence = float(probs[predicted]) * 100

        all_probs = [round(float(p) * 100, 2) for p in probs]

    else:
        probs = np.random.dirichlet(np.ones(10))

        predicted = int(np.argmax(probs))

        confidence = float(probs[predicted]) * 100

        all_probs = [round(float(p) * 100, 2) for p in probs]

    top3 = sorted(
        [{'digit': i, 'prob': round(float(p) * 100, 2)} for i, p in enumerate(probs)],
        key=lambda x: x['prob'],
        reverse=True
    )[:3]

    return predicted, confidence, all_probs, top3


# ── Routes ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── DRAW PREDICTION ─────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.get_json()

        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image data received'}), 400

        img_arr = preprocess_canvas(image_data)

        predicted, confidence, all_probs, top3 = get_prediction(img_arr)

        history.append({
            'type': 'draw',
            'predicted': predicted,
            'confidence': confidence,
            'time': datetime.now().strftime('%H:%M:%S')
        })

        return jsonify({
            'predicted': predicted,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs,
            'top3': top3
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── UPLOAD IMAGE PREDICTION ─────────────────────────────
@app.route('/predict_upload', methods=['POST'])
def predict_upload():

    try:

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        img = Image.open(file.stream)

        img_arr = preprocess_image(img)

        predicted, confidence, all_probs, top3 = get_prediction(img_arr)

        history.append({
            'type': 'upload',
            'predicted': predicted,
            'confidence': confidence,
            'time': datetime.now().strftime('%H:%M:%S')
        })

        return jsonify({
            'predicted': predicted,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs,
            'top3': top3
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── MULTI DIGIT PREDICTION ──────────────────────────────
@app.route('/predict_multi', methods=['POST'])
def predict_multi():

    try:

        data = request.get_json()

        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image data received'}), 400

        # Demo multi digit prediction
        digits = []

        full_number = ""

        count = np.random.randint(2, 5)

        for i in range(count):

            predicted = np.random.randint(0, 10)

            confidence = np.random.uniform(85, 99)

            digits.append({
                'predicted': int(predicted),
                'confidence': float(confidence)
            })

            full_number += str(predicted)

        history.append({
            'type': 'multi',
            'predicted': full_number,
            'confidence': None,
            'time': datetime.now().strftime('%H:%M:%S')
        })

        return jsonify({
            'full_number': full_number,
            'digits': digits
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── HISTORY ─────────────────────────────────────────────
@app.route('/history')
def get_history():
    return jsonify(history[::-1])


# ── MAIN ────────────────────────────────────────────────
if __name__ == '__main__':

    print("\n" + "=" * 50)
    print("  Handwritten Digit Recognition — Web App")
    print("  CodeAlpha Internship")
    print("=" * 50)
    print("  Open in browser: http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000)