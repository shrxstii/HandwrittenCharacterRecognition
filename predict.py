"""
predict.py — Apni khud ki image pe test karo!
Koi bhi 0-9 digit haath se likho, photo lo, aur yeh script
batayegi model ne kya predict kiya.

Usage:
    python predict.py                  # test set se random images
    python predict.py --image mine.png # apni image pe test
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
import argparse
import os
import sys

# ─────────────────────────────────────────────
# Load trained model
# ─────────────────────────────────────────────
MODEL_PATH = "handwriting_cnn_final.h5"

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found: {MODEL_PATH}")
    print("  Pehle handwriting_model.py run karo!")
    sys.exit(1)

print("Loading saved model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# ─────────────────────────────────────────────
# Option A — Test on MNIST test set (random)
# ─────────────────────────────────────────────
def test_on_mnist(n_samples=20):
    print(f"Testing on {n_samples} random MNIST test images...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test_norm = x_test.astype("float32") / 255.0
    x_test_norm = np.expand_dims(x_test_norm, -1)

    # Random indices choose karo
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    images  = x_test_norm[indices]
    labels  = y_test[indices]

    # Predict
    preds_probs = model.predict(images, verbose=0)
    preds       = np.argmax(preds_probs, axis=1)
    confidences = np.max(preds_probs, axis=1) * 100

    # Plot
    cols = 5
    rows = n_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle("Model Predictions on Random Test Images",
                 fontsize=14, fontweight='bold')

    for i in range(n_samples):
        ax = axes[i // cols, i % cols]
        ax.imshow(images[i].squeeze(), cmap='gray')

        correct = preds[i] == labels[i]
        color   = '#2e7d32' if correct else '#c62828'
        symbol  = '✓' if correct else '✗'

        ax.set_title(
            f"{symbol} True: {labels[i]}  Pred: {preds[i]}\n"
            f"Confidence: {confidences[i]:.1f}%",
            fontsize=9,
            color=color,
            fontweight='bold'
        )
        ax.axis('off')

        # Border color by correctness
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.tight_layout()
    output_path = "predictions_random.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_path}")

    correct_count = np.sum(preds == labels)
    print(f"\nResults: {correct_count}/{n_samples} correct ({correct_count/n_samples*100:.0f}%)")

# ─────────────────────────────────────────────
# Option B — Test on your own image
# ─────────────────────────────────────────────
def test_on_custom_image(image_path):
    from PIL import Image

    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    # Load aur preprocess
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((28, 28))                  # resize to 28x28
    img_arr = np.array(img).astype("float32")

    # MNIST mein background kala aur digit safed hota hai
    # Agar teri image opposite hai (safed background, kala digit)
    # toh invert karo
    if img_arr.mean() > 127:
        img_arr = 255 - img_arr   # invert

    img_arr = img_arr / 255.0
    img_input = img_arr.reshape(1, 28, 28, 1)

    # Predict
    pred_probs  = model.predict(img_input, verbose=0)[0]
    pred_class  = np.argmax(pred_probs)
    confidence  = pred_probs[pred_class] * 100

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Custom Image Prediction", fontsize=14, fontweight='bold')

    # Image
    ax1.imshow(img_arr, cmap='gray')
    ax1.set_title(f"Your Image (28×28)\nPredicted: {pred_class}  ({confidence:.1f}% confident)",
                  fontsize=12, fontweight='bold', color='#1565c0')
    ax1.axis('off')

    # Bar chart of all probabilities
    colors = ['#ef5350' if i == pred_class else '#90caf9' for i in range(10)]
    bars = ax2.bar(range(10), pred_probs * 100, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Digit Class', fontsize=11)
    ax2.set_ylabel('Confidence (%)', fontsize=11)
    ax2.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')

    # Values on top of bars
    for bar, prob in zip(bars, pred_probs):
        if prob > 0.01:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_path = "custom_prediction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPredicted digit : {pred_class}")
    print(f"Confidence      : {confidence:.2f}%")
    print(f"Saved           : {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwriting digit predictor")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to your own image (optional)")
    args = parser.parse_args()

    if args.image:
        test_on_custom_image(args.image)
    else:
        test_on_mnist(n_samples=20)