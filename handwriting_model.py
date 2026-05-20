"""
Handwritten Character Recognition — CodeAlpha Internship
Author: [Tera Naam Yahan Likhna]
Dataset: MNIST (0-9 digits) + EMNIST (A-Z letters)
Model: Convolutional Neural Network (CNN)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY (same results har baar)
# ─────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 55)
print("  Handwritten Character Recognition — CodeAlpha")
print("=" * 55)
print(f"  TensorFlow version: {tf.__version__}")
print("=" * 55)

# ─────────────────────────────────────────────
# 2. DATASET LOAD — MNIST (digits 0-9)
# ─────────────────────────────────────────────
print("\n[1/6] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"  Training images : {x_train.shape[0]:,}  ({x_train.shape[1]}x{x_train.shape[2]} pixels each)")
print(f"  Testing  images : {x_test.shape[0]:,}")
print(f"  Classes         : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")

# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2/6] Preprocessing data...")

# Normalize: pixel values 0-255  →  0.0 to 1.0
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Reshape: add channel dimension (required by CNN)
# Shape: (samples, 28, 28) → (samples, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test,  -1)

print(f"  x_train shape after reshape : {x_train.shape}")
print(f"  x_test  shape after reshape : {x_test.shape}")

# ─────────────────────────────────────────────
# 4. VISUALIZE SAMPLE IMAGES
# ─────────────────────────────────────────────
print("\n[3/6] Saving sample images visualization...")

fig, axes = plt.subplots(3, 10, figsize=(15, 5))
fig.suptitle("MNIST Sample Images — CodeAlpha Handwriting Recognition",
             fontsize=13, fontweight='bold', y=1.02)

for digit in range(10):
    indices = np.where(y_train == digit)[0]
    for row in range(3):
        ax = axes[row, digit]
        ax.imshow(x_train[indices[row]].squeeze(), cmap='gray')
        ax.axis('off')
        if row == 0:
            ax.set_title(str(digit), fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sample_images.png")

# ─────────────────────────────────────────────
# 5. BUILD CNN MODEL
# ─────────────────────────────────────────────
print("\n[4/6] Building CNN model...")

num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        # Block 1 — edges aur basic shapes seekhta hai
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2 — complex patterns seekhta hai
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 3 — high-level features
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ],
    name="HandwritingCNN"
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.summary()

total_params = model.count_params()
print(f"\n  Total trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────
print("\n[5/6] Training model (yeh 5-10 min le sakta hai)...")

# Callbacks
callbacks = [
    # Learning rate automatically kam karta hai jab improvement ruk jaaye
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=3,
        min_lr=1e-6, verbose=1
    ),
    # Best model save karta hai automatically
    keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    # Agar 10 epochs tak improvement na ho toh early stop
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    )
]

batch_size = 128
epochs = 25

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,   # 10% training data validation ke liye
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
print("\n[6/6] Evaluating model on test data...")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification Report
print("\n  Classification Report:")
print("  " + "─" * 50)
report = classification_report(y_test, y_pred,
                                target_names=[str(i) for i in range(10)])
for line in report.split('\n'):
    print("  " + line)

# ─────────────────────────────────────────────
# 8. PLOTS — Training History
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Training History — CodeAlpha", fontsize=14, fontweight='bold')

# Accuracy plot
axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='#2196F3', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#FF5722', linewidth=2)
axes[0].set_title('Accuracy over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.9, 1.01])

# Loss plot
axes[1].plot(history.history['loss'],     label='Train Loss', color='#2196F3', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#FF5722', linewidth=2)
axes[1].set_title('Loss over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: training_history.png")

# ─────────────────────────────────────────────
# 9. CONFUSION MATRIX
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title(f'Confusion Matrix — Test Accuracy: {test_acc*100:.2f}%',
          fontsize=14, fontweight='bold', pad=15)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrix.png")

# ─────────────────────────────────────────────
# 10. WRONG PREDICTIONS — Interesting dikhane ke liye
# ─────────────────────────────────────────────
wrong_idx = np.where(y_pred != y_test)[0]
print(f"\n  Total wrong predictions: {len(wrong_idx)} out of {len(y_test):,}")

fig, axes = plt.subplots(3, 8, figsize=(16, 7))
fig.suptitle("Where the Model Made Mistakes", fontsize=13, fontweight='bold')

for i, idx in enumerate(wrong_idx[:24]):
    ax = axes[i // 8, i % 8]
    ax.imshow(x_test[idx].squeeze(), cmap='Reds')
    ax.set_title(f"True:{y_test[idx]}\nPred:{y_pred[idx]}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig("wrong_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: wrong_predictions.png")

# ─────────────────────────────────────────────
# 11. SAVE FINAL MODEL
# ─────────────────────────────────────────────
model.save("handwriting_cnn_final.h5")
print("\n  Saved: handwriting_cnn_final.h5")

print("\n" + "=" * 55)
print(f"  FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
print("=" * 55)
print("\n  Files generated:")
print("    sample_images.png       — dataset preview")
print("    training_history.png    — accuracy & loss curves")
print("    confusion_matrix.png    — per-digit performance")
print("    wrong_predictions.png   — model mistakes")
print("    handwriting_cnn_final.h5 — saved model")
print("\n  Run predict.py to test on your own images!")
print("=" * 55)