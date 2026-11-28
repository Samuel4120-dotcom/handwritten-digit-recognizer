"""
evaluate.py  (Pair 5)
Tests the Baseline (classical ML) and CNN models on the 10,000 MNIST test
images, prints accuracy, and saves confusion matrices into results/.

Run from the project root:
    python evaluate.py
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts to files without needing a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras

BASELINE_PATH = os.path.join("models", "baseline_model.pkl")
CNN_PATH = os.path.join("models", "cnn_model.keras")
RESULTS_DIR = "results"


def load_test_data():
    """Load the 10,000 MNIST test images, scaled to 0-1."""
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    return x_test, y_test


def evaluate_baseline(x_test, y_test):
    """Baseline model expects flat rows of 784 numbers."""
    if not os.path.exists(BASELINE_PATH):
        print(f"[skip] {BASELINE_PATH} not found yet.")
        return None
    model = joblib.load(BASELINE_PATH)
    x_flat = x_test.reshape(len(x_test), 784)
    return model.predict(x_flat)


def evaluate_cnn(x_test, y_test):
    """CNN expects 28x28x1 images."""
    if not os.path.exists(CNN_PATH):
        print(f"[skip] {CNN_PATH} not found yet.")
        return None
    model = keras.models.load_model(CNN_PATH)
    x_img = x_test.reshape(len(x_test), 28, 28, 1)
    probs = model.predict(x_img, verbose=0)
    return np.argmax(probs, axis=1)


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred) * 100
    print(f"\n===== {name} =====")
    print(f"Test accuracy: {acc:.2f}%")
    print(classification_report(y_true, y_pred, digits=3))
    return acc


def save_single_matrix(name, y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"{name} - Confusion Matrix (10,000 test images)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close(fig)


def save_combined_matrix(results, y_true):
    """results: list of (name, y_pred). Saves confusion_matrix.png side by side."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 7))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_true, y_pred, labels=range(10))
        ConfusionMatrixDisplay(cm, display_labels=range(10)).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        acc = accuracy_score(y_true, y_pred) * 100
        ax.set_title(f"{name} ({acc:.2f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    x_test, y_test = load_test_data()

    results = []

    base_pred = evaluate_baseline(x_test, y_test)
    if base_pred is not None:
        report("Baseline Model (Classical ML)", y_test, base_pred)
        save_single_matrix(
            "Baseline Model", y_test, base_pred, "confusion_matrix_baseline.png"
        )
        results.append(("Baseline Model", base_pred))

    cnn_pred = evaluate_cnn(x_test, y_test)
    if cnn_pred is not None:
        report("CNN Model (Deep Learning)", y_test, cnn_pred)
        save_single_matrix("CNN Model", y_test, cnn_pred, "confusion_matrix_cnn.png")
        results.append(("CNN Model", cnn_pred))

    if not results:
        print("\nNo models found. Pull from GitHub once Pairs 2 and 3 have pushed.")
        return

    save_combined_matrix(results, y_test)
    print(f"\nSaved charts in {RESULTS_DIR}/ (confusion_matrix.png is the main one).")


if __name__ == "__main__":
    main()
