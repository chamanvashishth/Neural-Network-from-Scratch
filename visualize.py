"""Generate training and evaluation visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_loader import load_mnist
from src.metrics import precision_recall_f1
from src.network import NeuralNetwork

RESULTS_DIR = Path("results")


def _load_history() -> dict:
    with (RESULTS_DIR / "history.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curve(history: dict) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.axvline(best_epoch, linestyle=":", color="black", label=f"Best Val Epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=160)
    plt.close()


def plot_accuracy_curve(history: dict) -> None:
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, np.array(history["train_acc"]) * 100, label="Train Acc")
    plt.plot(epochs, np.array(history["val_acc"]) * 100, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=160)
    plt.close()


def plot_confusion(cm: np.ndarray) -> None:
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(8, 7))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Confusion Matrix (n=10000)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def plot_weight_distributions(model: NeuralNetwork) -> None:
    dense_layers = model.dense_layers()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, layer, name in zip(axes, dense_layers, ["W1", "W2", "W3"]):
        ax.hist(layer.initial_W.ravel(), bins=80, alpha=0.45, density=True, label="He init")
        ax.hist(layer.W.ravel(), bins=80, alpha=0.45, density=True, label="Final")
        ax.set_title(f"{name} distribution")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "weight_distributions.png", dpi=160)
    plt.close(fig)


def plot_sample_predictions(X_test: np.ndarray, y_test: np.ndarray, probs: np.ndarray) -> None:
    true = np.argmax(y_test, axis=1)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1) * 100
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for ax, idx in zip(axes.ravel(), range(25)):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"True:{true[idx]} | Pred:{pred[idx]} | Conf:{conf[idx]:.1f}%", fontsize=8)
        color = "green" if true[idx] == pred[idx] else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions.png", dpi=160)
    plt.close(fig)


def plot_per_class_f1(f1: np.ndarray) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(10), f1 * 100)
    plt.xticks(np.arange(10))
    plt.ylim(0, 100)
    plt.xlabel("Digit")
    plt.ylabel("F1 Score (%)")
    plt.title("Per-Class F1 Score")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_f1.png", dpi=160)
    plt.close()


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    history = _load_history()
    _, _, _, _, X_test, y_test, _ = load_mnist()
    model = NeuralNetwork()
    model.load("checkpoints")
    probs = model.predict(X_test)
    labels_true = np.argmax(y_test, axis=1)
    labels_pred = np.argmax(probs, axis=1)
    scores = precision_recall_f1(labels_true, labels_pred)

    plot_loss_curve(history)
    plot_accuracy_curve(history)
    plot_confusion(scores["confusion_matrix"])
    plot_weight_distributions(model)
    plot_sample_predictions(X_test, y_test, probs)
    plot_per_class_f1(scores["f1_per_class"])
    print(f"Saved plots to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
