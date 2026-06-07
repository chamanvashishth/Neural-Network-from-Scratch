"""Evaluate a saved NumPy neural-network checkpoint on MNIST test data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data_loader import load_mnist
from src.metrics import accuracy, precision_recall_f1
from src.network import NeuralNetwork


def main() -> None:
    _, _, _, _, X_test, y_test, _ = load_mnist()
    model = NeuralNetwork()
    model.load("checkpoints")
    y_pred = model.predict(X_test)
    labels_pred = np.argmax(y_pred, axis=1)
    labels_true = np.argmax(y_test, axis=1)
    scores = precision_recall_f1(labels_true, labels_pred)
    acc = accuracy(labels_true, labels_pred)
    report = {
        "accuracy": acc,
        "precision_macro": scores["precision_macro"],
        "recall_macro": scores["recall_macro"],
        "f1_macro": scores["f1_macro"],
        "per_class_f1": scores["f1_per_class"].tolist(),
        "confusion_matrix": scores["confusion_matrix"].tolist(),
    }
    Path("results").mkdir(exist_ok=True)
    with Path("results/evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Macro Precision: {scores['precision_macro'] * 100:.2f}%")
    print(f"Macro Recall: {scores['recall_macro'] * 100:.2f}%")
    print(f"Macro F1: {scores['f1_macro'] * 100:.2f}%")


if __name__ == "__main__":
    main()
