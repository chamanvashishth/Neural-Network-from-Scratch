"""Train the NumPy-only MNIST neural network."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

np.random.seed(42)

from src.data_loader import iterate_minibatches, load_mnist
from src.metrics import accuracy
from src.network import NeuralNetwork
from src.optimizers import Adam

EPOCHS = 50
BATCH_SIZE = 128
LR = 0.001
LR_DECAY = 0.95
LR_DECAY_EVERY = 10
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def evaluate_split(model: NeuralNetwork, X: np.ndarray, y: np.ndarray, batch_size: int = 1024):
    losses = []
    preds = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        y_hat = model.forward(xb, training=False)
        losses.append(model.compute_loss(yb, y_hat) * xb.shape[0])
        preds.append(y_hat)
    y_pred = np.vstack(preds)
    return float(np.sum(losses) / X.shape[0]), accuracy(y, y_pred)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    X_train, y_train, X_val, y_val, _, _, _ = load_mnist()
    model = NeuralNetwork()
    optimizer = Adam(lr=LR, beta1=0.9, beta2=0.999, eps=1e-8)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "epoch_time": []}
    best_val_acc = -np.inf

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        running_loss = 0.0
        train_preds = []
        train_targets = []

        for xb, yb in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            y_hat = model.forward(xb, training=True)
            loss = model.compute_loss(yb, y_hat)
            model.backward()
            model.update(optimizer)
            running_loss += loss * xb.shape[0]
            train_preds.append(y_hat)
            train_targets.append(yb)

        train_loss = running_loss / X_train.shape[0]
        train_acc = accuracy(np.vstack(train_targets), np.vstack(train_preds))
        val_loss, val_acc = evaluate_split(model, X_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(CHECKPOINT_DIR)

        elapsed = time.time() - start_time
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.lr)
        history["epoch_time"].append(elapsed)

        print(
            f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {optimizer.lr:.6f} | Time: {elapsed:.1f}s"
        )

        if epoch % LR_DECAY_EVERY == 0:
            optimizer.lr *= LR_DECAY

    with (RESULTS_DIR / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
