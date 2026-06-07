"""MNIST loading utilities using only NumPy and requests."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with destination.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_images, rows * cols)
    return (data.astype(np.float32) / 255.0).astype(np.float32)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        if labels.shape[0] != n_labels:
            raise ValueError(f"Expected {n_labels} labels, found {labels.shape[0]}")
    return labels


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    r"""Encode integer labels as $Y \in \{0,1\}^{m \times C}$ one-hot rows."""
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return encoded


def load_mnist(data_dir: str | Path = "data", val_size: int = 5000, seed: int = 42):
    """Fetch MNIST, normalize pixels to $[0,1]$, and return train/val/test splits.

    Returns ``(X_train, y_train, X_val, y_val, X_test, y_test, labels_test)`` where labels
    are one-hot encoded except ``labels_test``, which contains integer classes for reports.
    """
    data_path = Path(data_dir)
    files = {key: data_path / Path(url).name for key, url in MNIST_URLS.items()}
    for key, url in MNIST_URLS.items():
        _download(url, files[key])

    X_train_full = _read_images(files["train_images"])
    y_train_full_int = _read_labels(files["train_labels"])
    X_test = _read_images(files["test_images"])
    y_test_int = _read_labels(files["test_labels"])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X_train_full.shape[0])
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_val = X_train_full[val_idx]
    y_val = one_hot(y_train_full_int[val_idx])
    X_train = X_train_full[train_idx]
    y_train = one_hot(y_train_full_int[train_idx])
    y_test = one_hot(y_test_int)
    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_int


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Yield mini-batches $X_b, Y_b$ of size at most ``batch_size``."""
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, X.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
