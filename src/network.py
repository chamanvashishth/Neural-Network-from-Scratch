"""Composable fully-connected neural network."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .layers import BatchNorm, Dense, Dropout, ReLU, Softmax
from .losses import CrossEntropyLoss
from .optimizers import Adam


class NeuralNetwork:
    r"""Exact MNIST MLP: $784 \to 256 \to 128 \to 10$.

    The model evaluates
    $X \to XW^{[1]}+b^{[1]} \to \operatorname{ReLU} \to \operatorname{BN} \to
    \operatorname{Dropout} \to XW^{[2]}+b^{[2]} \to \operatorname{ReLU} \to
    \operatorname{BN} \to \operatorname{Dropout} \to XW^{[3]}+b^{[3]} \to
    \operatorname{Softmax}$, and its ``backward`` method applies the reverse chain rule manually.
    """

    def __init__(self):
        self.layers = [
            Dense(784, 256, "dense1", he_init=True),
            ReLU(),
            BatchNorm(256, "bn1"),
            Dropout(0.3),
            Dense(256, 128, "dense2", he_init=True),
            ReLU(),
            BatchNorm(128, "bn2"),
            Dropout(0.2),
            Dense(128, 10, "dense3", he_init=True),
            Softmax(),
        ]
        self.loss = CrossEntropyLoss()

    def add(self, layer) -> None:
        self.layers.append(layer)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.loss.forward(y_true, y_pred)

    def backward(self) -> None:
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.update(layer.parameters())
        return params

    def gradients(self) -> dict[str, np.ndarray]:
        grads: dict[str, np.ndarray] = {}
        for layer in self.layers:
            if hasattr(layer, "gradients"):
                grads.update(layer.gradients())
        return grads

    def update(self, optimizer: Adam) -> None:
        optimizer.step(self.parameters(), self.gradients())

    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        outputs = []
        for start in range(0, X.shape[0], batch_size):
            outputs.append(self.forward(X[start:start + batch_size], training=False))
        return np.vstack(outputs)

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                arrays[f"layer_{i}_W"] = layer.W
                arrays[f"layer_{i}_b"] = layer.b
                arrays[f"layer_{i}_initial_W"] = layer.initial_W
            elif isinstance(layer, BatchNorm):
                arrays[f"layer_{i}_gamma"] = layer.gamma
                arrays[f"layer_{i}_beta"] = layer.beta
                arrays[f"layer_{i}_running_mean"] = layer.running_mean
                arrays[f"layer_{i}_running_var"] = layer.running_var
        np.save(path / "best_model.npy", arrays, allow_pickle=True)
        for name, array in arrays.items():
            np.save(path / f"{name}.npy", array)

    def load(self, directory: str | Path) -> None:
        arrays = np.load(Path(directory) / "best_model.npy", allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.W[...] = arrays[f"layer_{i}_W"]
                layer.b[...] = arrays[f"layer_{i}_b"]
                layer.initial_W[...] = arrays.get(f"layer_{i}_initial_W", layer.initial_W)
            elif isinstance(layer, BatchNorm):
                layer.gamma[...] = arrays[f"layer_{i}_gamma"]
                layer.beta[...] = arrays[f"layer_{i}_beta"]
                layer.running_mean[...] = arrays[f"layer_{i}_running_mean"]
                layer.running_var[...] = arrays[f"layer_{i}_running_var"]

    def dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]
