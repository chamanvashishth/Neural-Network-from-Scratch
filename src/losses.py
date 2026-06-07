"""Loss functions for NumPy neural networks."""

from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    r"""Multiclass cross-entropy paired with softmax logits.

    For one-hot labels $Y$ and logits $Z$, the stable loss is
    $L = -\frac{1}{m}\sum_i \sum_j Y_{ij}\left(Z_{ij} - \log\sum_k e^{Z_{ik}}\right)$.
    The gradient with respect to logits is
    $\partial L / \partial Z = (\operatorname{softmax}(Z) - Y) / m$.
    """

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, self.eps, 1.0)
        return float(-np.sum(y_true * np.log(self.y_pred)) / y_true.shape[0])

    def forward_from_logits(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp
        return float(-np.sum(y_true * log_probs) / y_true.shape[0])

    def backward(self) -> np.ndarray:
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("CrossEntropyLoss.backward called before forward.")
        return (self.y_pred - self.y_true) / self.y_true.shape[0]
