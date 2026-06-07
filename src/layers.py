"""Core neural-network layers implemented with NumPy only."""

from __future__ import annotations

import numpy as np


class Dense:
    r"""Fully connected affine layer.

    Computes the forward equation
    $Z = XW + b$ and the backward gradients
    $\partial L / \partial W = X^T \delta$,
    $\partial L / \partial b = \sum_i \delta_i$, and
    $\partial L / \partial X = \delta W^T$.
    """

    def __init__(self, input_dim: int, output_dim: int, name: str, he_init: bool = True):
        self.name = name
        scale = np.sqrt(2.0 / input_dim) if he_init else np.sqrt(1.0 / input_dim)
        self.W = (np.random.randn(input_dim, output_dim) * scale).astype(np.float32)
        self.b = np.zeros((1, output_dim), dtype=np.float32)
        self.initial_W = self.W.copy()
        self.X: np.ndarray | None = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        if self.X is None:
            raise RuntimeError("Dense.backward called before forward.")
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.W.T

    def parameters(self) -> dict[str, np.ndarray]:
        return {f"{self.name}.W": self.W, f"{self.name}.b": self.b}

    def gradients(self) -> dict[str, np.ndarray]:
        return {f"{self.name}.W": self.dW, f"{self.name}.b": self.db}


class ReLU:
    r"""Rectified linear activation.

    Applies $A = \operatorname{ReLU}(Z) = \max(0, Z)$ and propagates
    $\partial L / \partial Z = (Z > 0) \odot \partial L / \partial A$.
    """

    def __init__(self):
        self.Z: np.ndarray | None = None

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        if self.Z is None:
            raise RuntimeError("ReLU.backward called before forward.")
        return dA * (self.Z > 0)


class BatchNorm:
    r"""Batch normalization layer for dense activations.

    During training, computes $\hat{x} = (x - \mu_B) / \sqrt{\sigma_B^2 + \epsilon}$
    and $y = \gamma \hat{x} + \beta$. Running mean and variance are updated for
    inference. The backward pass manually evaluates gradients for $\gamma$, $\beta$,
    variance, mean, and the input.
    """

    def __init__(self, dim: int, name: str, momentum: float = 0.9, eps: float = 1e-5):
        self.name = name
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones((1, dim), dtype=np.float32)
        self.beta = np.zeros((1, dim), dtype=np.float32)
        self.running_mean = np.zeros((1, dim), dtype=np.float32)
        self.running_var = np.ones((1, dim), dtype=np.float32)
        self.X: np.ndarray | None = None
        self.X_hat: np.ndarray | None = None
        self.batch_mean: np.ndarray | None = None
        self.batch_var: np.ndarray | None = None
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.X = X
            self.batch_mean = np.mean(X, axis=0, keepdims=True)
            self.batch_var = np.var(X, axis=0, keepdims=True)
            self.X_hat = (X - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.batch_var
            return self.gamma * self.X_hat + self.beta
        X_hat = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * X_hat + self.beta

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if self.X is None or self.X_hat is None or self.batch_mean is None or self.batch_var is None:
            raise RuntimeError("BatchNorm.backward called before training forward.")
        m = dY.shape[0]
        self.dgamma = np.sum(dY * self.X_hat, axis=0, keepdims=True)
        self.dbeta = np.sum(dY, axis=0, keepdims=True)
        dX_hat = dY * self.gamma
        inv_std = 1.0 / np.sqrt(self.batch_var + self.eps)
        x_mu = self.X - self.batch_mean
        dvar = np.sum(dX_hat * x_mu * -0.5 * inv_std**3, axis=0, keepdims=True)
        dmean = np.sum(-dX_hat * inv_std, axis=0, keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=0, keepdims=True)
        return dX_hat * inv_std + dvar * 2.0 * x_mu / m + dmean / m

    def parameters(self) -> dict[str, np.ndarray]:
        return {f"{self.name}.gamma": self.gamma, f"{self.name}.beta": self.beta}

    def gradients(self) -> dict[str, np.ndarray]:
        return {f"{self.name}.gamma": self.dgamma, f"{self.name}.beta": self.dbeta}


class Dropout:
    r"""Inverted dropout regularization.

    In training mode, samples a Bernoulli mask $M \sim \operatorname{Bernoulli}(1-p)$
    and returns $A' = A \odot M / (1-p)$. Backward propagation uses the same scaled
    mask: $\partial L / \partial A = \partial L / \partial A' \odot M / (1-p)$.
    """

    def __init__(self, p: float):
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout probability p must be in [0, 1).")
        self.p = p
        self.mask: np.ndarray | None = None

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.p > 0.0:
            self.mask = (np.random.rand(*X.shape) >= self.p).astype(X.dtype) / (1.0 - self.p)
            return X * self.mask
        self.mask = None
        return X

    def backward(self, dX: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return dX
        return dX * self.mask


class Softmax:
    r"""Softmax output layer.

    Computes class probabilities $\hat{Y}_{ij} = \exp(z_{ij} - c_i) /
    \sum_k \exp(z_{ik} - c_i)$ where $c_i = \max_k z_{ik}$ for numerical stability.
    """

    def __init__(self):
        self.probs: np.ndarray | None = None

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        return dZ
