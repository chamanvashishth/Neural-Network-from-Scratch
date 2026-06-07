"""Optimizers implemented from first principles."""

from __future__ import annotations

import numpy as np


class Adam:
    r"""Adam optimizer.

    Updates each parameter $\theta$ with
    $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$,
    $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$,
    $\hat{m}_t = m_t/(1-\beta_1^t)$,
    $\hat{v}_t = v_t/(1-\beta_2^t)$, and
    $\theta_t = \theta_{t-1} - \alpha \hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, param in params.items():
            grad = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - self.beta1**self.t)
            v_hat = self.v[name] / (1.0 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
