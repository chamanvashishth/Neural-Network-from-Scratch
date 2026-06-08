"""
Neural Network From Scratch (NumPy Only)

Package Exports
---------------
Layers:
    Dense
    ReLU
    BatchNorm
    Dropout
    Softmax

Loss:
    CrossEntropyLoss

Optimizer:
    Adam

Network:
    NeuralNetwork

Data Utilities:
    load_mnist
    one_hot_encode
    train_val_split

Metrics:
    accuracy
    precision_score_macro
    recall_score_macro
    f1_score_macro
    confusion_matrix_np
"""

from .layers import (
    Dense,
    ReLU,
    BatchNorm,
    Dropout,
    Softmax
)

from .losses import CrossEntropyLoss

from .optimizers import Adam

from .network import NeuralNetwork

from .data_loader import (
    load_mnist,
    one_hot_encode,
    train_val_split
)

from .metrics import (
    accuracy,
    precision_score_macro,
    recall_score_macro,
    f1_score_macro,
    confusion_matrix_np
)

__version__ = "1.0.0"

__all__ = [
    "Dense",
    "ReLU",
    "BatchNorm",
    "Dropout",
    "Softmax",
    "CrossEntropyLoss",
    "Adam",
    "NeuralNetwork",
    "load_mnist",
    "one_hot_encode",
    "train_val_split",
    "accuracy",
    "precision_score_macro",
    "recall_score_macro",
    "f1_score_macro",
    "confusion_matrix_np",
]
