"""Classification metrics without sklearn dependencies."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Compute $\frac{1}{m}\sum_i \mathbb{1}[\hat{y}_i = y_i]$."""
    true = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
    pred = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
    return float(np.mean(true == pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    r"""Build confusion matrix $C_{ij}=|\{n:y_n=i,\hat{y}_n=j\}|$."""
    true = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
    pred = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true.astype(int), pred.astype(int)):
        cm[t, p] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    r"""Return per-class and macro $P=TP/(TP+FP)$, $R=TP/(TP+FN)$, and $F_1$."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix": cm,
    }
