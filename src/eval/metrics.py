"""Evaluation metric functions."""

from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((y_true >= low) & (y_true <= high)))


def mean_interval_width(low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean(high - low))


def brier_binary(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


def ece_binary(y_true: np.ndarray, p_pred: np.ndarray, bins: int = 10) -> float:
    # Expected calibration error for binary probabilities.
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    score = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (p_pred >= left) & (p_pred < right if idx < bins - 1 else p_pred <= right)
        if not np.any(mask):
            continue
        conf = float(np.mean(p_pred[mask]))
        acc = float(np.mean(y_true[mask]))
        score += abs(acc - conf) * (float(np.sum(mask)) / total)
    return score

