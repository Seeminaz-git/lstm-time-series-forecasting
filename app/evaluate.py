"""Evaluation against a naive baseline.

A forecasting model is only useful if it beats the trivial "predict the last
observed value" baseline — comparing against it is standard practice and
catches a model that looks fine on a loss curve but isn't actually adding
value.
"""
from __future__ import annotations

import torch

from app.dataset import create_windows
from app.model import LSTMForecaster
from app.scaler import MinMaxScaler


def naive_baseline_predictions(series: list[float], window_size: int) -> list[float]:
    """Predicts each next value as simply the last value in its window."""
    return series[window_size - 1 : -1]


def mae(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def evaluate_model(model: LSTMForecaster, scaler: MinMaxScaler, series: list[float], window_size: int) -> dict:
    normalized = scaler.transform(series)
    X, _ = create_windows(normalized, window_size)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        predictions_normalized = model(x_tensor).squeeze(-1).tolist()

    predictions = scaler.inverse_transform(predictions_normalized)
    actual = series[window_size:]
    naive_predictions = naive_baseline_predictions(series, window_size)

    return {
        "model_mae": mae(actual, predictions),
        "naive_baseline_mae": mae(actual, naive_predictions),
    }
