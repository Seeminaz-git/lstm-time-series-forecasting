from __future__ import annotations

import torch

from app.model import LSTMForecaster
from app.scaler import MinMaxScaler


def forecast_future(model: LSTMForecaster, scaler: MinMaxScaler, last_window: list[float], steps: int) -> list[float]:
    """Iterative (autoregressive) multi-step forecasting: predict one step
    ahead, roll it into the input window, and repeat — the standard approach
    for turning a single-step model into a multi-step forecaster."""
    model.eval()
    window_size = len(last_window)
    window = scaler.transform(last_window)
    predictions_normalized: list[float] = []

    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window[-window_size:], dtype=torch.float32).view(1, window_size, 1)
            pred = model(x).item()
            predictions_normalized.append(pred)
            window.append(pred)

    return scaler.inverse_transform(predictions_normalized)
