from __future__ import annotations

import torch
from torch.utils.data import Dataset


def create_windows(series: list[float], window_size: int) -> tuple[list[list[float]], list[float]]:
    """Slides a fixed-size window over the series: X[i] is `window_size`
    consecutive values, y[i] is the value immediately following them."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return X, y


class TimeSeriesDataset(Dataset):
    def __init__(self, X: list[list[float]], y: list[float]):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, window_size, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
