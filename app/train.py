from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model import LSTMForecaster


def train_model(model: LSTMForecaster, data_loader: DataLoader, epochs: int = 10, lr: float = 1e-3) -> list[float]:
    """Standard supervised training loop. Returns the mean training loss per
    epoch so callers/tests can confirm the loop actually ran."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses = []
    for _ in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

    return epoch_losses
