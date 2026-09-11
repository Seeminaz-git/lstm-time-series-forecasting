from __future__ import annotations

import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # (batch, hidden_size) - final timestep's hidden state
        return self.fc(last_step)  # (batch, output_size)
