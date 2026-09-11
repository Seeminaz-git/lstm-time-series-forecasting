import torch
from torch.utils.data import DataLoader

from app.data import generate_synthetic_series
from app.dataset import TimeSeriesDataset, create_windows
from app.model import LSTMForecaster
from app.scaler import MinMaxScaler
from app.train import train_model


def test_model_forward_pass_shape():
    model = LSTMForecaster(input_size=1, hidden_size=16, num_layers=1, output_size=1)
    x = torch.randn(4, 10, 1)  # batch=4, seq_len=10, features=1

    out = model(x)

    assert tuple(out.shape) == (4, 1)


def test_training_loop_runs_and_returns_finite_losses():
    series = generate_synthetic_series(n_days=200, seed=1)
    scaler = MinMaxScaler()
    scaler.fit(series)
    normalized = scaler.transform(series)

    X, y = create_windows(normalized, window_size=14)
    dataset = TimeSeriesDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LSTMForecaster(hidden_size=16)
    torch.manual_seed(0)

    losses = train_model(model, loader, epochs=3, lr=1e-2)

    assert len(losses) == 3
    assert all(isinstance(loss, float) and loss >= 0 for loss in losses)
