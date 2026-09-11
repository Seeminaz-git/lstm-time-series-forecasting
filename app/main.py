"""FastAPI app exposing LSTM-based appointment-volume forecasts.

Run locally with:  uvicorn app.main:app --reload

For demo purposes the model trains in-process at startup on a synthetic
series (a few seconds on CPU). Set MODEL_PATH to load persisted weights
instead of retraining every time the service starts (and to save freshly
trained weights there when no checkpoint exists yet) — see "Going to
production" in the README.
"""
from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader

from app.data import generate_synthetic_series
from app.dataset import TimeSeriesDataset, create_windows
from app.forecast import forecast_future
from app.model import LSTMForecaster
from app.scaler import MinMaxScaler
from app.train import train_model

WINDOW_SIZE = 14

series = generate_synthetic_series(n_days=300, seed=42)
scaler = MinMaxScaler()
scaler.fit(series)
normalized = scaler.transform(series)

X, y = create_windows(normalized, WINDOW_SIZE)
dataset = TimeSeriesDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = LSTMForecaster(hidden_size=16)
_model_path = os.environ.get("MODEL_PATH")

if _model_path and os.path.exists(_model_path):
    model.load_state_dict(torch.load(_model_path, map_location="cpu"))
else:
    torch.manual_seed(42)
    train_model(model, loader, epochs=5, lr=1e-2)
    if _model_path:
        torch.save(model.state_dict(), _model_path)

last_window = series[-WINDOW_SIZE:]

app = FastAPI(title="LSTM Time Series Forecasting")


class ForecastRequest(BaseModel):
    steps: int = 7


@app.post("/forecast")
def forecast(payload: ForecastRequest):
    predictions = forecast_future(model, scaler, last_window, payload.steps)
    return {"steps": payload.steps, "forecast": predictions}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    preds = forecast_future(model, scaler, last_window, steps=7)
    print("Next 7 days forecast:", [round(p, 2) for p in preds])
