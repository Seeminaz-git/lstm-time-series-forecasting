# LSTM Time Series Forecasting

An LSTM-based forecaster for daily clinic appointment volume — the kind of
demand-forecasting model that would feed into staffing/scheduling decisions
for a healthcare operations team. Built with PyTorch: sliding-window
sequence data, an `nn.LSTM` model, a standard training loop, iterative
multi-step forecasting, and evaluation against a naive baseline.


## Architecture

```
generate_synthetic_series() ──▶ MinMaxScaler.fit/transform ──▶ create_windows(window_size=14)
                                                                        │
                                                                        ▼
                                                          TimeSeriesDataset + DataLoader
                                                                        │
                                                                        ▼
                                              LSTMForecaster (nn.LSTM → last timestep → nn.Linear)
                                                                        │
                                                    train_model() (Adam + MSE loss)
                                                                        │
                                        ┌───────────────────────────────┼───────────────────────────┐
                                        ▼                                                             ▼
                          forecast_future() — iterative,                              evaluate_model() — MAE vs. a
                          autoregressive multi-step forecast                          naive "predict the last value"
                                                                                       baseline
```

- **Data** (`app/data.py`): synthetic daily series combining a linear trend,
  weekly seasonality (clinics see fewer bookings on weekends), and Gaussian
  noise — deterministic for a given seed.
- **Scaling** (`app/scaler.py`): min-max normalization to `[0, 1]`, which
  LSTMs train far more reliably on than raw counts with a trend.
- **Windowing** (`app/dataset.py`): a sliding window turns the series into
  `(window_size consecutive values) -> (next value)` supervised examples,
  wrapped in a `torch.utils.data.Dataset`.
- **Model** (`app/model.py`): a single-layer `nn.LSTM` followed by a linear
  head on the final timestep's hidden state — a standard sequence-to-one
  forecasting architecture.
- **Training** (`app/train.py`): Adam optimizer, MSE loss, returns per-epoch
  mean loss.
- **Forecasting** (`app/forecast.py`): iterative (autoregressive) multi-step
  forecasting — predict one step ahead, roll it into the input window, and
  repeat — the standard way to turn a single-step model into an N-step
  forecaster.
- **Evaluation** (`app/evaluate.py`): compares model MAE against a naive
  "predict the last observed value" baseline — a forecasting model that
  can't beat this trivial baseline isn't adding value, so comparing against
  it is standard practice rather than just reporting the training loss.

## Running locally

```bash
pip install -r requirements.txt
python -m app.main
```

or as an API (trains in-process at startup, a few seconds on CPU):

```bash
uvicorn app.main:app --reload
curl -X POST localhost:8000/forecast -H "Content-Type: application/json" -d '{"steps": 7}'
```

## Tests

```bash
pytest
```

Covers the synthetic-series generator (length, non-negativity,
determinism), the scaler's round-trip correctness (including a constant
-series edge case), sliding-window slicing, dataset tensor shapes, the
LSTM's forward-pass output shape, a training-loop smoke test (runs and
returns finite losses), the naive baseline and MAE utilities, multi-step
forecasting returning the requested number of steps, and the `/forecast`
API endpoint end to end. Note: the model's forecast *accuracy* isn't
asserted in tests — only that the pipeline runs correctly and produces
correctly-shaped output — since a few epochs on synthetic data isn't a
reliable target for a strict accuracy threshold.

## Going to production

Set `MODEL_PATH` to load a persisted checkpoint instead of retraining at
every startup (and to save freshly trained weights there when none exist
yet). Replace `generate_synthetic_series()` with a real historical series
pulled from wherever appointment data lives (CRM/EHR export, a
PostgreSQL query), and consider adding exogenous features (day-of-week,
holidays, marketing campaigns) as additional input channels alongside the
raw volume.

## Tech stack

Python, PyTorch (LSTM), FastAPI.
