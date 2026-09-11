from app.evaluate import mae, naive_baseline_predictions
from app.forecast import forecast_future
from app.model import LSTMForecaster
from app.scaler import MinMaxScaler


def test_naive_baseline_predictions():
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    predictions = naive_baseline_predictions(series, window_size=3)

    assert predictions == [3.0, 4.0, 5.0, 6.0]


def test_mae_computation():
    actual = [4.0, 5.0, 6.0, 7.0]
    predicted = [3.0, 4.0, 5.0, 6.0]

    assert mae(actual, predicted) == 1.0


def test_mae_requires_equal_length_inputs():
    try:
        mae([1.0, 2.0], [1.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_forecast_future_returns_requested_number_of_steps():
    model = LSTMForecaster(hidden_size=8)
    scaler = MinMaxScaler()
    scaler.fit([0.0, 100.0])
    last_window = [10.0, 20.0, 30.0, 40.0, 50.0]

    predictions = forecast_future(model, scaler, last_window, steps=5)

    assert len(predictions) == 5
    assert all(isinstance(p, float) for p in predictions)
