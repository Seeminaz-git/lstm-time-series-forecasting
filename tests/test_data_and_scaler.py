import pytest

from app.data import generate_synthetic_series
from app.scaler import MinMaxScaler


def test_generate_synthetic_series_length_and_nonnegativity():
    series = generate_synthetic_series(n_days=100, seed=1)

    assert len(series) == 100
    assert all(v >= 0 for v in series)


def test_generate_synthetic_series_is_deterministic_for_a_given_seed():
    series_a = generate_synthetic_series(n_days=50, seed=7)
    series_b = generate_synthetic_series(n_days=50, seed=7)

    assert series_a == series_b


def test_scaler_round_trip():
    scaler = MinMaxScaler()
    scaler.fit([0, 5, 10])

    transformed = scaler.transform([0, 5, 10])
    assert transformed == pytest.approx([0.0, 0.5, 1.0])

    restored = scaler.inverse_transform(transformed)
    assert restored == pytest.approx([0.0, 5.0, 10.0])


def test_scaler_handles_constant_series_without_division_by_zero():
    scaler = MinMaxScaler()
    scaler.fit([5, 5, 5])

    transformed = scaler.transform([5, 5, 5])
    assert transformed == pytest.approx([0.0, 0.0, 0.0])
