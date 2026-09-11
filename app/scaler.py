"""Min-max scaling. LSTMs train far more reliably on normalized inputs than
on raw counts with a trend, so every value is scaled to [0, 1] before it
reaches the model and un-scaled again for reporting."""
from __future__ import annotations


class MinMaxScaler:
    def __init__(self) -> None:
        self.min_: float | None = None
        self.max_: float | None = None

    def fit(self, data: list[float]) -> None:
        self.min_ = min(data)
        self.max_ = max(data)

    def _range(self) -> float:
        return (self.max_ - self.min_) or 1.0

    def transform(self, data: list[float]) -> list[float]:
        if self.min_ is None:
            raise RuntimeError("Scaler has not been fit yet")
        range_ = self._range()
        return [(x - self.min_) / range_ for x in data]

    def inverse_transform(self, data: list[float]) -> list[float]:
        if self.min_ is None:
            raise RuntimeError("Scaler has not been fit yet")
        range_ = self._range()
        return [x * range_ + self.min_ for x in data]
