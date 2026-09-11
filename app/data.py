"""Synthetic daily appointment-volume series generator.

Stands in for a real historical dataset (e.g. a clinic's daily booked-
appointment counts pulled from a CRM/EHR) so the repo is runnable and fully
reproducible with no external data file. Combines a linear trend, weekly
seasonality (clinics see fewer bookings on weekends), and Gaussian noise.
"""
from __future__ import annotations

import math
import random


def generate_synthetic_series(n_days: int = 365, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    series = []
    for day in range(n_days):
        trend = 0.03 * day
        weekly_seasonality = 10 * math.sin(2 * math.pi * day / 7)
        noise = rng.gauss(0, 2)
        value = 50 + trend + weekly_seasonality + noise
        series.append(max(value, 0.0))
    return series
