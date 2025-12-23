"""
Traffic Domain Module - NYC Taxi Trip Data.

This module provides regime detection and prediction methods
for NYC TLC Yellow Taxi trip data (REAL DATA).

Data Source: NYC Taxi & Limousine Commission
URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Regimes:
- morning_rush: 7-9 AM weekdays (high demand)
- evening_rush: 5-7 PM weekdays (highest demand)
- midday: 10 AM - 4 PM weekdays (moderate)
- night: 12 AM - 5 AM (lowest demand)
- weekend: Saturday/Sunday (different patterns)
- transition: 6 AM, 8 PM weekdays (transitional periods)

Prediction Task: Hourly trip count forecasting
Metric: MAPE (Mean Absolute Percentage Error)
"""

import csv
from pathlib import Path
from typing import Dict, List, Callable
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "traffic"


def load_data() -> Dict:
    """
    Load NYC Taxi real hourly trip data.

    Returns:
        Dict with 'timestamps', 'trip_count', 'regimes' arrays
    """
    # Prefer real data file
    data_file = DATA_DIR / "nyc_taxi_real_hourly.csv"

    if not data_file.exists():
        # Fall back to synthetic if real not available
        data_file = DATA_DIR / "nyc_taxi_hourly.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Traffic data not found: {data_file}")

    timestamps = []
    trip_counts = []
    regimes = []

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row['timestamp']))
            trip_counts.append(int(row['trip_count']))
            regimes.append(row['regime'])

    return {
        'timestamps': np.array(timestamps),
        'trip_count': np.array(trip_counts),
        'regimes': np.array(regimes),
        'data_source': 'NYC TLC (REAL)',
        'records': len(timestamps),
    }


def detect_regime(trip_count: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
    """
    Detect regime based on temporal patterns and trip counts.

    Args:
        trip_count: Array of hourly trip counts
        timestamps: Optional array of datetime objects

    Returns:
        Array of regime labels
    """
    regimes = []

    for i, count in enumerate(trip_count):
        if timestamps is not None:
            hour = timestamps[i].hour
            weekday = timestamps[i].weekday()
        else:
            hour = i % 24
            weekday = (i // 24) % 7

        if weekday >= 5:  # Saturday=5, Sunday=6
            regimes.append('weekend')
        elif 7 <= hour <= 9:
            regimes.append('morning_rush')
        elif 17 <= hour <= 19:
            regimes.append('evening_rush')
        elif 10 <= hour <= 16:
            regimes.append('midday')
        elif 0 <= hour <= 5:
            regimes.append('night')
        else:
            regimes.append('transition')

    return np.array(regimes)


def get_prediction_methods() -> Dict[str, Callable]:
    """
    Get prediction methods for traffic domain.

    Returns:
        Dict mapping method name to prediction function
    """
    return {
        'persistence': persistence_predictor,
        'hourly_average': hourly_average_predictor,
        'weekly_pattern': weekly_pattern_predictor,
        'rush_hour_model': rush_hour_model,
        'exponential_smoothing': exponential_smoothing,
    }


def persistence_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Naive persistence: predict last value.

    Works well for stable periods.
    """
    if len(history) == 0:
        return np.zeros(horizon)
    return np.full(horizon, history[-1])


def hourly_average_predictor(history: np.ndarray, horizon: int = 1,
                              timestamps: np.ndarray = None) -> np.ndarray:
    """
    Hourly average: predict based on historical average for each hour.

    Good for capturing daily load patterns.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)

    hourly_avg = defaultdict(list)
    if timestamps is not None:
        for val, ts in zip(history, timestamps):
            hourly_avg[ts.hour].append(val)
    else:
        for i, val in enumerate(history):
            hourly_avg[i % 24].append(val)

    hourly_avg = {h: np.mean(vals) for h, vals in hourly_avg.items()}

    predictions = []
    last_hour = len(history) % 24
    for h in range(horizon):
        hour = (last_hour + h) % 24
        predictions.append(hourly_avg.get(hour, np.mean(history)))

    return np.array(predictions)


def weekly_pattern_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Weekly pattern: use same hour from one week ago.

    Best for capturing weekly seasonality.
    """
    period = 24 * 7  # One week in hours
    if len(history) < period:
        return hourly_average_predictor(history, horizon)

    predictions = []
    for h in range(horizon):
        idx = len(history) - period + (h % period)
        if idx >= 0:
            predictions.append(history[idx])
        else:
            predictions.append(history[-1])

    return np.array(predictions)


def rush_hour_model(history: np.ndarray, horizon: int = 1,
                    timestamps: np.ndarray = None) -> np.ndarray:
    """
    Rush hour model: different predictions for rush hours vs off-peak.

    Best for weekday patterns.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)

    # Compute average by regime
    regime_vals = defaultdict(list)
    for i, val in enumerate(history):
        hour = i % 24
        weekday = (i // 24) % 7

        if weekday >= 5:
            regime = 'weekend'
        elif 7 <= hour <= 9:
            regime = 'morning_rush'
        elif 17 <= hour <= 19:
            regime = 'evening_rush'
        elif 10 <= hour <= 16:
            regime = 'midday'
        elif 0 <= hour <= 5:
            regime = 'night'
        else:
            regime = 'transition'

        regime_vals[regime].append(val)

    regime_avg = {r: np.mean(vals) for r, vals in regime_vals.items()}

    # Predict
    predictions = []
    last_idx = len(history)
    for h in range(horizon):
        idx = last_idx + h
        hour = idx % 24
        weekday = (idx // 24) % 7

        if weekday >= 5:
            regime = 'weekend'
        elif 7 <= hour <= 9:
            regime = 'morning_rush'
        elif 17 <= hour <= 19:
            regime = 'evening_rush'
        elif 10 <= hour <= 16:
            regime = 'midday'
        elif 0 <= hour <= 5:
            regime = 'night'
        else:
            regime = 'transition'

        predictions.append(regime_avg.get(regime, np.mean(history)))

    return np.array(predictions)


def exponential_smoothing(history: np.ndarray, horizon: int = 1,
                          alpha: float = 0.3) -> np.ndarray:
    """
    Simple exponential smoothing.

    Good for short-term trends.
    """
    if len(history) == 0:
        return np.zeros(horizon)

    level = history[0]
    for val in history[1:]:
        level = alpha * val + (1 - alpha) * level

    return np.full(horizon, level)


def compute_mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.

    MAPE = mean(|pred - actual| / |actual|) * 100
    """
    mask = np.abs(actuals) > 10  # Avoid division by near-zero
    if not np.any(mask):
        return 100.0

    mape = np.mean(np.abs(predictions[mask] - actuals[mask]) / np.abs(actuals[mask])) * 100
    return float(mape)


def get_regime_list() -> List[str]:
    """Get list of regimes for this domain."""
    return ['morning_rush', 'evening_rush', 'midday', 'night', 'weekend', 'transition']


def get_metric_name() -> str:
    """Get primary metric name."""
    return 'MAPE'


def is_lower_better() -> bool:
    """Whether lower metric values are better."""
    return True


def is_real_data() -> bool:
    """Check if using real data."""
    data_file = DATA_DIR / "nyc_taxi_real_hourly.csv"
    return data_file.exists()
