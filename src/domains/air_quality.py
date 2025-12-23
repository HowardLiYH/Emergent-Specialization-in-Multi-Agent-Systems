"""
Air Quality Domain Module - NYC PM2.5 Monitoring.

This module provides regime detection and prediction methods
for air quality data from Open-Meteo (REAL DATA).

Data Source: Open-Meteo Air Quality API
URL: https://open-meteo.com/en/docs/air-quality-api

Regimes (based on EPA PM2.5 AQI categories):
- good: PM2.5 < 12 μg/m³ (AQI 0-50)
- moderate: PM2.5 12-35 μg/m³ (AQI 51-100)
- unhealthy_sensitive: PM2.5 35-55 μg/m³ (AQI 101-150)
- unhealthy: PM2.5 ≥ 55 μg/m³ (AQI 151+)

Prediction Task: Hourly PM2.5 level forecasting
Metric: RMSE (Root Mean Square Error)
"""

import csv
from pathlib import Path
from typing import Dict, List, Callable
from datetime import datetime
from collections import defaultdict

import numpy as np

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "air_quality"


def load_data() -> Dict:
    """
    Load Open-Meteo real air quality data.

    Returns:
        Dict with 'timestamps', 'pm25', 'regimes' arrays
    """
    data_file = DATA_DIR / "openmeteo_real_air_quality.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Air quality data not found: {data_file}")

    timestamps = []
    pm25_values = []
    regimes = []

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row['timestamp']))
            pm25_values.append(float(row['pm25']))
            regimes.append(row['regime'])

    return {
        'timestamps': np.array(timestamps),
        'pm25': np.array(pm25_values),
        'regimes': np.array(regimes),
        'data_source': 'Open-Meteo (REAL)',
        'records': len(timestamps),
    }


def detect_regime(pm25: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
    """
    Detect regime based on PM2.5 levels (EPA AQI categories).

    Args:
        pm25: Array of PM2.5 values in μg/m³
        timestamps: Optional array of datetime objects

    Returns:
        Array of regime labels
    """
    regimes = []

    for val in pm25:
        if val >= 55:
            regimes.append('unhealthy')
        elif val >= 35:
            regimes.append('unhealthy_sensitive')
        elif val >= 12:
            regimes.append('moderate')
        else:
            regimes.append('good')

    return np.array(regimes)


def get_prediction_methods() -> Dict[str, Callable]:
    """
    Get prediction methods for air quality domain.

    Returns:
        Dict mapping method name to prediction function
    """
    return {
        'persistence': persistence_predictor,
        'hourly_average': hourly_average_predictor,
        'moving_average': moving_average_predictor,
        'regime_average': regime_average_predictor,
        'exponential_smoothing': exponential_smoothing,
    }


def persistence_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Naive persistence: predict last value.
    """
    if len(history) == 0:
        return np.zeros(horizon)
    return np.full(horizon, history[-1])


def hourly_average_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Hourly average: predict based on historical average for each hour.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)

    hourly_avg = defaultdict(list)
    for i, val in enumerate(history):
        hourly_avg[i % 24].append(val)

    hourly_avg = {h: np.mean(vals) for h, vals in hourly_avg.items()}

    predictions = []
    last_hour = len(history) % 24
    for h in range(horizon):
        hour = (last_hour + h) % 24
        predictions.append(hourly_avg.get(hour, np.mean(history)))

    return np.array(predictions)


def moving_average_predictor(history: np.ndarray, horizon: int = 1,
                              window: int = 24) -> np.ndarray:
    """
    Moving average predictor.

    Good for smoothing noisy air quality readings.
    """
    if len(history) < window:
        return persistence_predictor(history, horizon)

    ma = np.mean(history[-window:])
    return np.full(horizon, ma)


def regime_average_predictor(history: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Regime-based average: predict based on current regime's historical average.
    """
    if len(history) < 24:
        return persistence_predictor(history, horizon)

    # Determine current regime
    current_val = history[-1]
    if current_val >= 55:
        current_regime = 'unhealthy'
    elif current_val >= 35:
        current_regime = 'unhealthy_sensitive'
    elif current_val >= 12:
        current_regime = 'moderate'
    else:
        current_regime = 'good'

    # Compute regime averages
    regime_vals = defaultdict(list)
    for val in history:
        if val >= 55:
            regime = 'unhealthy'
        elif val >= 35:
            regime = 'unhealthy_sensitive'
        elif val >= 12:
            regime = 'moderate'
        else:
            regime = 'good'
        regime_vals[regime].append(val)

    regime_avg = {r: np.mean(vals) for r, vals in regime_vals.items()}

    return np.full(horizon, regime_avg.get(current_regime, np.mean(history)))


def exponential_smoothing(history: np.ndarray, horizon: int = 1,
                          alpha: float = 0.3) -> np.ndarray:
    """
    Simple exponential smoothing.
    """
    if len(history) == 0:
        return np.zeros(horizon)

    level = history[0]
    for val in history[1:]:
        level = alpha * val + (1 - alpha) * level

    return np.full(horizon, level)


def compute_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.
    """
    if len(predictions) == 0 or len(actuals) == 0:
        return float('inf')

    mse = np.mean((predictions - actuals) ** 2)
    return float(np.sqrt(mse))


def get_regime_list() -> List[str]:
    """Get list of regimes for this domain."""
    return ['good', 'moderate', 'unhealthy_sensitive', 'unhealthy']


def get_metric_name() -> str:
    """Get primary metric name."""
    return 'RMSE (μg/m³)'


def is_lower_better() -> bool:
    """Whether lower metric values are better."""
    return True


def is_real_data() -> bool:
    """Check if using real data."""
    data_file = DATA_DIR / "openmeteo_real_air_quality.csv"
    return data_file.exists()
