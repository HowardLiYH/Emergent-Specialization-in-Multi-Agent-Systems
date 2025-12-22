#!/usr/bin/env python3
"""
Unified Prediction Experiment v2 - Domain-Appropriate Methods.

Key improvement: Each domain uses prediction methods suited to its data characteristics:
- Finance: Momentum, Mean-Reversion, Volatility (appropriate for returns)
- Traffic: Hourly Persistence, Daily Pattern, Trend (captures 24h periodicity)
- Energy: Peak/Off-Peak, Seasonal, Load Tracking (captures demand patterns)
- Weather: Persistence, Seasonal, Volatility (captures weather patterns)

Statistical Framework:
- Paired t-test with Bonferroni correction
- 95% bootstrap confidence intervals
- Cohen's d effect size
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000,
                 confidence: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, alpha / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha / 2) * 100)
    return lower, upper


# ==============================================================================
# Abstract Prediction Method
# ==============================================================================

class PredictionMethod(ABC):
    """Base class for prediction methods."""

    def __init__(self, name: str, optimal_regimes: List[str] = None):
        self.name = name
        self.optimal_regimes = optimal_regimes or []

    @abstractmethod
    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        """
        Predict next value.

        Args:
            history: Historical values
            regime: Current regime label
            time_idx: Current time index (for periodic data)
            period: Period length (e.g., 24 for hourly data)
        """
        pass


# ==============================================================================
# Universal Methods (work across domains)
# ==============================================================================

class NaivePredictor(PredictionMethod):
    """Predicts last value (random walk baseline)."""
    def __init__(self):
        super().__init__("Naive")

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        return history[-1] if len(history) > 0 else 0.0


class MAPredictor(PredictionMethod):
    """Moving average prediction."""
    def __init__(self, window: int = 10):
        super().__init__(f"MA({window})")
        self.window = window

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < self.window:
            return np.mean(history) if len(history) > 0 else 0.0
        return np.mean(history[-self.window:])


# ==============================================================================
# Finance-Specific Methods
# ==============================================================================

class MomentumPredictor(PredictionMethod):
    """Predicts continuation of trend."""
    def __init__(self):
        super().__init__("Momentum", ["trend_up", "trend_down"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < 5:
            return history[-1] if len(history) > 0 else 0.0
        trend = (history[-1] - history[-5]) / 5
        return history[-1] + trend


class MeanRevertPredictor(PredictionMethod):
    """Predicts reversion to mean."""
    def __init__(self):
        super().__init__("MeanRevert", ["mean_revert"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < 10:
            return np.mean(history) if len(history) > 0 else 0.0
        ma = np.mean(history[-10:])
        return history[-1] + 0.3 * (ma - history[-1])


class VolatilityPredictor(PredictionMethod):
    """Predicts based on volatility bands."""
    def __init__(self):
        super().__init__("Volatility", ["volatile"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < 20:
            return history[-1] if len(history) > 0 else 0.0
        ma = np.mean(history[-20:])
        std = np.std(history[-20:])
        if std == 0:
            return history[-1]
        if history[-1] > ma + std:
            return history[-1] - 0.5 * std
        elif history[-1] < ma - std:
            return history[-1] + 0.5 * std
        return history[-1]


# ==============================================================================
# Traffic-Specific Methods (designed for 24h periodicity)
# ==============================================================================

class HourlyPersistencePredictor(PredictionMethod):
    """Predicts using same hour from previous day(s)."""
    def __init__(self, lookback_days: int = 1):
        super().__init__(f"HourlyPersist({lookback_days}d)", ["night", "midday"])
        self.lookback_days = lookback_days

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        # Use value from same hour, lookback_days ago
        lookback_idx = period * self.lookback_days
        if len(history) >= lookback_idx:
            return history[-lookback_idx]
        return history[-1] if len(history) > 0 else 0.0


class WeeklyPatternPredictor(PredictionMethod):
    """Predicts using weekly pattern (same hour, 7 days ago)."""
    def __init__(self):
        super().__init__("WeeklyPattern", ["weekend_active", "weekend_quiet"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        # Use value from 7 days ago (168 hours for hourly data)
        lookback_idx = 7 * period
        if len(history) >= lookback_idx:
            return history[-lookback_idx]
        # Fallback to 1 day ago
        if len(history) >= period:
            return history[-period]
        return history[-1] if len(history) > 0 else 0.0


class RushHourPredictor(PredictionMethod):
    """Predicts rush hour patterns."""
    def __init__(self):
        super().__init__("RushHour", ["morning_rush", "evening_rush"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        # Average of past week same hours
        if len(history) < period:
            return history[-1] if len(history) > 0 else 0.0

        # Get values from same hour in previous days
        same_hour_values = []
        for d in range(1, min(8, len(history) // period + 1)):
            idx = d * period
            if len(history) >= idx:
                same_hour_values.append(history[-idx])

        if same_hour_values:
            return np.mean(same_hour_values)
        return history[-1]


# ==============================================================================
# Energy-Specific Methods
# ==============================================================================

class PeakLoadPredictor(PredictionMethod):
    """Predicts based on peak/off-peak patterns."""
    def __init__(self):
        super().__init__("PeakLoad", ["peak_demand"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < period:
            return history[-1] if len(history) > 0 else 0.0
        # Use 24h ago value with trend adjustment
        val_24h_ago = history[-period]
        recent_trend = (history[-1] - history[-min(3, len(history))]) / 3 if len(history) >= 3 else 0
        return val_24h_ago + recent_trend


class LoadTrackingPredictor(PredictionMethod):
    """Tracks load patterns with exponential smoothing."""
    def __init__(self, alpha: float = 0.3):
        super().__init__("LoadTracking", ["normal"])
        self.alpha = alpha

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < 2:
            return history[-1] if len(history) > 0 else 0.0
        # Simple exponential smoothing
        smoothed = history[-1]
        for v in reversed(history[-10:]):
            smoothed = self.alpha * v + (1 - self.alpha) * smoothed
        return smoothed


class RenewableAwarePredictor(PredictionMethod):
    """Predicts considering renewable patterns (midday solar)."""
    def __init__(self):
        super().__init__("RenewableAware", ["renewable_surplus", "low_demand"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        if len(history) < period:
            return history[-1] if len(history) > 0 else 0.0
        # Average of past 3 days same hour
        same_hour = [history[-i*period] for i in range(1, 4) if len(history) >= i*period]
        if same_hour:
            return np.mean(same_hour)
        return history[-period]


# ==============================================================================
# Weather-Specific Methods
# ==============================================================================

class WeatherPersistencePredictor(PredictionMethod):
    """Temperature persistence (yesterday's temp is good predictor)."""
    def __init__(self):
        super().__init__("Persistence", ["stable_warm", "stable_cold"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 1) -> float:  # Daily data
        return history[-1] if len(history) > 0 else 0.0


class SeasonalWeatherPredictor(PredictionMethod):
    """Uses seasonal patterns for prediction."""
    def __init__(self):
        super().__init__("Seasonal", ["transition"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 1) -> float:
        if len(history) < 7:
            return history[-1] if len(history) > 0 else 0.0
        # 7-day moving average (captures seasonal trend)
        return np.mean(history[-7:])


class StormPredictor(PredictionMethod):
    """Predicts volatile weather using recent volatility."""
    def __init__(self):
        super().__init__("StormAware", ["volatile_storm"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 1) -> float:
        if len(history) < 5:
            return history[-1] if len(history) > 0 else 0.0
        # During storms, regress to mean
        ma = np.mean(history[-7:]) if len(history) >= 7 else np.mean(history)
        return history[-1] + 0.4 * (ma - history[-1])


# ==============================================================================
# Healthcare-Specific Methods (for ILI/Flu prediction)
# ==============================================================================

class FluPersistencePredictor(PredictionMethod):
    """ILI persistence - last week's rate is good baseline."""
    def __init__(self):
        super().__init__("FluPersist", ["off_season", "flu_low"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 52) -> float:  # Weekly data
        return history[-1] if len(history) > 0 else 0.0


class FluSeasonalPredictor(PredictionMethod):
    """Uses same week from previous year."""
    def __init__(self):
        super().__init__("FluSeasonal", ["flu_moderate"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 52) -> float:
        # Use same week from ~1 year ago
        if len(history) >= 52:
            return history[-52]
        # Fallback to 4-week MA
        if len(history) >= 4:
            return np.mean(history[-4:])
        return history[-1] if len(history) > 0 else 0.0


class FluPeakPredictor(PredictionMethod):
    """Predicts peak flu dynamics with volatility awareness."""
    def __init__(self):
        super().__init__("FluPeak", ["flu_peak"])

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 52) -> float:
        if len(history) < 4:
            return history[-1] if len(history) > 0 else 0.0
        # During peak: use recent trend
        recent_trend = (history[-1] - history[-4]) / 4
        # Cap extreme predictions
        pred = history[-1] + recent_trend
        return max(0.5, min(pred, 15.0))  # ILI rate range


# ==============================================================================
# Domain Configuration
# ==============================================================================

DOMAIN_CONFIGS = {
    "finance": {
        "methods": [
            MomentumPredictor(),
            MeanRevertPredictor(),
            VolatilityPredictor(),
        ],
        "baselines": [NaivePredictor(), MAPredictor(10)],
        "period": 1,  # Not periodic
        "target_column": "close",
    },
    "traffic": {
        "methods": [
            HourlyPersistencePredictor(1),
            WeeklyPatternPredictor(),
            RushHourPredictor(),
        ],
        "baselines": [NaivePredictor(), MAPredictor(24)],  # MA(24) for hourly
        "period": 24,  # 24-hour cycle
        "target_column": "trip_count",
    },
    "energy": {
        "methods": [
            PeakLoadPredictor(),
            LoadTrackingPredictor(),
            RenewableAwarePredictor(),
        ],
        "baselines": [NaivePredictor(), MAPredictor(24)],
        "period": 24,  # 24-hour cycle
        "target_column": "demand",
    },
    "weather": {
        "methods": [
            WeatherPersistencePredictor(),
            SeasonalWeatherPredictor(),
            StormPredictor(),
        ],
        "baselines": [NaivePredictor(), MAPredictor(7)],  # Weekly MA
        "period": 1,  # Daily data
        "target_column": "temperature",
    },
    "healthcare": {
        "methods": [
            FluPersistencePredictor(),
            FluSeasonalPredictor(),
            FluPeakPredictor(),
        ],
        "baselines": [NaivePredictor(), MAPredictor(4)],  # 4-week MA
        "period": 52,  # 52-week annual cycle
        "target_column": "ili_rate",
    },
}


# ==============================================================================
# Diverse Population (learns to specialize)
# ==============================================================================

class DiversePopulation:
    """Diverse population that learns to specialize based on regime."""

    def __init__(self, methods: List[PredictionMethod], n_agents: int = 8, seed: int = None):
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        self.methods = methods
        self.n_methods = len(methods)

        # Each agent has belief scores for each method
        self.agent_beliefs = []
        for _ in range(n_agents):
            # Random initialization with slight variation
            beliefs = np.ones(self.n_methods) * 0.5 + self.rng.uniform(-0.1, 0.1, self.n_methods)
            self.agent_beliefs.append(beliefs)

        # Track method usage per agent
        self.method_usage = [np.zeros(self.n_methods) for _ in range(n_agents)]

    def select_method_idx(self, agent_idx: int) -> int:
        """Select method index using softmax over beliefs."""
        beliefs = self.agent_beliefs[agent_idx]
        # Softmax with temperature
        exp_beliefs = np.exp(beliefs * 3.0)
        probs = exp_beliefs / np.sum(exp_beliefs)
        return self.rng.choice(self.n_methods, p=probs)

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> Tuple[float, Dict]:
        """
        Get ensemble prediction from all agents.

        Returns:
            prediction: Ensemble mean prediction
            selections: Dict of agent_idx -> (method_idx, prediction)
        """
        predictions = []
        selections = {}

        for i in range(self.n_agents):
            method_idx = self.select_method_idx(i)
            method = self.methods[method_idx]
            pred = method.predict(history, regime, time_idx, period)
            predictions.append(pred)
            selections[i] = (method_idx, pred)
            self.method_usage[i][method_idx] += 1

        return np.mean(predictions), selections

    def update(self, selections: Dict, actual: float, lr: float = 0.1):
        """Update beliefs based on prediction errors."""
        errors = {i: abs(pred - actual) for i, (_, pred) in selections.items()}
        mean_error = np.mean(list(errors.values()))

        for agent_idx, (method_idx, _) in selections.items():
            # Beat average = success
            if errors[agent_idx] < mean_error:
                self.agent_beliefs[agent_idx][method_idx] += lr
            else:
                self.agent_beliefs[agent_idx][method_idx] -= lr * 0.5

            # Clamp beliefs
            self.agent_beliefs[agent_idx] = np.clip(self.agent_beliefs[agent_idx], 0.1, 2.0)

    def get_specialization_index(self) -> float:
        """Compute average specialization index across agents."""
        sis = []
        for usage in self.method_usage:
            total = np.sum(usage)
            if total == 0:
                sis.append(0.0)
                continue
            probs = usage / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            max_entropy = np.log(self.n_methods)
            si = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
            sis.append(si)
        return np.mean(sis)


class HomogeneousPopulation:
    """Population using single best method (benchmark)."""

    def __init__(self, method: PredictionMethod):
        self.method = method

    def predict(self, history: np.ndarray, regime: str,
                time_idx: int = 0, period: int = 24) -> float:
        return self.method.predict(history, regime, time_idx, period)


# ==============================================================================
# Data Loading
# ==============================================================================

def load_domain_data(domain: str) -> Tuple[np.ndarray, List[str], int]:
    """
    Load data and compute regimes for a domain.

    Returns:
        values: The time series values
        regimes: List of regime labels
        period: Periodicity of data
    """
    data_dir = Path(__file__).parent.parent / "data"
    config = DOMAIN_CONFIGS[domain]

    if domain == "finance":
        filepath = data_dir / "bybit" / "Bybit_BTC.csv"
        if not filepath.exists():
            filepath = data_dir / "bybit" / "BTCUSDT_4H.csv"
        df = pd.read_csv(filepath)
        col = 'close' if 'close' in df.columns else 'Close'
        values = df[col].values.astype(float)

        # Compute regimes from returns
        returns = np.diff(values) / values[:-1]
        returns = np.insert(returns, 0, 0)

        regimes = []
        for i in range(len(values)):
            if i < 20:
                regimes.append('mean_revert')
                continue
            window_ret = returns[i-20:i]
            vol = np.std(window_ret)
            trend = np.mean(window_ret)

            if vol > 0.02:
                regimes.append('volatile')
            elif trend > 0.003:
                regimes.append('trend_up')
            elif trend < -0.003:
                regimes.append('trend_down')
            else:
                regimes.append('mean_revert')

    elif domain == "traffic":
        filepath = data_dir / "traffic" / "nyc_taxi" / "hourly_aggregated.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Traffic data not found: {filepath}")
        df = pd.read_csv(filepath)
        values = df['trip_count'].values.astype(float)

        # Use existing regime labels if available
        if 'regime' in df.columns:
            regimes = df['regime'].tolist()
        else:
            # Compute from hour_of_day
            regimes = []
            for i, row in df.iterrows():
                hour = row.get('hour_of_day', i % 24)
                if hour in [7, 8, 9]:
                    regimes.append('morning_rush')
                elif hour in [17, 18, 19]:
                    regimes.append('evening_rush')
                elif hour in [0, 1, 2, 3, 4, 5]:
                    regimes.append('night')
                else:
                    regimes.append('midday')

    elif domain == "energy":
        filepath = data_dir / "energy" / "eia_hourly_demand.csv"
        if not filepath.exists():
            filepath = data_dir / "energy" / "hourly_demand.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Energy data not found")
        df = pd.read_csv(filepath)
        values = df['demand'].values.astype(float)

        # Use existing regime labels if available
        if 'regime' in df.columns:
            regimes = df['regime'].tolist()
        else:
            # Compute regimes from demand levels
            median = np.median(values)
            q75 = np.percentile(values, 75)
            q25 = np.percentile(values, 25)
            regimes = []
            for v in values:
                if v > q75:
                    regimes.append('peak_demand')
                elif v < q25:
                    regimes.append('low_demand')
                else:
                    regimes.append('normal')

    elif domain == "weather":
        filepath = data_dir / "weather" / "noaa" / "daily_weather.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Weather data not found: {filepath}")
        df = pd.read_csv(filepath)
        values = df['temperature'].values.astype(float)
        
        # Use existing regime labels
        if 'regime' in df.columns:
            regimes = df['regime'].tolist()
        else:
            regimes = ['stable_warm'] * len(values)

    elif domain == "healthcare":
        filepath = data_dir / "healthcare" / "cdc_fluview" / "weekly_ili.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Healthcare data not found: {filepath}")
        df = pd.read_csv(filepath)
        values = df['ili_rate'].values.astype(float)
        
        # Use existing regime labels
        if 'regime' in df.columns:
            regimes = df['regime'].tolist()
        else:
            # Compute regimes from ILI rates
            regimes = []
            for v in values:
                if v > 4.0:
                    regimes.append('flu_peak')
                elif v > 2.5:
                    regimes.append('flu_moderate')
                elif v > 1.5:
                    regimes.append('flu_low')
                else:
                    regimes.append('off_season')

    else:
        raise ValueError(f"Unknown domain: {domain}")

    print(f"{domain.capitalize()}: Loaded {len(values)} data points")
    regime_counts = pd.Series(regimes).value_counts().to_dict()
    print(f"  Regime distribution: {regime_counts}")

    return values, regimes, config["period"]


# ==============================================================================
# Experiment Runner
# ==============================================================================

def run_prediction_experiment(domain: str, n_trials: int = 30,
                              n_iterations: int = 500) -> Dict:
    """Run prediction experiment for a domain."""
    print(f"\n{'='*60}")
    print(f"{domain.upper()} DOMAIN")
    print(f"{'='*60}")

    config = DOMAIN_CONFIGS[domain]
    values, regimes, period = load_domain_data(domain)

    # Ensure enough history
    min_history = max(period * 8, 50)  # At least 8 periods or 50 points
    n_points = min(n_iterations, len(values) - min_history - 1)

    if n_points < 100:
        print(f"  WARNING: Only {n_points} prediction points available")

    diverse_mses = []
    homo_mses = []
    naive_mses = []
    baseline_mses = []
    specialization_indices = []

    for trial in range(n_trials):
        # Initialize populations
        diverse_pop = DiversePopulation(config["methods"], n_agents=8, seed=trial)
        # Use the first method as "best single" for homogeneous
        homo_pop = HomogeneousPopulation(config["methods"][0])
        naive = NaivePredictor()
        baseline = config["baselines"][1]  # MA baseline

        diverse_errors = []
        homo_errors = []
        naive_errors = []
        baseline_errors = []

        for i in range(min_history, min_history + n_points):
            history = values[max(0, i-min_history):i]
            regime = regimes[i]
            actual = values[i]
            time_idx = i

            # Diverse prediction
            diverse_pred, selections = diverse_pop.predict(history, regime, time_idx, period)
            diverse_errors.append((diverse_pred - actual) ** 2)

            # Homogeneous prediction
            homo_pred = homo_pop.predict(history, regime, time_idx, period)
            homo_errors.append((homo_pred - actual) ** 2)

            # Baselines
            naive_pred = naive.predict(history, regime, time_idx, period)
            naive_errors.append((naive_pred - actual) ** 2)

            baseline_pred = baseline.predict(history, regime, time_idx, period)
            baseline_errors.append((baseline_pred - actual) ** 2)

            # Update diverse population
            diverse_pop.update(selections, actual)

        diverse_mses.append(np.mean(diverse_errors))
        homo_mses.append(np.mean(homo_errors))
        naive_mses.append(np.mean(naive_errors))
        baseline_mses.append(np.mean(baseline_errors))
        specialization_indices.append(diverse_pop.get_specialization_index())

    # Statistics
    diverse_arr = np.array(diverse_mses)
    homo_arr = np.array(homo_mses)
    naive_arr = np.array(naive_mses)
    baseline_arr = np.array(baseline_mses)

    t_stat, p_value = stats.ttest_rel(diverse_arr, homo_arr)
    effect_size = cohens_d(diverse_arr, homo_arr)
    ci_lower, ci_upper = bootstrap_ci(diverse_arr)

    # Bonferroni correction for 4 domains
    n_comparisons = 4
    alpha_corrected = 0.05 / n_comparisons
    significant = p_value < alpha_corrected if not np.isnan(p_value) else False

    results = {
        "domain": domain,
        "n_trials": n_trials,
        "n_points": n_points,
        "period": period,
        "strategies": {
            "Diverse": {
                "mse_mean": float(np.mean(diverse_arr)),
                "mse_std": float(np.std(diverse_arr)),
                "rmse_mean": float(np.sqrt(np.mean(diverse_arr))),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper)
            },
            "Homogeneous": {
                "mse_mean": float(np.mean(homo_arr)),
                "mse_std": float(np.std(homo_arr)),
                "rmse_mean": float(np.sqrt(np.mean(homo_arr)))
            },
            "Naive": {
                "mse_mean": float(np.mean(naive_arr)),
                "rmse_mean": float(np.sqrt(np.mean(naive_arr)))
            },
            "MA": {
                "mse_mean": float(np.mean(baseline_arr)),
                "rmse_mean": float(np.sqrt(np.mean(baseline_arr)))
            }
        },
        "specialization": {
            "si_mean": float(np.mean(specialization_indices)),
            "si_std": float(np.std(specialization_indices))
        },
        "comparison": {
            "diverse_vs_homo_pct": float((np.mean(homo_arr) - np.mean(diverse_arr)) / np.mean(homo_arr) * 100) if np.mean(homo_arr) != 0 else 0,
            "diverse_vs_naive_pct": float((np.mean(naive_arr) - np.mean(diverse_arr)) / np.mean(naive_arr) * 100) if np.mean(naive_arr) != 0 else 0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "cohens_d": float(effect_size),
            "significant_bonferroni": bool(significant)
        }
    }

    print(f"\n{domain.upper()} Results (MSE):")
    print(f"  Diverse:      {np.mean(diverse_arr):.6f} [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  Homogeneous:  {np.mean(homo_arr):.6f}")
    print(f"  Naive:        {np.mean(naive_arr):.6f}")
    print(f"  MA:           {np.mean(baseline_arr):.6f}")
    print(f"  Improvement vs Homo: {results['comparison']['diverse_vs_homo_pct']:.2f}%")
    print(f"  Improvement vs Naive: {results['comparison']['diverse_vs_naive_pct']:.2f}%")
    print(f"  p-value:      {results['comparison']['p_value']:.6f}")
    print(f"  Significant:  {significant}")
    print(f"  Avg SI:       {np.mean(specialization_indices):.3f}")

    return results


def run_all_domains(n_trials: int = 30, include_traffic: bool = False) -> Dict:
    """Run all domain experiments."""
    # Main 4 domains (Traffic excluded - see Appendix D)
    main_domains = ["finance", "energy", "weather", "healthcare"]
    
    print("="*70)
    print("UNIFIED PREDICTION EXPERIMENT v2")
    print(f"Domains: {', '.join(d.capitalize() for d in main_domains)}")
    print(f"Trials: {n_trials}, Bonferroni α = {0.05/len(main_domains):.4f}")
    print("="*70)

    all_results = {
        "experiment": "unified_prediction_v2",
        "date": pd.Timestamp.now().isoformat(),
        "config": {"n_trials": n_trials, "bonferroni_alpha": 0.05/len(main_domains)},
        "domains": {}
    }

    domains = main_domains
    if include_traffic:
        domains = domains + ["traffic"]  # For appendix analysis

    for domain in domains:
        try:
            results = run_prediction_experiment(domain, n_trials)
            all_results["domains"][domain] = results
        except Exception as e:
            print(f"Error in {domain}: {e}")
            import traceback
            traceback.print_exc()
            all_results["domains"][domain] = {"error": str(e)}

    # Summary
    print("\n" + "="*70)
    print("CROSS-DOMAIN SUMMARY (MSE)")
    print("="*70)
    print(f"{'Domain':<12} {'Diverse':<14} {'Homo':<14} {'Naive':<14} {'Δ% Homo':<10} {'SI':<6} {'Sig?'}")
    print("-"*82)

    for domain, res in all_results["domains"].items():
        if "error" not in res:
            s = res["strategies"]
            c = res["comparison"]
            si = res["specialization"]["si_mean"]
            print(f"{domain.capitalize():<12} {s['Diverse']['mse_mean']:<14.4f} "
                  f"{s['Homogeneous']['mse_mean']:<14.4f} "
                  f"{s['Naive']['mse_mean']:<14.4f} "
                  f"{c['diverse_vs_homo_pct']:>+8.1f}%  "
                  f"{si:.3f}  "
                  f"{'✓' if c['significant_bonferroni'] else '✗'}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "unified_prediction_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'results.json'}")

    return all_results


if __name__ == "__main__":
    run_all_domains(n_trials=30)
