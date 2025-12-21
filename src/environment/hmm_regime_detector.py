"""
HMM-based Regime Detection for Real Market Data.

Uses Gaussian Hidden Markov Model to detect market regimes from unlabeled data.
This enables applying our specialization framework to real markets where
regime labels are not known a priori.

Reference:
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series." Econometrica.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RegimeStats:
    """Statistics for a detected regime."""
    mean_return: float
    std_return: float
    mean_volatility: float
    count: int
    proportion: float


class HMMRegimeDetector:
    """
    Detect market regimes using Gaussian HMM.

    Features used:
    - Returns (for trend direction)
    - Rolling volatility (for regime stability)
    - Optional: volume, funding rate, etc.

    Regimes are mapped to semantic names based on characteristics:
    - trend_up: positive mean return, moderate volatility
    - trend_down: negative mean return, moderate volatility
    - mean_revert: near-zero mean return, low volatility
    - volatile: high volatility regardless of return
    """

    REGIME_NAMES = ["trend_up", "trend_down", "mean_revert", "volatile"]

    def __init__(
        self,
        n_regimes: int = 4,
        volatility_window: int = 20,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.volatility_window = volatility_window
        self.random_state = random_state

        # HMM parameters (learned)
        self.means_ = None
        self.covars_ = None
        self.transmat_ = None
        self.startprob_ = None

        # Regime mapping
        self.state_to_regime_: Dict[int, str] = {}
        self.regime_stats_: Dict[str, RegimeStats] = {}

        self._fitted = False

    def _compute_features(
        self,
        prices: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute features for HMM from price data.

        Returns:
            returns: Log returns
            volatility: Rolling standard deviation of returns
        """
        close = prices["close"].values

        # Log returns
        returns = np.diff(np.log(close))

        # Rolling volatility
        volatility = pd.Series(returns).rolling(
            self.volatility_window, min_periods=5
        ).std().fillna(0.01).values

        return returns, volatility

    def _simple_kmeans_init(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Simple k-means initialization for HMM states."""
        n_samples = len(features)

        # Initialize with random assignment
        rng = np.random.default_rng(self.random_state)
        labels = rng.integers(0, self.n_regimes, size=n_samples)

        # Run a few iterations of k-means
        for _ in range(10):
            # Compute centroids
            centroids = np.array([
                features[labels == k].mean(axis=0) if (labels == k).sum() > 0
                else features[rng.integers(0, n_samples)]
                for k in range(self.n_regimes)
            ])

            # Assign to nearest centroid
            distances = np.array([
                np.linalg.norm(features - c, axis=1)
                for c in centroids
            ]).T
            labels = distances.argmin(axis=1)

        return labels

    def fit(self, prices: pd.DataFrame) -> "HMMRegimeDetector":
        """
        Fit HMM to price data.

        Uses a simplified Gaussian mixture approach since full HMM
        requires hmmlearn which may not be installed.
        """
        returns, volatility = self._compute_features(prices)

        # Stack features
        features = np.column_stack([returns, volatility])

        # Remove NaN rows
        valid_mask = ~np.isnan(features).any(axis=1)
        features = features[valid_mask]

        # Initialize with k-means
        labels = self._simple_kmeans_init(features)

        # Compute Gaussian parameters for each state
        self.means_ = np.zeros((self.n_regimes, 2))
        self.covars_ = np.zeros((self.n_regimes, 2, 2))

        for k in range(self.n_regimes):
            mask = labels == k
            if mask.sum() > 2:
                self.means_[k] = features[mask].mean(axis=0)
                self.covars_[k] = np.cov(features[mask].T) + 1e-6 * np.eye(2)
            else:
                self.means_[k] = features.mean(axis=0)
                self.covars_[k] = np.cov(features.T) + 1e-6 * np.eye(2)

        # Estimate transition matrix
        self.transmat_ = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(len(labels) - 1):
            self.transmat_[labels[i], labels[i + 1]] += 1

        # Normalize rows
        row_sums = self.transmat_.sum(axis=1, keepdims=True)
        self.transmat_ = np.where(
            row_sums > 0,
            self.transmat_ / row_sums,
            1.0 / self.n_regimes
        )

        # Starting probabilities
        unique, counts = np.unique(labels[:100], return_counts=True)
        self.startprob_ = np.zeros(self.n_regimes)
        for u, c in zip(unique, counts):
            self.startprob_[u] = c
        self.startprob_ /= self.startprob_.sum()

        # Map states to regime names
        self._map_states_to_regimes(features, labels)

        self._fitted = True
        return self

    def _map_states_to_regimes(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Map HMM states to semantic regime names based on characteristics.

        Rules:
        - Highest volatility state -> volatile
        - Highest positive return state -> trend_up
        - Lowest (most negative) return state -> trend_down
        - Remaining state -> mean_revert
        """
        # Compute stats per state
        state_stats = []
        for k in range(self.n_regimes):
            mask = labels == k
            if mask.sum() > 0:
                mean_ret = features[mask, 0].mean()
                mean_vol = features[mask, 1].mean()
                count = mask.sum()
            else:
                mean_ret = 0
                mean_vol = 0
                count = 0
            state_stats.append({
                "state": k,
                "mean_return": mean_ret,
                "mean_volatility": mean_vol,
                "count": count,
            })

        # Sort by volatility (descending) to find volatile state
        by_vol = sorted(state_stats, key=lambda x: -x["mean_volatility"])
        volatile_state = by_vol[0]["state"]

        # Remaining states
        remaining = [s for s in state_stats if s["state"] != volatile_state]

        # Sort by return to find trend states
        by_ret = sorted(remaining, key=lambda x: -x["mean_return"])
        trend_up_state = by_ret[0]["state"]
        trend_down_state = by_ret[-1]["state"]

        # Remaining is mean_revert
        mean_revert_state = [
            s["state"] for s in remaining
            if s["state"] not in [trend_up_state, trend_down_state]
        ]
        if mean_revert_state:
            mean_revert_state = mean_revert_state[0]
        else:
            # If only 3 regimes, use the middle one
            mean_revert_state = by_ret[len(by_ret) // 2]["state"]

        # Create mapping
        self.state_to_regime_ = {
            volatile_state: "volatile",
            trend_up_state: "trend_up",
            trend_down_state: "trend_down",
            mean_revert_state: "mean_revert",
        }

        # Store regime stats
        total = len(labels)
        for s in state_stats:
            regime = self.state_to_regime_.get(s["state"], "unknown")
            self.regime_stats_[regime] = RegimeStats(
                mean_return=s["mean_return"],
                std_return=features[labels == s["state"], 0].std() if s["count"] > 0 else 0,
                mean_volatility=s["mean_volatility"],
                count=s["count"],
                proportion=s["count"] / total if total > 0 else 0,
            )

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for price data.

        Returns:
            Series with regime labels aligned to price index
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        returns, volatility = self._compute_features(prices)
        features = np.column_stack([returns, volatility])

        # Predict state for each observation using Gaussian likelihood
        n_samples = len(features)
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            if np.isnan(features[i]).any():
                labels[i] = 0
                continue

            # Compute log-likelihood for each state
            log_probs = np.zeros(self.n_regimes)
            for k in range(self.n_regimes):
                diff = features[i] - self.means_[k]
                cov_inv = np.linalg.inv(self.covars_[k])
                log_probs[k] = -0.5 * diff @ cov_inv @ diff

            labels[i] = log_probs.argmax()

        # Map to regime names
        regime_labels = [self.state_to_regime_.get(l, "unknown") for l in labels]

        # Align with price index (returns are 1 shorter)
        # Pad first element
        regime_labels = ["unknown"] + regime_labels

        return pd.Series(regime_labels, index=prices.index, name="regime")

    def fit_predict(self, prices: pd.DataFrame) -> pd.Series:
        """Fit and predict in one call."""
        self.fit(prices)
        return self.predict(prices)

    def get_regime_statistics(self) -> Dict[str, RegimeStats]:
        """Get statistics for each detected regime."""
        return self.regime_stats_

    def __repr__(self) -> str:
        if self._fitted:
            stats = ", ".join(
                f"{r}: {s.proportion:.1%}"
                for r, s in self.regime_stats_.items()
            )
            return f"HMMRegimeDetector(fitted, {stats})"
        return "HMMRegimeDetector(not fitted)"


def detect_regimes_for_bybit(
    csv_path: str,
    n_regimes: int = 4,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Bybit CSV and detect regimes.

    Returns:
        prices: DataFrame with OHLCV
        regimes: Series with regime labels
    """
    # Load data
    df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
    df = df.set_index("timestamp_utc")

    # Extract OHLCV
    prices = df[["open", "high", "low", "close", "volume"]].copy()

    # Detect regimes
    detector = HMMRegimeDetector(n_regimes=n_regimes)
    regimes = detector.fit_predict(prices)

    return prices, regimes, detector
