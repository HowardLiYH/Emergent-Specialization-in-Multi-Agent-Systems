"""
Oracle Specialist Baseline.

The Oracle has perfect knowledge of the current market regime and
always selects the optimal methods for that regime. This represents
the theoretical upper bound on performance.

This baseline is critical for understanding:
1. How close emergent specialists get to optimal
2. The gap between learned and perfect regime detection
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..agents.inventory_v2 import METHOD_INVENTORY_V2


# Optimal methods for each regime (based on method design)
REGIME_OPTIMAL_METHODS_V2 = {
    "trend_up": ["BuyMomentum", "TrendRider"],
    "trend_down": ["SellMomentum", "TrendRider"],
    "mean_revert": ["MeanRevert", "BollingerMR", "RSI_MR"],
    "volatile": ["VolBreakout", "VolScalp", "VolFade"],
}


class OracleSpecialist:
    """
    Oracle baseline with perfect regime knowledge.

    The Oracle:
    1. Knows the true regime at each timestep (from ground truth labels)
    2. Always selects the optimal methods for that regime
    3. Represents the theoretical upper bound on method selection

    Usage:
        oracle = OracleSpecialist()
        methods = oracle.select(regime="trend_up")
        reward = compute_reward(methods, prices)
    """

    def __init__(
        self,
        regime_to_methods: Optional[Dict[str, List[str]]] = None,
        max_methods: int = 2,
    ):
        """
        Initialize Oracle.

        Args:
            regime_to_methods: Mapping from regime to optimal methods.
                             If None, uses default optimal methods.
            max_methods: Maximum methods to select per regime
        """
        self.regime_to_methods = regime_to_methods or REGIME_OPTIMAL_METHODS_V2.copy()
        self.max_methods = max_methods
        self.total_reward = 0.0
        self.rewards = []

        # Validate that all methods exist in inventory
        for regime, methods in self.regime_to_methods.items():
            for method in methods:
                if method not in METHOD_INVENTORY_V2:
                    print(f"Warning: Method '{method}' not in inventory, skipping")

    def select(self, regime: str) -> List[str]:
        """
        Select optimal methods for the given regime.

        Args:
            regime: Current market regime (ground truth)

        Returns:
            List of optimal method names
        """
        if regime not in self.regime_to_methods:
            # Unknown regime - return first available methods
            return list(METHOD_INVENTORY_V2.keys())[:self.max_methods]

        methods = self.regime_to_methods[regime]
        # Filter to valid methods
        valid_methods = [m for m in methods if m in METHOD_INVENTORY_V2]
        return valid_methods[:self.max_methods]

    def run_iteration(
        self,
        prices: pd.DataFrame,
        regime: str,
        reward_fn,
    ) -> Dict:
        """Run single iteration."""
        methods = self.select(regime)
        reward = reward_fn(methods, prices)
        self.total_reward += reward
        self.rewards.append(reward)

        return {
            "methods": methods,
            "reward": reward,
            "regime": regime,
        }

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series,
        reward_fn,
        window_size: int = 20,
    ) -> Dict:
        """
        Run Oracle through an entire episode.

        Args:
            prices: Price DataFrame
            regimes: Ground truth regime labels
            reward_fn: Function to compute reward
            window_size: Price window size

        Returns:
            Dict with performance metrics
        """
        self.total_reward = 0.0
        self.rewards = []

        for i in range(window_size, len(prices) - 1):
            regime = regimes.iloc[i]
            price_window = prices.iloc[i-window_size:i+1]
            self.run_iteration(price_window, regime, reward_fn)

        return {
            "total_reward": self.total_reward,
            "mean_reward": np.mean(self.rewards) if self.rewards else 0.0,
            "std_reward": np.std(self.rewards) if self.rewards else 0.0,
            "n_steps": len(self.rewards),
            "sharpe": np.mean(self.rewards) / (np.std(self.rewards) + 1e-8) if self.rewards else 0.0,
        }

    def get_total_reward(self) -> float:
        """Get accumulated total reward."""
        return self.total_reward

    def reset(self):
        """Reset accumulated rewards."""
        self.total_reward = 0.0
        self.rewards = []

    def __repr__(self) -> str:
        return f"OracleSpecialist(regimes={list(self.regime_to_methods.keys())})"


class NoisyOracle(OracleSpecialist):
    """
    Oracle with imperfect regime detection.

    This represents a more realistic upper bound where regime
    detection has some error rate.
    """

    def __init__(
        self,
        regime_to_methods: Optional[Dict[str, List[str]]] = None,
        max_methods: int = 2,
        error_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize Noisy Oracle.

        Args:
            regime_to_methods: Mapping from regime to optimal methods
            max_methods: Maximum methods to select
            error_rate: Probability of detecting wrong regime
            seed: Random seed
        """
        super().__init__(regime_to_methods, max_methods)
        self.error_rate = error_rate
        self.rng = np.random.default_rng(seed)
        self.all_regimes = list(self.regime_to_methods.keys())

    def select(self, regime: str) -> List[str]:
        """Select methods with possible regime detection error."""
        # With error_rate probability, pick a random regime instead
        if self.rng.random() < self.error_rate:
            regime = self.rng.choice(self.all_regimes)

        return super().select(regime)


class AdaptiveOracle(OracleSpecialist):
    """
    Oracle that learns optimal methods from experience.

    This represents a realistic upper bound where the agent
    knows the regime but must learn which methods are optimal.
    """

    def __init__(
        self,
        max_methods: int = 2,
        learning_rate: float = 0.1,
        seed: Optional[int] = None,
    ):
        # Start with no prior knowledge
        super().__init__(regime_to_methods={}, max_methods=max_methods)
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Method values per regime (learned)
        self.method_values: Dict[str, Dict[str, float]] = {}
        self.all_methods = list(METHOD_INVENTORY_V2.keys())

    def select(self, regime: str) -> List[str]:
        """Select methods based on learned values."""
        if regime not in self.method_values:
            self.method_values[regime] = {m: 0.0 for m in self.all_methods}

        # Epsilon-greedy selection
        if self.rng.random() < 0.1:
            return list(self.rng.choice(self.all_methods, size=self.max_methods, replace=False))

        # Select top methods by value
        values = self.method_values[regime]
        sorted_methods = sorted(values.keys(), key=lambda m: values[m], reverse=True)
        return sorted_methods[:self.max_methods]

    def update(self, regime: str, methods: List[str], reward: float):
        """Update method values based on reward."""
        if regime not in self.method_values:
            self.method_values[regime] = {m: 0.0 for m in self.all_methods}

        for method in methods:
            old_value = self.method_values[regime][method]
            self.method_values[regime][method] = old_value + self.learning_rate * (reward - old_value)
