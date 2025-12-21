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

from ..agents.inventory import METHOD_INVENTORY, get_methods_for_regime
from ..environment.regime_generators import REGIME_OPTIMAL_METHODS


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
        max_methods: int = 3,
    ):
        """
        Initialize Oracle.

        Args:
            regime_to_methods: Mapping from regime to optimal methods.
                             If None, uses default from regime_generators.
            max_methods: Maximum methods to select per regime
        """
        self.regime_to_methods = regime_to_methods or REGIME_OPTIMAL_METHODS.copy()
        self.max_methods = max_methods

        # Validate that all methods exist in inventory
        for regime, methods in self.regime_to_methods.items():
            for method in methods:
                if method not in METHOD_INVENTORY:
                    raise ValueError(f"Method '{method}' not in inventory")

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
            return list(METHOD_INVENTORY.keys())[:self.max_methods]

        methods = self.regime_to_methods[regime]
        return methods[:self.max_methods]

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series,
        reward_fn,
    ) -> Dict:
        """
        Run Oracle through an entire episode.

        Args:
            prices: Price DataFrame
            regimes: Ground truth regime labels
            reward_fn: Function to compute reward

        Returns:
            Dict with performance metrics
        """
        total_reward = 0.0
        rewards = []

        # Need at least 2 bars for reward calculation
        window_size = 20

        for i in range(window_size, len(prices)):
            regime = regimes.iloc[i]
            methods = self.select(regime)

            # Get price window
            price_window = prices.iloc[i-window_size:i+1]

            # Compute reward
            reward = reward_fn(methods, price_window)
            total_reward += reward
            rewards.append(reward)

        return {
            "total_reward": total_reward,
            "mean_reward": np.mean(rewards) if rewards else 0.0,
            "std_reward": np.std(rewards) if rewards else 0.0,
            "n_steps": len(rewards),
            "sharpe": np.mean(rewards) / (np.std(rewards) + 1e-8) if rewards else 0.0,
        }

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
        max_methods: int = 3,
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
