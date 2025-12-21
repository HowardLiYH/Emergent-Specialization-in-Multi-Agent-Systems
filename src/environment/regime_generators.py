"""
Regime Generators for Synthetic Market Environment.

Each regime generates price dynamics with distinct statistical properties,
creating a testbed for agent specialization.

Theoretical References:
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series." Econometrica, 57(2), 357-384.
- Cont, R. (2001). "Empirical properties of asset returns: stylized facts
  and statistical issues." Quantitative Finance, 1(2), 223-236.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class RegimeConfig:
    """Configuration for a market regime."""
    name: str
    drift: float  # Expected return per bar
    volatility: float  # Standard deviation per bar
    mean_reversion_speed: float = 0.0  # Ornstein-Uhlenbeck theta
    mean_reversion_level: float = 0.0  # Long-term mean
    jump_intensity: float = 0.0  # Poisson jump rate
    jump_size_mean: float = 0.0  # Jump size mean
    jump_size_std: float = 0.0  # Jump size std


class BaseRegime(ABC):
    """Abstract base class for regime generators."""

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    def generate_returns(
        self,
        n_bars: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate array of log returns for this regime."""
        pass

    def generate_prices(
        self,
        n_bars: int,
        initial_price: float = 100.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate price series from returns."""
        returns = self.generate_returns(n_bars, seed)
        log_prices = np.log(initial_price) + np.cumsum(returns)
        return np.exp(log_prices)

    @property
    def optimal_methods(self) -> list:
        """Methods that perform best in this regime."""
        return []


class TrendRegime(BaseRegime):
    """
    Trending market regime using Geometric Brownian Motion.

    Model: dS/S = μdt + σdW

    Characteristics:
    - Persistent directional movement
    - Momentum strategies work well
    - Breakout strategies capture moves

    Optimal methods: Momentum, Breakout, TrendFollowing
    """

    def __init__(
        self,
        drift: float = 0.001,  # 0.1% per bar expected return
        volatility: float = 0.02,  # 2% per bar volatility
        direction: str = "up",  # "up" or "down"
    ):
        if direction == "down":
            drift = -abs(drift)

        config = RegimeConfig(
            name=f"trend_{direction}",
            drift=drift,
            volatility=volatility,
        )
        super().__init__(config)
        self.direction = direction

    def generate_returns(
        self,
        n_bars: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # GBM: log returns are normally distributed
        returns = rng.normal(
            loc=self.config.drift,
            scale=self.config.volatility,
            size=n_bars,
        )
        return returns

    @property
    def optimal_methods(self) -> list:
        return ["Momentum", "Breakout", "TrendFollowing"]


class MeanRevertRegime(BaseRegime):
    """
    Mean-reverting market regime using Ornstein-Uhlenbeck process.

    Model: dX = θ(μ - X)dt + σdW

    Characteristics:
    - Prices oscillate around a mean
    - Overshoots correct quickly
    - Mean reversion strategies work well

    Optimal methods: MeanReversion, RSI, BollingerBands

    Reference: Uhlenbeck, G.E. & Ornstein, L.S. (1930).
              "On the Theory of Brownian Motion." Physical Review.
    """

    def __init__(
        self,
        mean_reversion_speed: float = 0.1,  # How fast it reverts
        volatility: float = 0.015,
        mean_level: float = 0.0,  # Long-term mean of log returns
    ):
        config = RegimeConfig(
            name="mean_revert",
            drift=0.0,
            volatility=volatility,
            mean_reversion_speed=mean_reversion_speed,
            mean_reversion_level=mean_level,
        )
        super().__init__(config)

    def generate_returns(
        self,
        n_bars: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        theta = self.config.mean_reversion_speed
        mu = self.config.mean_reversion_level
        sigma = self.config.volatility

        # Ornstein-Uhlenbeck discretization
        returns = np.zeros(n_bars)
        x = 0.0  # Current deviation from mean

        for i in range(n_bars):
            dx = theta * (mu - x) + sigma * rng.standard_normal()
            x += dx
            returns[i] = dx

        return returns

    @property
    def optimal_methods(self) -> list:
        return ["MeanReversion", "RSI", "BollingerBands"]


class VolatileRegime(BaseRegime):
    """
    High volatility regime with near-zero drift.

    Model: dS/S = σdW (high σ)

    Characteristics:
    - Large price swings
    - No clear direction
    - Risk management critical
    - Best to stay flat or use volatility-scaled positions

    Optimal methods: StayFlat, VolatilityScaling, ReduceExposure
    """

    def __init__(
        self,
        volatility: float = 0.05,  # 5% per bar - very high
        drift: float = 0.0,
    ):
        config = RegimeConfig(
            name="volatile",
            drift=drift,
            volatility=volatility,
        )
        super().__init__(config)

    def generate_returns(
        self,
        n_bars: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # High volatility, near-zero mean
        returns = rng.normal(
            loc=self.config.drift,
            scale=self.config.volatility,
            size=n_bars,
        )

        # Add occasional jumps for extra volatility
        jump_mask = rng.random(n_bars) < 0.05  # 5% chance of jump
        jump_sizes = rng.normal(0, 0.03, n_bars)  # 3% jump std
        returns[jump_mask] += jump_sizes[jump_mask]

        return returns

    @property
    def optimal_methods(self) -> list:
        return ["StayFlat", "VolatilityScaling", "ReduceExposure"]


class SidewaysRegime(BaseRegime):
    """
    Low volatility, no trend regime.

    Model: dS/S = σdW (low σ)

    Characteristics:
    - Tight price range
    - No clear direction
    - Range-bound trading possible
    - Most momentum/trend strategies fail

    Optimal methods: RangeTrading, StayFlat, WaitForBreakout
    """

    def __init__(
        self,
        volatility: float = 0.008,  # Very low volatility
        drift: float = 0.0,
    ):
        config = RegimeConfig(
            name="sideways",
            drift=drift,
            volatility=volatility,
        )
        super().__init__(config)

    def generate_returns(
        self,
        n_bars: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        returns = rng.normal(
            loc=self.config.drift,
            scale=self.config.volatility,
            size=n_bars,
        )
        return returns

    @property
    def optimal_methods(self) -> list:
        return ["RangeTrading", "StayFlat", "WaitForBreakout"]


# Mapping from regime name to optimal methods (for Oracle baseline)
REGIME_OPTIMAL_METHODS = {
    "trend_up": ["Momentum", "Breakout", "TrendFollowing"],
    "trend_down": ["Momentum", "Breakout", "TrendFollowing"],  # Same, just short
    "mean_revert": ["MeanReversion", "RSI", "BollingerBands"],
    "volatile": ["StayFlat", "VolatilityScaling", "ReduceExposure"],
    "sideways": ["RangeTrading", "StayFlat", "WaitForBreakout"],
}


def create_regime(name: str, **kwargs) -> BaseRegime:
    """Factory function to create regime by name."""
    regime_map = {
        "trend_up": lambda: TrendRegime(direction="up", **kwargs),
        "trend_down": lambda: TrendRegime(direction="down", **kwargs),
        "mean_revert": lambda: MeanRevertRegime(**kwargs),
        "volatile": lambda: VolatileRegime(**kwargs),
        "sideways": lambda: SidewaysRegime(**kwargs),
    }

    if name not in regime_map:
        raise ValueError(f"Unknown regime: {name}. Choose from {list(regime_map.keys())}")

    return regime_map[name]()
