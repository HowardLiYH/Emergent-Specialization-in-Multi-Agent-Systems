"""
Environment module: Synthetic and real market environments.

- SyntheticMarketEnvironment: Controllable regime-switching market
- RegimeGenerators: Individual regime dynamics (trend, mean-revert, volatile)
- RealDataLoader: Load Bybit CSV data for validation
"""

from .synthetic_market import SyntheticMarketEnvironment
from .regime_generators import (
    TrendRegime,
    MeanRevertRegime,
    VolatileRegime,
    SidewaysRegime,
)

__all__ = [
    "SyntheticMarketEnvironment",
    "TrendRegime",
    "MeanRevertRegime",
    "VolatileRegime",
    "SidewaysRegime",
]
