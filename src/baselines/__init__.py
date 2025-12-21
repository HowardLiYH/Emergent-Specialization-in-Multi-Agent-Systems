"""
Baselines module: Comparison strategies for experiments.

Baselines:
1. BuyAndHold - Naive passive investment
2. MomentumStrategy - Classic 12-1 momentum
3. MeanReversionStrategy - Bollinger bands
4. SingleAgentRL - PPO/DQN single agent
5. FinRLAgent - SOTA RL trading (optional)
6. HomogeneousPopulation - Clone best agent
7. RandomSelection - Random method selection
8. OracleSpecialist - Perfect regime knowledge (upper bound)
"""

from .oracle import OracleSpecialist
from .simple_strategies import BuyAndHold, MomentumStrategy, MeanReversionStrategy
from .random_selection import RandomSelectionPopulation
from .homogeneous import HomogeneousPopulation

__all__ = [
    "OracleSpecialist",
    "BuyAndHold",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "RandomSelectionPopulation",
    "HomogeneousPopulation",
]
