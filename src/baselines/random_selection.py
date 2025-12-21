"""
Random Selection Baseline.

Population of agents that select methods randomly without learning.
This is the lower bound - represents no learning at all.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..agents.inventory import get_method_names


class RandomSelectionPopulation:
    """
    Population that selects methods randomly.

    This baseline shows the performance of random method selection,
    establishing a lower bound for what learning should beat.
    """

    def __init__(
        self,
        n_agents: int = 5,
        max_methods: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize random selection population.

        Args:
            n_agents: Number of agents
            max_methods: Max methods per agent per selection
            seed: Random seed
        """
        self.n_agents = n_agents
        self.max_methods = max_methods
        self.rng = np.random.default_rng(seed)
        self.method_names = get_method_names()
        self.name = "RandomSelection"

    def select(self) -> Dict[str, List[str]]:
        """
        Random method selection for all agents.

        Returns:
            Dict mapping agent_id to selected methods
        """
        selections = {}
        for i in range(self.n_agents):
            n_select = self.rng.integers(1, self.max_methods + 1)
            methods = list(self.rng.choice(
                self.method_names,
                size=min(n_select, len(self.method_names)),
                replace=False,
            ))
            selections[f"agent_{i}"] = methods
        return selections

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series,
        reward_fn,
        window_size: int = 20,
    ) -> Dict:
        """
        Run random selection through episode.

        Args:
            prices: Price data
            regimes: Regime labels
            reward_fn: Reward function
            window_size: Price window size

        Returns:
            Performance metrics
        """
        total_rewards = []
        win_counts = {f"agent_{i}": 0 for i in range(self.n_agents)}

        for i in range(window_size, len(prices)):
            # Random selection
            selections = self.select()

            # Compute rewards
            rewards = {}
            for agent_id, methods in selections.items():
                price_window = prices.iloc[i-window_size:i+1]
                reward = reward_fn(methods, price_window)
                rewards[agent_id] = reward

            # Find winner
            winner = max(rewards, key=rewards.get)
            win_counts[winner] += 1

            # Track best reward
            total_rewards.append(max(rewards.values()))

        return {
            "total_reward": sum(total_rewards),
            "mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "std_reward": np.std(total_rewards) if total_rewards else 0.0,
            "sharpe": np.mean(total_rewards) / (np.std(total_rewards) + 1e-8) if total_rewards else 0.0,
            "win_counts": win_counts,
            "n_steps": len(total_rewards),
        }
