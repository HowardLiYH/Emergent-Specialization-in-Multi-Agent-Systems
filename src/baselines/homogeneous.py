"""
Homogeneous Population Baseline.

A population where all agents are clones of the best agent.
This tests whether diversity provides value beyond just having
the best strategy.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..agents.method_selector import MethodSelector
from ..agents.population import Population, PopulationConfig


class HomogeneousPopulation:
    """
    Homogeneous population baseline.

    All agents have identical preferences (cloned from best).
    This tests the value of diversity specifically.

    If PopAgent (diverse) beats HomogeneousPopulation, it shows
    that diversity itself provides value.
    """

    def __init__(
        self,
        source_population: Optional[Population] = None,
        n_agents: int = 5,
        fixed_methods: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize homogeneous population.

        Args:
            source_population: Population to clone from (uses best agent)
            n_agents: Number of cloned agents
            fixed_methods: If provided, all agents use these methods
            seed: Random seed
        """
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        self.name = "Homogeneous"

        if fixed_methods:
            self.fixed_methods = fixed_methods
            self.agents = None
        elif source_population:
            # Clone best agent from source population
            config = PopulationConfig(
                n_agents=n_agents,
                seed=seed,
            )
            self.population = Population(config)

            # Get best agent's beliefs
            best_id, best_agent = source_population.get_best_agent()

            # Copy to all agents
            for agent in self.population.agents.values():
                agent.copy_beliefs_from(best_agent, tau=1.0)

            self.agents = self.population.agents
            self.fixed_methods = None
        else:
            # Create uniform agents (no learning yet)
            config = PopulationConfig(n_agents=n_agents, seed=seed)
            self.population = Population(config)
            self.agents = self.population.agents
            self.fixed_methods = None

    def select(self) -> Dict[str, List[str]]:
        """Select methods (all agents select same due to homogeneity)."""
        if self.fixed_methods:
            return {f"agent_{i}": self.fixed_methods for i in range(self.n_agents)}

        # All agents should select similarly (but may have small differences)
        selections = {}
        for agent_id, agent in self.agents.items():
            result = agent.select()
            selections[agent_id] = result.methods
        return selections

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series,
        reward_fn,
        window_size: int = 20,
    ) -> Dict:
        """
        Run homogeneous population through episode.

        Note: Since all agents are identical, performance should be
        similar across agents (unlike diverse population).
        """
        total_rewards = []

        for i in range(window_size, len(prices)):
            # Select (all agents select similarly)
            selections = self.select()

            # Compute rewards
            rewards = {}
            for agent_id, methods in selections.items():
                price_window = prices.iloc[i-window_size:i+1]
                reward = reward_fn(methods, price_window)
                rewards[agent_id] = reward

            # Track best reward
            total_rewards.append(max(rewards.values()))

            # Update agents (if learning enabled)
            if self.agents and not self.fixed_methods:
                for agent_id, agent in self.agents.items():
                    agent.update(selections[agent_id], rewards[agent_id])

        return {
            "total_reward": sum(total_rewards),
            "mean_reward": np.mean(total_rewards) if total_rewards else 0.0,
            "std_reward": np.std(total_rewards) if total_rewards else 0.0,
            "sharpe": np.mean(total_rewards) / (np.std(total_rewards) + 1e-8) if total_rewards else 0.0,
            "n_steps": len(total_rewards),
        }

    @classmethod
    def from_trained_population(
        cls,
        population: Population,
        n_agents: int = 5,
    ) -> "HomogeneousPopulation":
        """Create homogeneous population by cloning best from trained population."""
        return cls(source_population=population, n_agents=n_agents)
