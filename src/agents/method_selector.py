"""
Method Selector Agent using Thompson Sampling.

Each agent maintains beliefs about method effectiveness and uses
Thompson Sampling for exploration-exploitation balance.

Theoretical References:
- Thompson, W.R. (1933). "On the Likelihood that One Unknown Probability
  Exceeds Another." Biometrika, 25(3-4), 285-294.
- Russo, D., et al. (2018). "A Tutorial on Thompson Sampling."
  Foundations and Trends in Machine Learning, 11(1), 1-96.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from .inventory import METHOD_INVENTORY, get_method_names


@dataclass
class SelectionResult:
    """Result of method selection."""
    methods: List[str]  # Selected methods
    confidence: float   # Selection confidence (0-1)
    exploration: bool   # Whether this was an exploration action


@dataclass
class MethodBelief:
    """
    Bayesian belief about method performance.

    We use a Beta distribution (conjugate prior for Bernoulli outcomes)
    to model P(method is good | observations).

    Beta(α, β) where:
    - α = successes + 1 (prior)
    - β = failures + 1 (prior)
    """
    successes: float = 1.0  # α - 1
    failures: float = 1.0   # β - 1

    @property
    def alpha(self) -> float:
        return self.successes + 1

    @property
    def beta(self) -> float:
        return self.failures + 1

    @property
    def mean(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def sample(self, rng: np.random.Generator) -> float:
        """Sample from the Beta distribution (Thompson Sampling)."""
        return rng.beta(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        """
        Update beliefs based on observed reward.

        For continuous rewards in [0, 1], we use fractional updates.
        For binary outcomes, reward should be 0 or 1.
        """
        # Clip reward to [0, 1]
        reward = np.clip(reward, 0, 1)
        self.successes += reward
        self.failures += (1 - reward)

    def decay(self, factor: float = 0.99) -> None:
        """
        Apply forgetting factor to emphasize recent observations.

        This prevents the agent from being locked into historical beliefs.
        """
        self.successes *= factor
        self.failures *= factor

        # Maintain minimum uncertainty
        self.successes = max(self.successes, 0.5)
        self.failures = max(self.failures, 0.5)


class MethodSelector:
    """
    Agent that selects methods from inventory using Thompson Sampling.

    Key features:
    1. Maintains Beta-distributed beliefs about each method
    2. Uses Thompson Sampling for exploration-exploitation balance
    3. Supports context-aware selection (can condition on market state)
    4. Tracks selection history for specialization analysis

    This is the core unit of learning in our population-based system.
    """

    def __init__(
        self,
        agent_id: str,
        max_methods: int = 3,
        forgetting_factor: float = 0.995,
        min_exploration_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize the method selector.

        Args:
            agent_id: Unique identifier for this agent
            max_methods: Maximum methods to select per decision
            forgetting_factor: How much to decay old observations
            min_exploration_rate: Minimum probability of random exploration
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.max_methods = max_methods
        self.forgetting_factor = forgetting_factor
        self.min_exploration_rate = min_exploration_rate

        self.rng = np.random.default_rng(seed)

        # Initialize beliefs for each method
        self.method_names = get_method_names()
        self.beliefs: Dict[str, MethodBelief] = {
            name: MethodBelief() for name in self.method_names
        }

        # Track selection history for specialization analysis
        self.selection_history: List[List[str]] = []
        self.reward_history: List[float] = []

        # Cumulative statistics
        self.total_selections: Dict[str, int] = defaultdict(int)
        self.total_rewards: Dict[str, float] = defaultdict(float)

    def select(
        self,
        context: Optional[Dict] = None,
        force_exploration: bool = False,
    ) -> SelectionResult:
        """
        Select methods using Thompson Sampling.

        Args:
            context: Optional market context (for future extension)
            force_exploration: Force random exploration

        Returns:
            SelectionResult with selected methods
        """
        exploration = force_exploration or (self.rng.random() < self.min_exploration_rate)

        if exploration:
            # Random exploration: select random methods
            n_select = self.rng.integers(1, self.max_methods + 1)
            methods = list(self.rng.choice(
                self.method_names,
                size=min(n_select, len(self.method_names)),
                replace=False,
            ))
            confidence = 0.3  # Low confidence for exploration
        else:
            # Thompson Sampling: sample from each belief, select top-k
            samples = {
                name: belief.sample(self.rng)
                for name, belief in self.beliefs.items()
            }

            # Sort by sampled value (descending)
            sorted_methods = sorted(
                samples.keys(),
                key=lambda x: samples[x],
                reverse=True,
            )

            # Select top-k methods
            methods = sorted_methods[:self.max_methods]

            # Confidence based on sample values
            top_values = [samples[m] for m in methods]
            confidence = float(np.mean(top_values))

        # Record selection
        self.selection_history.append(methods)
        for method in methods:
            self.total_selections[method] += 1

        return SelectionResult(
            methods=methods,
            confidence=confidence,
            exploration=exploration,
        )

    def update(self, methods: List[str], reward: float) -> None:
        """
        Update beliefs based on observed reward.

        Args:
            methods: Methods that were used
            reward: Observed reward (normalized to [0, 1])
        """
        # Normalize reward to [0, 1] if needed
        # Assume reward is already in reasonable range
        normalized_reward = (reward + 1) / 2  # Map [-1, 1] to [0, 1]
        normalized_reward = np.clip(normalized_reward, 0, 1)

        # Update beliefs for selected methods
        for method in methods:
            if method in self.beliefs:
                self.beliefs[method].update(normalized_reward)
                self.total_rewards[method] += reward

        # Apply forgetting factor to all beliefs
        for belief in self.beliefs.values():
            belief.decay(self.forgetting_factor)

        # Record reward
        self.reward_history.append(reward)

    def get_preferences(self) -> Dict[str, float]:
        """
        Get current preference scores for each method.

        Returns:
            Dict mapping method name to preference score (mean of belief)
        """
        return {
            name: belief.mean
            for name, belief in self.beliefs.items()
        }

    def get_method_usage_distribution(self) -> Dict[str, float]:
        """
        Get normalized distribution of method usage.

        This is used to compute specialization index.
        """
        total = sum(self.total_selections.values())
        if total == 0:
            # Uniform if no selections yet
            n = len(self.method_names)
            return {name: 1.0 / n for name in self.method_names}

        return {
            name: self.total_selections[name] / total
            for name in self.method_names
        }

    def get_dominant_method(self) -> Optional[str]:
        """Get the most frequently selected method."""
        if not self.total_selections:
            return None
        return max(self.total_selections, key=self.total_selections.get)

    def copy_beliefs_from(self, other: "MethodSelector", tau: float = 0.1) -> None:
        """
        Copy beliefs from another agent (knowledge transfer).

        Args:
            other: Source agent to copy from
            tau: Interpolation factor (0 = keep own, 1 = full copy)
        """
        for name in self.method_names:
            own = self.beliefs[name]
            src = other.beliefs[name]

            # Interpolate alpha and beta
            own.successes = (1 - tau) * own.successes + tau * src.successes
            own.failures = (1 - tau) * own.failures + tau * src.failures

    def reset_history(self) -> None:
        """Reset selection history (keep beliefs)."""
        self.selection_history = []
        self.reward_history = []
        self.total_selections = defaultdict(int)
        self.total_rewards = defaultdict(float)

    def __repr__(self) -> str:
        dominant = self.get_dominant_method()
        return f"MethodSelector(id={self.agent_id}, dominant={dominant})"


def create_population_of_selectors(
    n_agents: int,
    base_seed: int = 0,
    **kwargs,
) -> List[MethodSelector]:
    """Factory function to create a population of agents."""
    return [
        MethodSelector(
            agent_id=f"agent_{i}",
            seed=base_seed + i,
            **kwargs,
        )
        for i in range(n_agents)
    ]
