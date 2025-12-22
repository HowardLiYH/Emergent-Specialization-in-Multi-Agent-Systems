"""
Experiment: Adaptive Lambda Mechanism

Tests whether adaptive lambda scheduling can achieve both
high specialization (SI > 0.7) AND high reward.

Strategy: Start with high lambda to encourage niche formation,
then decay to allow exploitation and exploration.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.analysis.statistical_utils import compute_stats, compare_groups


# Import base NicheAgent components
from src.agents.niche_population import NicheAgent


def compute_reward(methods, prices):
    """Compute reward for selected methods."""
    if len(prices) < 2:
        return 0.0
    signals, confs = [], []
    for m in methods:
        if m in METHOD_INVENTORY_V2:
            result = METHOD_INVENTORY_V2[m].execute(prices)
            signals.append(result['signal'])
            confs.append(result['confidence'])
    if not signals:
        return 0.0
    weights = np.array(confs) / (sum(confs) + 1e-8)
    signal = sum(s * w for s, w in zip(signals, weights))
    price_return = (prices['close'].iloc[-1] / prices['close'].iloc[-2]) - 1
    return float(np.clip(signal * price_return * 10, -1, 1))


def compute_si(niche_affinities):
    """Compute specialization index from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


class AdaptiveLambdaPopulation:
    """Population with adaptive lambda scheduling."""

    def __init__(
        self,
        n_agents: int = 8,
        lambda_max: float = 0.5,
        lambda_min: float = 0.1,
        warmup_fraction: float = 0.2,
        max_iterations: int = 2000,
        schedule: str = "linear",  # "linear", "cosine", "step"
        seed: Optional[int] = None,
    ):
        self.n_agents = n_agents
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.warmup_fraction = warmup_fraction
        self.max_iterations = max_iterations
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)

        self.regimes = ["trend_up", "trend_down", "mean_revert", "volatile"]

        # Create agents
        self.agents: Dict[str, NicheAgent] = {}
        for i in range(n_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = NicheAgent(
                agent_id=agent_id,
                regimes=self.regimes,
                seed=(seed + i * 100) if seed else None,
            )

        self.iteration = 0
        self.history = []
        self.lambda_history = []

    def get_lambda(self) -> float:
        """Get current lambda based on schedule."""
        warmup_iters = int(self.warmup_fraction * self.max_iterations)

        if self.iteration < warmup_iters:
            # Warmup phase: keep high lambda
            return self.lambda_max

        # Decay phase
        progress = (self.iteration - warmup_iters) / (self.max_iterations - warmup_iters)
        progress = min(1.0, progress)

        if self.schedule == "linear":
            return self.lambda_max - (self.lambda_max - self.lambda_min) * progress

        elif self.schedule == "cosine":
            return self.lambda_min + (self.lambda_max - self.lambda_min) * \
                   (1 + np.cos(np.pi * progress)) / 2

        elif self.schedule == "step":
            # Step decay at 50% and 75%
            if progress < 0.5:
                return self.lambda_max
            elif progress < 0.75:
                return (self.lambda_max + self.lambda_min) / 2
            else:
                return self.lambda_min

        return self.lambda_max

    def run_iteration(self, prices, regime: str, reward_fn):
        """Run one iteration with adaptive lambda."""
        self.iteration += 1
        current_lambda = self.get_lambda()
        self.lambda_history.append(current_lambda)

        # Each agent selects a method
        selections: Dict[str, str] = {}
        raw_rewards: Dict[str, float] = {}

        for agent_id, agent in self.agents.items():
            method = agent.select_method(regime)
            selections[agent_id] = method
            reward = reward_fn([method], prices)
            raw_rewards[agent_id] = reward

        # Apply niche bonus with current lambda
        adjusted_rewards: Dict[str, float] = {}
        for agent_id, agent in self.agents.items():
            raw = raw_rewards[agent_id]

            agent_niche = agent.get_primary_niche()
            if agent_niche == regime:
                bonus = current_lambda * agent.niche_affinity[regime]
            else:
                bonus = -current_lambda * 0.3 * (1 - agent.niche_affinity[regime])

            adjusted_rewards[agent_id] = raw + bonus

        # Determine winner
        winner_id = max(adjusted_rewards, key=adjusted_rewards.get)

        # Update agents
        for agent_id, agent in self.agents.items():
            won = (agent_id == winner_id)
            agent.update(regime, selections[agent_id], won=won)

        self.history.append({
            "iteration": self.iteration,
            "lambda": current_lambda,
            "regime": regime,
            "winner": winner_id,
        })

        return {
            "winner_id": winner_id,
            "winner_method": selections[winner_id],
            "lambda": current_lambda,
        }

    def get_niche_distribution(self) -> Dict[str, Dict[str, float]]:
        """Get niche affinity for all agents."""
        return {
            agent_id: agent.niche_affinity.copy()
            for agent_id, agent in self.agents.items()
        }


def run_adaptive_lambda_experiment(
    n_trials: int = 30,
    n_iterations: int = 2000,
    output_dir: str = "results/exp_adaptive_lambda",
):
    """
    Compare fixed vs adaptive lambda.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ADAPTIVE LAMBDA EXPERIMENT")
    print("=" * 60)

    # Configurations to test
    configs = {
        "fixed_0.0": {"type": "fixed", "lambda": 0.0},
        "fixed_0.25": {"type": "fixed", "lambda": 0.25},
        "fixed_0.5": {"type": "fixed", "lambda": 0.5},
        "adaptive_linear": {"type": "adaptive", "schedule": "linear"},
        "adaptive_cosine": {"type": "adaptive", "schedule": "cosine"},
        "adaptive_step": {"type": "adaptive", "schedule": "step"},
    }

    results = {name: {"si": [], "reward": []} for name in configs}

    for trial in tqdm(range(n_trials), desc="Trials"):
        # Generate data
        config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
        market = SyntheticMarketEnvironment(config)
        all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=trial)

        for name, cfg in configs.items():
            if cfg["type"] == "fixed":
                from src.agents.niche_population import NichePopulation
                pop = NichePopulation(n_agents=8, niche_bonus=cfg["lambda"], seed=trial)
            else:
                pop = AdaptiveLambdaPopulation(
                    n_agents=8,
                    schedule=cfg["schedule"],
                    max_iterations=n_iterations,
                    seed=trial,
                )

            total_reward = 0
            window_size = 20

            for i in range(n_iterations):
                start_idx = i
                end_idx = i + window_size + 1
                prices = all_prices.iloc[start_idx:end_idx].copy()
                regime = all_regimes.iloc[end_idx - 1]

                result = pop.run_iteration(prices, regime, compute_reward)
                reward = compute_reward([result["winner_method"]], prices)
                total_reward += reward

            # Compute SI
            niche_dist = pop.get_niche_distribution()
            si_values = [compute_si(aff) for aff in niche_dist.values()]
            avg_si = np.mean(si_values)

            results[name]["si"].append(avg_si)
            results[name]["reward"].append(total_reward)

    # Compute stats
    stats = {}
    for name in configs:
        stats[name] = {
            "si": compute_stats(results[name]["si"]),
            "reward": compute_stats(results[name]["reward"]),
        }

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Configuration':<20} | {'SI':>15} | {'Reward':>20}")
    print("-" * 70)

    for name in configs:
        si_stat = stats[name]["si"]
        reward_stat = stats[name]["reward"]
        print(f"{name:<20} | {si_stat.mean:.3f}±{si_stat.std:.3f} | "
              f"{reward_stat.mean:>8.1f}±{reward_stat.std:.1f}")

    # Find best configuration
    print("\n--- Best Configurations ---")

    # Best reward with SI > 0.7
    best_reward = -np.inf
    best_config = None

    for name in configs:
        si = stats[name]["si"].mean
        reward = stats[name]["reward"].mean

        if si >= 0.7 and reward > best_reward:
            best_reward = reward
            best_config = name

    if best_config:
        print(f"Best (SI >= 0.7): {best_config}")
        print(f"  SI: {stats[best_config]['si'].mean:.3f}")
        print(f"  Reward: {stats[best_config]['reward'].mean:.1f}")

    # Compare adaptive vs best fixed
    print("\n--- Adaptive vs Fixed Comparison ---")

    adaptive_rewards = results["adaptive_linear"]["reward"]
    fixed_best = max(
        [("fixed_0.0", results["fixed_0.0"]["reward"]),
         ("fixed_0.25", results["fixed_0.25"]["reward"]),
         ("fixed_0.5", results["fixed_0.5"]["reward"])],
        key=lambda x: np.mean(x[1])
    )

    comparison = compare_groups(adaptive_rewards, fixed_best[1], paired=True)
    print(f"Adaptive Linear vs {fixed_best[0]}:")
    print(f"  Difference: {comparison.difference:+.1f}")
    print(f"  p-value: {comparison.p_value:.4e}")
    print(f"  Cohen's d: {comparison.cohens_d:.2f}")

    # Save results
    summary = {
        "experiment": "adaptive_lambda",
        "n_trials": n_trials,
        "n_iterations": n_iterations,
        "best_config": best_config,
    }

    for name in configs:
        summary[f"{name}_si_mean"] = stats[name]["si"].mean
        summary[f"{name}_si_std"] = stats[name]["si"].std
        summary[f"{name}_reward_mean"] = stats[name]["reward"].mean
        summary[f"{name}_reward_std"] = stats[name]["reward"].std

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_adaptive_lambda_experiment(n_trials=30, n_iterations=2000)
