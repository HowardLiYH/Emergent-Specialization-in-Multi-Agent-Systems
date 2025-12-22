"""
Experiment: Out-of-Sample Generalization Test

Tests whether learned specialization generalizes to unseen data.
This validates that the system is learning meaningful patterns,
not just overfitting to training data.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.analysis.statistical_utils import compute_stats, compare_groups


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
    """Compute specialization index."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


def run_out_of_sample_experiment(
    n_trials: int = 30,
    train_iterations: int = 1500,
    test_iterations: int = 500,
    output_dir: str = "results/exp_out_of_sample",
):
    """
    Train on first portion, test on remainder (frozen weights).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_iterations = train_iterations + test_iterations

    print("=" * 60)
    print("OUT-OF-SAMPLE GENERALIZATION TEST")
    print(f"Train: {train_iterations}, Test: {test_iterations}")
    print(f"Trials: {n_trials}")
    print("=" * 60)

    results = {
        "train_reward": [],
        "test_reward": [],
        "train_si": [],
        "test_si": [],
        "generalization_gap": [],
    }

    for trial in tqdm(range(n_trials), desc="Trials"):
        # Generate data (same seed for train and test)
        config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
        market = SyntheticMarketEnvironment(config)
        all_prices, all_regimes = market.generate(
            n_bars=total_iterations + 50,
            seed=trial
        )

        # Create population
        pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial)

        window_size = 20

        # === TRAINING PHASE ===
        train_rewards = []

        for i in range(train_iterations):
            start_idx = i
            end_idx = i + window_size + 1
            prices = all_prices.iloc[start_idx:end_idx].copy()
            regime = all_regimes.iloc[end_idx - 1]

            result = pop.run_iteration(prices, regime, compute_reward)
            reward = compute_reward([result["winner_method"]], prices)
            train_rewards.append(reward)

        # Compute train SI
        train_niche_dist = pop.get_niche_distribution()
        train_si = np.mean([compute_si(aff) for aff in train_niche_dist.values()])

        # Store agent states for test phase
        # We'll freeze learning by not calling update
        frozen_agents = {}
        for agent_id, agent in pop.agents.items():
            frozen_agents[agent_id] = {
                "beliefs": deepcopy(agent.beliefs),
                "niche_affinity": deepcopy(agent.niche_affinity),
            }

        # === TEST PHASE (frozen weights) ===
        test_rewards = []

        for i in range(train_iterations, total_iterations):
            start_idx = i
            end_idx = i + window_size + 1
            prices = all_prices.iloc[start_idx:end_idx].copy()
            regime = all_regimes.iloc[end_idx - 1]

            # Select best agent for this regime based on frozen affinities
            best_agent_id = max(
                frozen_agents.keys(),
                key=lambda a: frozen_agents[a]["niche_affinity"].get(regime, 0)
            )

            # Get method from frozen beliefs
            agent = pop.agents[best_agent_id]
            method = agent.select_method(regime)  # Uses current beliefs

            reward = compute_reward([method], prices)
            test_rewards.append(reward)

        # Compute test SI (should be same as train since frozen)
        test_si = train_si

        # Compute metrics
        train_total = sum(train_rewards)
        test_total = sum(test_rewards)

        # Normalize by number of iterations for fair comparison
        train_per_iter = train_total / train_iterations
        test_per_iter = test_total / test_iterations

        generalization_gap = (train_per_iter - test_per_iter) / (abs(train_per_iter) + 1e-8) * 100

        results["train_reward"].append(train_total)
        results["test_reward"].append(test_total)
        results["train_si"].append(train_si)
        results["test_si"].append(test_si)
        results["generalization_gap"].append(generalization_gap)

    # Compute statistics
    train_stats = compute_stats(results["train_reward"])
    test_stats = compute_stats(results["test_reward"])
    gap_stats = compute_stats(results["generalization_gap"])

    # Compare train vs test
    comparison = compare_groups(
        results["train_reward"],
        results["test_reward"],
        paired=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{'Phase':<10} | {'Reward':>20} | {'95% CI':>25}")
    print("-" * 60)
    print(f"{'Train':<10} | {train_stats.mean:>8.1f} ± {train_stats.std:.1f} | "
          f"[{train_stats.ci_lower:.1f}, {train_stats.ci_upper:.1f}]")
    print(f"{'Test':<10} | {test_stats.mean:>8.1f} ± {test_stats.std:.1f} | "
          f"[{test_stats.ci_lower:.1f}, {test_stats.ci_upper:.1f}]")

    print(f"\nGeneralization Gap: {gap_stats.mean:.1f}% ± {gap_stats.std:.1f}%")
    print(f"Train vs Test p-value: {comparison.p_value:.4e}")
    print(f"Cohen's d: {comparison.cohens_d:.2f}")

    # SI
    print(f"\nSpecialization Index: {np.mean(results['train_si']):.3f} ± {np.std(results['train_si']):.3f}")

    # Key insights
    print("\n--- Key Insights ---")

    if abs(gap_stats.mean) < 20:
        print("✅ Good generalization (gap < 20%)")
    elif abs(gap_stats.mean) < 50:
        print("⚠️ Moderate generalization gap (20-50%)")
    else:
        print("❌ Poor generalization (gap > 50%)")

    # Save results
    summary = {
        "experiment": "out_of_sample",
        "n_trials": n_trials,
        "train_iterations": train_iterations,
        "test_iterations": test_iterations,
        "train_reward_mean": train_stats.mean,
        "train_reward_std": train_stats.std,
        "test_reward_mean": test_stats.mean,
        "test_reward_std": test_stats.std,
        "generalization_gap_mean": gap_stats.mean,
        "generalization_gap_std": gap_stats.std,
        "train_test_p_value": comparison.p_value,
        "cohens_d": comparison.cohens_d,
        "si_mean": float(np.mean(results["train_si"])),
    }

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_out_of_sample_experiment(n_trials=30)
