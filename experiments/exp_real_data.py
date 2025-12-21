"""
Experiment: Real Data Validation

Tests whether specialization emerges and provides value on real
cryptocurrency market data (Bybit BTC 4-hour bars).

This addresses reviewer concern #2: "Synthetic Environment Only"

Key questions:
1. Does specialization emerge with HMM-detected regimes?
2. Does diverse population outperform baselines on real data?
3. Do results transfer from synthetic to real markets?
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from src.environment.real_data_loader import (
    load_bybit_data,
    prepare_real_data_experiment,
    get_regime_statistics,
)
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2


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


def compute_regime_si(niche_affinities):
    """Compute specialization index from regime affinities."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


class RandomBaseline:
    """Random method selection."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())

    def select(self, regime):
        return self.rng.choice(self.methods)


class HomogeneousBaseline:
    """Always use one method."""
    def __init__(self, method_name="VolScalp"):
        self.method = method_name

    def select(self, regime):
        return self.method


def run_real_data_experiment(
    symbol: str = "BTC",
    train_ratio: float = 0.7,
    n_trials: int = 10,
    output_dir: str = "results/exp_real_data",
):
    """
    Run experiment on real Bybit data.

    Args:
        symbol: Cryptocurrency symbol
        train_ratio: Fraction for training
        n_trials: Number of trials with different seeds
        output_dir: Where to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"REAL DATA EXPERIMENT: {symbol}")
    print("=" * 60)

    # Load and prepare data
    print("\nLoading data and detecting regimes...")
    train_prices, train_regimes, test_prices, test_regimes, detector = \
        prepare_real_data_experiment(symbol=symbol, train_ratio=train_ratio)

    print(f"Train: {len(train_prices)} bars, Test: {len(test_prices)} bars")

    # Regime statistics
    train_stats = get_regime_statistics(train_prices, train_regimes)
    print("\nTrain Regime Distribution:")
    print(train_stats.to_string(index=False))

    test_stats = get_regime_statistics(test_prices, test_regimes)
    print("\nTest Regime Distribution:")
    print(test_stats.to_string(index=False))

    # Run multiple trials
    results = {
        "diverse_train": [],
        "diverse_test": [],
        "homogeneous_train": [],
        "homogeneous_test": [],
        "random_train": [],
        "random_test": [],
        "si_train": [],
        "si_test": [],
        "diversity": [],
    }

    print(f"\nRunning {n_trials} trials...")

    for trial in tqdm(range(n_trials), desc="Trials"):
        # Initialize strategies
        diverse_pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial)
        homogeneous = HomogeneousBaseline(method_name="VolScalp")
        random_baseline = RandomBaseline(seed=trial)

        # === TRAINING PHASE ===
        diverse_train_reward = 0.0
        homo_train_reward = 0.0
        random_train_reward = 0.0

        window_size = 20
        for i in range(window_size, len(train_prices) - 1):
            price_window = train_prices.iloc[i-window_size:i+1]
            regime = train_regimes.iloc[i]

            if regime == "unknown":
                continue

            # Diverse population
            result = diverse_pop.run_iteration(price_window, regime, compute_reward)
            diverse_train_reward += compute_reward([result["winner_method"]], price_window)

            # Homogeneous
            homo_method = homogeneous.select(regime)
            homo_train_reward += compute_reward([homo_method], price_window)

            # Random
            rand_method = random_baseline.select(regime)
            random_train_reward += compute_reward([rand_method], price_window)

        # Compute train SI
        niche_dist = diverse_pop.get_niche_distribution()
        train_si = np.mean([compute_regime_si(aff) for aff in niche_dist.values()])

        # Compute diversity
        primary_niches = [max(aff, key=aff.get) for aff in niche_dist.values()]
        diversity = len(set(primary_niches)) / 4

        results["diverse_train"].append(diverse_train_reward)
        results["homogeneous_train"].append(homo_train_reward)
        results["random_train"].append(random_train_reward)
        results["si_train"].append(train_si)
        results["diversity"].append(diversity)

        # === TEST PHASE (no learning, just inference) ===
        diverse_test_reward = 0.0
        homo_test_reward = 0.0
        random_test_reward = 0.0

        # Reset random baseline for fair comparison
        random_baseline = RandomBaseline(seed=trial + 1000)

        for i in range(window_size, len(test_prices) - 1):
            price_window = test_prices.iloc[i-window_size:i+1]
            regime = test_regimes.iloc[i]

            if regime == "unknown":
                continue

            # Diverse population (inference only - no update)
            # Select methods based on learned preferences
            best_agent_id = max(
                diverse_pop.agents.keys(),
                key=lambda a: diverse_pop.agents[a].niche_affinity.get(regime, 0)
            )
            best_agent = diverse_pop.agents[best_agent_id]
            method = best_agent.select_method(regime)
            diverse_test_reward += compute_reward([method], price_window)

            # Homogeneous
            homo_method = homogeneous.select(regime)
            homo_test_reward += compute_reward([homo_method], price_window)

            # Random
            rand_method = random_baseline.select(regime)
            random_test_reward += compute_reward([rand_method], price_window)

        # Compute test SI (should be similar to train if stable)
        test_si = train_si  # Same model

        results["diverse_test"].append(diverse_test_reward)
        results["homogeneous_test"].append(homo_test_reward)
        results["random_test"].append(random_test_reward)
        results["si_test"].append(test_si)

    # Compute summary statistics
    summary = {
        "experiment": "real_data",
        "symbol": symbol,
        "n_trials": n_trials,
        "train_bars": len(train_prices),
        "test_bars": len(test_prices),

        # Training results
        "diverse_train_mean": float(np.mean(results["diverse_train"])),
        "diverse_train_std": float(np.std(results["diverse_train"])),
        "homogeneous_train_mean": float(np.mean(results["homogeneous_train"])),
        "random_train_mean": float(np.mean(results["random_train"])),

        # Test results
        "diverse_test_mean": float(np.mean(results["diverse_test"])),
        "diverse_test_std": float(np.std(results["diverse_test"])),
        "homogeneous_test_mean": float(np.mean(results["homogeneous_test"])),
        "random_test_mean": float(np.mean(results["random_test"])),

        # Specialization metrics
        "si_mean": float(np.mean(results["si_train"])),
        "si_std": float(np.std(results["si_train"])),
        "diversity_mean": float(np.mean(results["diversity"])),

        # Statistical tests
        "train_vs_homo_p": float(stats.ttest_rel(
            results["diverse_train"], results["homogeneous_train"]
        )[1]),
        "test_vs_homo_p": float(stats.ttest_rel(
            results["diverse_test"], results["homogeneous_test"]
        )[1]),
    }

    # Save results
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_path / "details.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("REAL DATA EXPERIMENT RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} | {'Train':>12} | {'Test':>12}")
    print("-" * 55)
    print(f"{'Diverse Population':<25} | {summary['diverse_train_mean']:>8.1f}±{summary['diverse_train_std']:.1f} | {summary['diverse_test_mean']:>8.1f}±{summary['diverse_test_std']:.1f}")
    print(f"{'Homogeneous (VolScalp)':<25} | {summary['homogeneous_train_mean']:>12.1f} | {summary['homogeneous_test_mean']:>12.1f}")
    print(f"{'Random':<25} | {summary['random_train_mean']:>12.1f} | {summary['random_test_mean']:>12.1f}")

    print(f"\n{'Specialization Metrics':}")
    print(f"  SI (Train): {summary['si_mean']:.3f} ± {summary['si_std']:.3f}")
    print(f"  Diversity: {summary['diversity_mean']:.2f}")

    # Key insights
    print("\nKey Insights:")
    train_improvement = (summary['diverse_train_mean'] - summary['homogeneous_train_mean']) / abs(summary['homogeneous_train_mean'] + 1e-8) * 100
    test_improvement = (summary['diverse_test_mean'] - summary['homogeneous_test_mean']) / abs(summary['homogeneous_test_mean'] + 1e-8) * 100

    if summary['si_mean'] > 0.5:
        print(f"  ✅ Specialization emerges on real data (SI = {summary['si_mean']:.3f})")
    else:
        print(f"  ⚠️ Limited specialization on real data (SI = {summary['si_mean']:.3f})")

    if train_improvement > 0:
        print(f"  ✅ Diverse beats Homogeneous on train by {train_improvement:.1f}%")
    else:
        print(f"  ⚠️ Homogeneous beats Diverse on train by {-train_improvement:.1f}%")

    if test_improvement > 0:
        print(f"  ✅ Diverse beats Homogeneous on test by {test_improvement:.1f}%")
    else:
        print(f"  ⚠️ Homogeneous beats Diverse on test by {-test_improvement:.1f}%")

    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_real_data_experiment(symbol="BTC", n_trials=10)
