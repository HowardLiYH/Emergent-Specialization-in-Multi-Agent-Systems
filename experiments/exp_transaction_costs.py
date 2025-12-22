"""
Experiment: Transaction Cost Analysis

Tests whether specialization provides advantages when transaction costs
are considered. Hypothesis: Specialists switch methods less frequently,
so they benefit more from lower transaction costs.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
from collections import defaultdict

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.baselines.oracle import OracleSpecialist
from src.analysis.statistical_utils import compute_stats


def compute_reward(methods, prices):
    """Compute reward for selected methods (without transaction costs)."""
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


class HomogeneousBaseline:
    """Baseline that always uses one method."""
    def __init__(self, method: str = "VolScalp"):
        self.method = method
        self.last_method = method

    def select(self, regime: str) -> str:
        return self.method

    def get_switch_count(self) -> int:
        return 0  # Never switches


class RandomBaseline:
    """Baseline that randomly selects methods."""
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.methods = list(METHOD_INVENTORY_V2.keys())
        self.last_method = None
        self.switches = 0

    def select(self, regime: str) -> str:
        method = self.rng.choice(self.methods)
        if self.last_method is not None and method != self.last_method:
            self.switches += 1
        self.last_method = method
        return method

    def get_switch_count(self) -> int:
        return self.switches


def run_transaction_costs_experiment(
    fee_rates: list = None,
    n_trials: int = 30,
    n_iterations: int = 2000,
    output_dir: str = "results/exp_transaction_costs",
):
    """
    Analyze impact of transaction costs on different strategies.
    """
    if fee_rates is None:
        fee_rates = [0.0, 0.001, 0.002, 0.005, 0.01]  # 0%, 0.1%, 0.2%, 0.5%, 1%

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRANSACTION COST EXPERIMENT")
    print(f"Fee rates: {fee_rates}")
    print(f"Trials: {n_trials}, Iterations: {n_iterations}")
    print("=" * 60)

    results = {
        fee: {
            "diverse": [],
            "homogeneous": [],
            "random": [],
            "oracle": [],
            "diverse_switches": [],
            "random_switches": [],
        }
        for fee in fee_rates
    }

    for trial in tqdm(range(n_trials), desc="Trials"):
        # Generate data
        config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
        market = SyntheticMarketEnvironment(config)
        all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=trial)

        # Initialize strategies
        diverse_pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial)
        homogeneous = HomogeneousBaseline(method="VolScalp")
        random_baseline = RandomBaseline(seed=trial)
        oracle = OracleSpecialist()

        # Track rewards and switches
        diverse_rewards = []
        homo_rewards = []
        random_rewards = []
        oracle_rewards = []

        diverse_last_method = None
        diverse_switches = 0

        window_size = 20

        for i in range(n_iterations):
            start_idx = i
            end_idx = i + window_size + 1
            prices = all_prices.iloc[start_idx:end_idx].copy()
            regime = all_regimes.iloc[end_idx - 1]

            # Diverse population
            result = diverse_pop.run_iteration(prices, regime, compute_reward)
            diverse_method = result["winner_method"]
            diverse_reward = compute_reward([diverse_method], prices)
            if diverse_last_method is not None and diverse_method != diverse_last_method:
                diverse_switches += 1
            diverse_last_method = diverse_method
            diverse_rewards.append(diverse_reward)

            # Homogeneous
            homo_method = homogeneous.select(regime)
            homo_reward = compute_reward([homo_method], prices)
            homo_rewards.append(homo_reward)

            # Random
            rand_method = random_baseline.select(regime)
            rand_reward = compute_reward([rand_method], prices)
            random_rewards.append(rand_reward)

            # Oracle
            oracle_methods = oracle.select(regime)
            oracle_reward = compute_reward(oracle_methods, prices)
            oracle_rewards.append(oracle_reward)

        # Calculate net rewards for each fee rate
        for fee in fee_rates:
            # Diverse: subtract fee for each switch
            diverse_net = sum(diverse_rewards) - fee * diverse_switches

            # Homogeneous: no switches
            homo_net = sum(homo_rewards)

            # Random: many switches
            random_net = sum(random_rewards) - fee * random_baseline.switches

            # Oracle: assume one switch per regime change
            # Estimate regime changes from data
            regime_changes = sum(1 for i in range(1, len(all_regimes))
                               if all_regimes.iloc[i] != all_regimes.iloc[i-1])
            oracle_net = sum(oracle_rewards) - fee * regime_changes

            results[fee]["diverse"].append(diverse_net)
            results[fee]["homogeneous"].append(homo_net)
            results[fee]["random"].append(random_net)
            results[fee]["oracle"].append(oracle_net)

        # Track switches (only need to record once, fee-independent)
        results[fee_rates[0]]["diverse_switches"].append(diverse_switches)
        results[fee_rates[0]]["random_switches"].append(random_baseline.switches)

    # Compute statistics
    print("\n" + "=" * 80)
    print(f"{'Fee':>6} | {'Diverse':>15} | {'Homogeneous':>15} | {'Random':>15} | {'Oracle':>15}")
    print("-" * 80)

    summary = {"fee_rates": fee_rates}

    for fee in fee_rates:
        diverse_stat = compute_stats(results[fee]["diverse"])
        homo_stat = compute_stats(results[fee]["homogeneous"])
        random_stat = compute_stats(results[fee]["random"])
        oracle_stat = compute_stats(results[fee]["oracle"])

        print(f"{fee*100:>5.1f}% | {diverse_stat.mean:>7.1f}±{diverse_stat.std:.1f} | "
              f"{homo_stat.mean:>7.1f}±{homo_stat.std:.1f} | "
              f"{random_stat.mean:>7.1f}±{random_stat.std:.1f} | "
              f"{oracle_stat.mean:>7.1f}±{oracle_stat.std:.1f}")

        summary[f"fee_{fee}_diverse"] = diverse_stat.mean
        summary[f"fee_{fee}_homogeneous"] = homo_stat.mean
        summary[f"fee_{fee}_random"] = random_stat.mean
        summary[f"fee_{fee}_oracle"] = oracle_stat.mean

    # Switch statistics
    print("\n--- Method Switching Statistics ---")
    diverse_switches = results[fee_rates[0]]["diverse_switches"]
    random_switches = results[fee_rates[0]]["random_switches"]

    print(f"Diverse population switches: {np.mean(diverse_switches):.1f} ± {np.std(diverse_switches):.1f}")
    print(f"Random baseline switches: {np.mean(random_switches):.1f} ± {np.std(random_switches):.1f}")

    summary["diverse_switches_mean"] = float(np.mean(diverse_switches))
    summary["random_switches_mean"] = float(np.mean(random_switches))

    # Key insights
    print("\n--- Key Insights ---")

    # At what fee rate does diverse beat homogeneous?
    for fee in fee_rates:
        diverse_mean = np.mean(results[fee]["diverse"])
        homo_mean = np.mean(results[fee]["homogeneous"])

        if diverse_mean > homo_mean:
            print(f"✅ Diverse beats Homogeneous at fee={fee*100:.1f}%")
        else:
            print(f"⚠️ Homogeneous beats Diverse at fee={fee*100:.1f}%")

    # Save results
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_transaction_costs_experiment(n_trials=30, n_iterations=2000)
