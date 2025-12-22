"""
Experiment: Specialization-Performance Trade-off Analysis

Analyzes the relationship between lambda (niche bonus) and performance.
Identifies optimal operating point on the Pareto frontier.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.analysis.statistical_utils import compute_stats


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


def run_tradeoff_analysis(
    lambda_values: list = None,
    n_trials: int = 30,
    n_iterations: int = 2000,
    output_dir: str = "results/exp_tradeoff",
):
    """
    Fine-grained lambda sweep to analyze SI vs Reward trade-off.
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SPECIALIZATION-PERFORMANCE TRADE-OFF ANALYSIS")
    print(f"Lambda values: {lambda_values}")
    print(f"Trials: {n_trials}, Iterations: {n_iterations}")
    print("=" * 60)

    results = {}

    for lam in lambda_values:
        print(f"\nTesting λ = {lam}")

        trial_results = {"si": [], "reward": [], "diversity": [], "exploration": []}

        for trial in tqdm(range(n_trials), desc=f"λ={lam}", leave=False):
            # Generate data
            config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
            market = SyntheticMarketEnvironment(config)
            all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=trial)

            # Create population with this lambda
            pop = NichePopulation(n_agents=8, niche_bonus=lam, seed=trial)

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

            # Compute metrics
            niche_dist = pop.get_niche_distribution()
            si_values = [compute_si(aff) for aff in niche_dist.values()]
            avg_si = np.mean(si_values)

            # Diversity: number of regimes covered by specialists
            primary_niches = [max(aff, key=aff.get) for aff in niche_dist.values()]
            diversity = len(set(primary_niches)) / 4

            # Average exploration rate
            avg_exploration = np.mean([
                pop.agents[aid].exploration_rate for aid in pop.agents
            ])

            trial_results["si"].append(avg_si)
            trial_results["reward"].append(total_reward)
            trial_results["diversity"].append(diversity)
            trial_results["exploration"].append(avg_exploration)

        # Compute stats
        results[lam] = {
            "si": compute_stats(trial_results["si"]),
            "reward": compute_stats(trial_results["reward"]),
            "diversity_mean": float(np.mean(trial_results["diversity"])),
            "exploration_mean": float(np.mean(trial_results["exploration"])),
        }

        print(f"  SI: {results[lam]['si'].mean:.3f}±{results[lam]['si'].std:.3f}")
        print(f"  Reward: {results[lam]['reward'].mean:.1f}±{results[lam]['reward'].std:.1f}")
        print(f"  Diversity: {results[lam]['diversity_mean']:.2f}")

    # Find Pareto optimal points
    print("\n--- Pareto Frontier Analysis ---")

    pareto_points = []
    for lam in lambda_values:
        si = results[lam]["si"].mean
        reward = results[lam]["reward"].mean

        # Check if dominated
        dominated = False
        for other_lam in lambda_values:
            other_si = results[other_lam]["si"].mean
            other_reward = results[other_lam]["reward"].mean

            # Dominated if other is better in both
            if other_si >= si and other_reward >= reward and (other_si > si or other_reward > reward):
                dominated = True
                break

        if not dominated:
            pareto_points.append(lam)

    print(f"Pareto optimal λ values: {pareto_points}")

    # Find optimal operating point (balance SI > 0.7 and max reward)
    print("\n--- Optimal Operating Point ---")

    best_lam = None
    best_reward = -np.inf

    for lam in lambda_values:
        si = results[lam]["si"].mean
        reward = results[lam]["reward"].mean

        if si >= 0.7 and reward > best_reward:
            best_reward = reward
            best_lam = lam

    if best_lam is not None:
        print(f"Optimal λ = {best_lam} (SI={results[best_lam]['si'].mean:.3f}, "
              f"Reward={results[best_lam]['reward'].mean:.1f})")
    else:
        print("No λ achieves SI >= 0.7")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'λ':>6} | {'SI':>12} | {'Reward':>15} | {'Diversity':>10} | {'Pareto':>8}")
    print("-" * 70)

    for lam in lambda_values:
        si_stat = results[lam]["si"]
        reward_stat = results[lam]["reward"]
        div = results[lam]["diversity_mean"]
        pareto = "✓" if lam in pareto_points else ""

        print(f"{lam:>6.2f} | {si_stat.mean:>5.3f}±{si_stat.std:.3f} | "
              f"{reward_stat.mean:>7.1f}±{reward_stat.std:.1f} | "
              f"{div:>10.2f} | {pareto:>8}")

    # Save results
    summary = {
        "experiment": "tradeoff_analysis",
        "n_trials": n_trials,
        "n_iterations": n_iterations,
        "lambda_values": lambda_values,
        "pareto_optimal": pareto_points,
        "optimal_lambda": best_lam,
    }

    for lam in lambda_values:
        summary[f"lambda_{lam}_si_mean"] = results[lam]["si"].mean
        summary[f"lambda_{lam}_si_std"] = results[lam]["si"].std
        summary[f"lambda_{lam}_reward_mean"] = results[lam]["reward"].mean
        summary[f"lambda_{lam}_reward_std"] = results[lam]["reward"].std
        summary[f"lambda_{lam}_diversity"] = results[lam]["diversity_mean"]

    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_tradeoff_analysis(n_trials=30, n_iterations=2000)
