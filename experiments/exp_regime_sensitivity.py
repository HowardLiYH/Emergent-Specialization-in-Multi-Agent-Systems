"""
Experiment: Regime Duration Sensitivity Analysis

Tests how regime duration affects specialization and performance.
Hypothesis: Short regimes favor generalists, long regimes favor specialists.
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
    """Compute specialization index."""
    affinities = np.array(list(niche_affinities.values()))
    affinities = affinities / (affinities.sum() + 1e-8)
    entropy = -np.sum(affinities * np.log(affinities + 1e-8))
    max_entropy = np.log(len(affinities))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0


class HomogeneousBaseline:
    """Always use one method."""
    def __init__(self, method: str = "VolScalp"):
        self.method = method

    def select(self, regime: str) -> str:
        return self.method


def run_regime_sensitivity_experiment(
    regime_durations: list = None,
    n_trials: int = 30,
    n_iterations: int = 2000,
    output_dir: str = "results/exp_regime_sensitivity",
):
    """
    Vary regime duration and measure impact on specialization and performance.
    """
    if regime_durations is None:
        regime_durations = [10, 20, 50, 100, 200, 500]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("REGIME DURATION SENSITIVITY ANALYSIS")
    print(f"Durations: {regime_durations}")
    print(f"Trials: {n_trials}, Iterations: {n_iterations}")
    print("=" * 60)

    results = {
        duration: {
            "diverse_si": [],
            "diverse_reward": [],
            "diverse_diversity": [],
            "homo_reward": [],
            "regime_changes": [],
            "specialist_win_rate": [],
        }
        for duration in regime_durations
    }

    for duration in regime_durations:
        print(f"\nTesting regime_duration={duration}")

        for trial in tqdm(range(n_trials), desc=f"dur={duration}", leave=False):
            # Generate data with specific regime duration
            config = SyntheticMarketConfig(
                regime_duration_mean=duration,
                regime_duration_std=duration // 5,  # 20% of mean
            )
            market = SyntheticMarketEnvironment(config)
            all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=trial)

            # Count regime changes
            regime_changes = sum(
                1 for i in range(1, len(all_regimes))
                if all_regimes.iloc[i] != all_regimes.iloc[i-1]
            )

            # Create strategies
            diverse_pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial)
            homogeneous = HomogeneousBaseline(method="VolScalp")

            window_size = 20
            diverse_rewards = []
            homo_rewards = []
            specialist_wins = 0
            total_wins = 0

            for i in range(n_iterations):
                start_idx = i
                end_idx = i + window_size + 1
                prices = all_prices.iloc[start_idx:end_idx].copy()
                regime = all_regimes.iloc[end_idx - 1]

                # Diverse
                result = diverse_pop.run_iteration(prices, regime, compute_reward)
                diverse_reward = compute_reward([result["winner_method"]], prices)
                diverse_rewards.append(diverse_reward)

                # Check if winner is a specialist for this regime
                winner_id = result["winner_id"]
                winner_niche = diverse_pop.agents[winner_id].get_primary_niche()
                if winner_niche == regime:
                    specialist_wins += 1
                total_wins += 1

                # Homogeneous
                homo_method = homogeneous.select(regime)
                homo_reward = compute_reward([homo_method], prices)
                homo_rewards.append(homo_reward)

            # Compute metrics
            niche_dist = diverse_pop.get_niche_distribution()
            si_values = [compute_si(aff) for aff in niche_dist.values()]
            avg_si = np.mean(si_values)

            primary_niches = [max(aff, key=aff.get) for aff in niche_dist.values()]
            diversity = len(set(primary_niches)) / 4

            specialist_win_rate = specialist_wins / total_wins if total_wins > 0 else 0

            results[duration]["diverse_si"].append(avg_si)
            results[duration]["diverse_reward"].append(sum(diverse_rewards))
            results[duration]["diverse_diversity"].append(diversity)
            results[duration]["homo_reward"].append(sum(homo_rewards))
            results[duration]["regime_changes"].append(regime_changes)
            results[duration]["specialist_win_rate"].append(specialist_win_rate)

    # Compute statistics and print
    print("\n" + "=" * 90)
    print(f"{'Duration':>8} | {'SI':>12} | {'Diverse':>15} | {'Homo':>15} | "
          f"{'Changes':>8} | {'Spec Win%':>10}")
    print("-" * 90)

    summary = {"regime_durations": regime_durations}

    for duration in regime_durations:
        si_stat = compute_stats(results[duration]["diverse_si"])
        diverse_stat = compute_stats(results[duration]["diverse_reward"])
        homo_stat = compute_stats(results[duration]["homo_reward"])
        changes = np.mean(results[duration]["regime_changes"])
        spec_win = np.mean(results[duration]["specialist_win_rate"])

        print(f"{duration:>8} | {si_stat.mean:.3f}±{si_stat.std:.3f} | "
              f"{diverse_stat.mean:>7.1f}±{diverse_stat.std:.1f} | "
              f"{homo_stat.mean:>7.1f}±{homo_stat.std:.1f} | "
              f"{changes:>8.1f} | {spec_win*100:>9.1f}%")

        summary[f"dur_{duration}_si_mean"] = si_stat.mean
        summary[f"dur_{duration}_si_std"] = si_stat.std
        summary[f"dur_{duration}_diverse_mean"] = diverse_stat.mean
        summary[f"dur_{duration}_homo_mean"] = homo_stat.mean
        summary[f"dur_{duration}_regime_changes"] = changes
        summary[f"dur_{duration}_specialist_win_rate"] = spec_win

    # Key insights
    print("\n--- Key Insights ---")

    # Find optimal duration for specialization
    best_si_duration = max(regime_durations,
                          key=lambda d: np.mean(results[d]["diverse_si"]))
    print(f"Best SI at duration={best_si_duration} "
          f"(SI={np.mean(results[best_si_duration]['diverse_si']):.3f})")

    # Find crossover point where diverse beats homo
    for duration in regime_durations:
        diverse_mean = np.mean(results[duration]["diverse_reward"])
        homo_mean = np.mean(results[duration]["homo_reward"])

        if diverse_mean > homo_mean:
            print(f"✅ Diverse beats Homogeneous at duration={duration}")
        else:
            print(f"⚠️ Homogeneous beats Diverse at duration={duration}")

    # Correlation analysis
    all_durations = []
    all_sis = []
    for duration in regime_durations:
        for si in results[duration]["diverse_si"]:
            all_durations.append(duration)
            all_sis.append(si)

    correlation = np.corrcoef(all_durations, all_sis)[0, 1]
    print(f"\nCorrelation (duration, SI): {correlation:.3f}")

    summary["correlation_duration_si"] = float(correlation)

    # Save results
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_regime_sensitivity_experiment(n_trials=30, n_iterations=2000)
