"""
Diagnostic Analysis: Real Data Performance Gap

Investigates why homogeneous VolScalp beats diverse population on BTC.

Key questions:
1. In which regimes does each strategy win?
2. Are detected regimes aligned with strategy strengths?
3. What is the regime distribution difference between train/test?
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.environment.real_data_loader import (
    load_bybit_data,
    prepare_real_data_experiment,
    get_regime_statistics,
)
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.baselines.oracle import OracleSpecialist, REGIME_OPTIMAL_METHODS_V2


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


def diagnose_real_data(
    symbol: str = "BTC",
    output_dir: str = "results/diagnose_real_data",
):
    """
    Run diagnostic analysis on real data.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DIAGNOSTIC ANALYSIS: {symbol}")
    print("=" * 60)

    # Load and prepare data
    train_prices, train_regimes, test_prices, test_regimes, detector = \
        prepare_real_data_experiment(symbol=symbol, train_ratio=0.7)

    print(f"\nData: {len(train_prices)} train, {len(test_prices)} test bars")

    # 1. Analyze regime distribution
    print("\n--- Regime Distribution ---")
    train_stats = get_regime_statistics(train_prices, train_regimes)
    test_stats = get_regime_statistics(test_prices, test_regimes)

    print("\nTrain:")
    print(train_stats.to_string(index=False))
    print("\nTest:")
    print(test_stats.to_string(index=False))

    # 2. Analyze per-method performance by regime
    print("\n--- Method Performance by Regime ---")

    window_size = 20
    method_regime_rewards = defaultdict(lambda: defaultdict(list))

    # Evaluate each method on training data
    for method_name in METHOD_INVENTORY_V2.keys():
        for i in range(window_size, len(train_prices) - 1):
            regime = train_regimes.iloc[i]
            if regime == "unknown":
                continue

            price_window = train_prices.iloc[i-window_size:i+1]
            reward = compute_reward([method_name], price_window)
            method_regime_rewards[method_name][regime].append(reward)

    # Summarize
    print(f"\n{'Method':<15} | {'trend_up':>10} | {'trend_down':>10} | {'mean_revert':>10} | {'volatile':>10}")
    print("-" * 70)

    method_summary = {}
    for method_name in METHOD_INVENTORY_V2.keys():
        regime_means = {}
        row = f"{method_name:<15} |"
        for regime in ["trend_up", "trend_down", "mean_revert", "volatile"]:
            rewards = method_regime_rewards[method_name][regime]
            mean_reward = np.mean(rewards) if rewards else 0.0
            regime_means[regime] = mean_reward
            row += f" {mean_reward:>10.4f} |"
        print(row)
        method_summary[method_name] = regime_means

    # 3. Identify best method per regime (on train data)
    print("\n--- Best Method per Regime (Training Data) ---")

    best_methods = {}
    for regime in ["trend_up", "trend_down", "mean_revert", "volatile"]:
        best_method = None
        best_reward = -np.inf
        for method_name, regime_rewards in method_summary.items():
            if regime_rewards[regime] > best_reward:
                best_reward = regime_rewards[regime]
                best_method = method_name
        best_methods[regime] = (best_method, best_reward)
        print(f"  {regime}: {best_method} ({best_reward:.4f})")

    # 4. Compare with Oracle's assumed optimal methods
    print("\n--- Oracle vs Empirical Optimal Methods ---")

    for regime in ["trend_up", "trend_down", "mean_revert", "volatile"]:
        oracle_methods = REGIME_OPTIMAL_METHODS_V2.get(regime, [])
        empirical_best = best_methods[regime][0]
        match = empirical_best in oracle_methods
        status = "✅" if match else "❌"
        print(f"  {regime}: Oracle={oracle_methods[0] if oracle_methods else 'N/A'}, "
              f"Empirical={empirical_best} {status}")

    # 5. Analyze VolScalp specifically
    print("\n--- VolScalp Analysis ---")

    volscalp_total = 0
    volscalp_by_regime = defaultdict(float)
    counts_by_regime = defaultdict(int)

    for i in range(window_size, len(train_prices) - 1):
        regime = train_regimes.iloc[i]
        if regime == "unknown":
            continue

        price_window = train_prices.iloc[i-window_size:i+1]
        reward = compute_reward(["VolScalp"], price_window)
        volscalp_total += reward
        volscalp_by_regime[regime] += reward
        counts_by_regime[regime] += 1

    print(f"  Total VolScalp reward: {volscalp_total:.2f}")
    for regime in ["trend_up", "trend_down", "mean_revert", "volatile"]:
        avg = volscalp_by_regime[regime] / max(counts_by_regime[regime], 1)
        total = volscalp_by_regime[regime]
        count = counts_by_regime[regime]
        print(f"  {regime}: total={total:.2f}, avg={avg:.4f}, count={count}")

    # 6. Check if diverse population assigns agents correctly
    print("\n--- Recommendation ---")

    # Identify misalignment
    misaligned = []
    for regime in ["trend_up", "trend_down", "mean_revert", "volatile"]:
        oracle_methods = REGIME_OPTIMAL_METHODS_V2.get(regime, [])
        empirical_best = best_methods[regime][0]
        if empirical_best not in oracle_methods:
            misaligned.append((regime, oracle_methods[0] if oracle_methods else "N/A", empirical_best))

    if misaligned:
        print("\n⚠️ MISALIGNMENT DETECTED:")
        print("   The Oracle's assumed optimal methods don't match empirical performance.")
        print("   This could explain why diverse population underperforms.")
        print("\n   Suggested fix: Update REGIME_OPTIMAL_METHODS_V2 based on empirical data:")

        new_optimal = {}
        for regime in ["trend_up", "trend_down", "mean_revert", "volatile"]:
            # Get top 2 methods for this regime
            sorted_methods = sorted(
                method_summary.items(),
                key=lambda x: x[1][regime],
                reverse=True
            )
            new_optimal[regime] = [m[0] for m in sorted_methods[:2]]

        print(f"\n   REGIME_OPTIMAL_METHODS_V2 = {{")
        for regime, methods in new_optimal.items():
            print(f'       "{regime}": {methods},')
        print("   }")
    else:
        print("\n✅ Oracle methods align with empirical performance.")
        print("   Issue may be in agent learning dynamics, not method selection.")

    # Save results
    results = {
        "symbol": symbol,
        "train_size": len(train_prices),
        "test_size": len(test_prices),
        "method_summary": {m: {r: float(v) for r, v in rv.items()}
                          for m, rv in method_summary.items()},
        "best_methods": {r: {"method": m, "reward": float(rw)}
                        for r, (m, rw) in best_methods.items()},
        "volscalp_by_regime": dict(volscalp_by_regime),
        "new_optimal_methods": new_optimal if misaligned else None,
    }

    with open(output_path / "diagnosis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    diagnose_real_data(symbol="BTC")
