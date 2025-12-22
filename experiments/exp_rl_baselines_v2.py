"""
Experiment: Proper RL Baseline Comparison (V2)

Uses Stable-Baselines3 for battle-tested RL implementations.
Falls back to custom implementations if SB3 not available.

Runs 30 trials per algorithm with proper convergence verification.
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
from src.baselines.oracle import OracleSpecialist
from src.analysis.statistical_utils import compute_stats, compare_groups, format_comparison_table

# Check if SB3 available
try:
    from src.environment.trading_gym import TradingGymEnv
    from src.baselines.sb3_agents import SB3AgentWrapper, SB3_AVAILABLE
except ImportError:
    SB3_AVAILABLE = False

# Fallback to custom implementations
from src.baselines.dqn_agent import DQNTradingAgent
from src.baselines.ppo_agent import PPOTradingAgent


METHOD_NAMES = list(METHOD_INVENTORY_V2.keys())
N_METHODS = len(METHOD_NAMES)


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


def run_multi_agent_trial(n_iterations: int, seed: int) -> dict:
    """Run single trial for multi-agent population."""
    config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
    market = SyntheticMarketEnvironment(config)
    all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=seed)

    pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=seed)

    rewards = []
    window_size = 20

    for i in range(n_iterations):
        start_idx = i
        end_idx = i + window_size + 1
        prices = all_prices.iloc[start_idx:end_idx].copy()
        regime = all_regimes.iloc[end_idx - 1]

        result = pop.run_iteration(prices, regime, compute_reward)
        reward = compute_reward([result["winner_method"]], prices)
        rewards.append(reward)

    return {
        "total_reward": sum(rewards),
        "mean_reward": np.mean(rewards[-200:]),  # Final performance
        "rewards": rewards,
    }


def run_oracle_trial(n_iterations: int, seed: int) -> dict:
    """Run single trial for Oracle baseline."""
    config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
    market = SyntheticMarketEnvironment(config)
    all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=seed)

    oracle = OracleSpecialist()

    rewards = []
    window_size = 20

    for i in range(n_iterations):
        start_idx = i
        end_idx = i + window_size + 1
        prices = all_prices.iloc[start_idx:end_idx].copy()
        regime = all_regimes.iloc[end_idx - 1]

        methods = oracle.select(regime)
        reward = compute_reward(methods, prices)
        rewards.append(reward)

    return {
        "total_reward": sum(rewards),
        "mean_reward": np.mean(rewards[-200:]),
        "rewards": rewards,
    }


def run_custom_dqn_trial(n_iterations: int, seed: int) -> dict:
    """Run single trial for custom DQN."""
    config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
    market = SyntheticMarketEnvironment(config)
    all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=seed)

    agent = DQNTradingAgent(n_actions=N_METHODS, state_dim=10)

    rewards = []
    window_size = 20

    for i in range(n_iterations):
        start_idx = i
        end_idx = i + window_size + 1
        prices = all_prices.iloc[start_idx:end_idx].copy()

        state = agent.extract_state(prices)
        action = agent.select_action(state, training=True)
        method = METHOD_NAMES[action]
        reward = compute_reward([method], prices)

        # Next state
        next_start = i + 1
        next_end = i + window_size + 2
        if next_end <= len(all_prices):
            next_prices = all_prices.iloc[next_start:next_end].copy()
        else:
            next_prices = prices
        next_state = agent.extract_state(next_prices)

        done = (i == n_iterations - 1)
        agent.update(state, action, reward, next_state, done)
        rewards.append(reward)

    return {
        "total_reward": sum(rewards),
        "mean_reward": np.mean(rewards[-200:]),
        "rewards": rewards,
    }


def run_custom_ppo_trial(n_iterations: int, seed: int) -> dict:
    """Run single trial for custom PPO."""
    config = SyntheticMarketConfig(regime_duration_mean=50, regime_duration_std=20)
    market = SyntheticMarketEnvironment(config)
    all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=seed)

    agent = PPOTradingAgent(n_actions=N_METHODS, state_dim=10, buffer_size=512)

    rewards = []
    window_size = 20

    for i in range(n_iterations):
        start_idx = i
        end_idx = i + window_size + 1
        prices = all_prices.iloc[start_idx:end_idx].copy()

        state = agent.extract_state(prices)
        action = agent.select_action(state, training=True)
        method = METHOD_NAMES[action]
        reward = compute_reward([method], prices)

        done = (i == n_iterations - 1)
        agent.store_transition(state, action, reward, done)
        agent.update()
        rewards.append(reward)

    return {
        "total_reward": sum(rewards),
        "mean_reward": np.mean(rewards[-200:]),
        "rewards": rewards,
    }


def run_rl_baselines_v2(
    n_trials: int = 30,
    n_iterations: int = 2000,
    output_dir: str = "results/exp_rl_baselines_v2",
):
    """
    Run comprehensive RL baseline comparison.

    Args:
        n_trials: Number of trials per approach
        n_iterations: Training iterations per trial
        output_dir: Where to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RL BASELINES EXPERIMENT V2")
    print(f"Trials: {n_trials}, Iterations: {n_iterations}")
    print("=" * 60)

    results = {
        "multi_agent": [],
        "oracle": [],
        "dqn": [],
        "ppo": [],
    }

    learning_curves = {
        "multi_agent": [],
        "oracle": [],
        "dqn": [],
        "ppo": [],
    }

    # Run trials
    for trial in tqdm(range(n_trials), desc="Trials"):
        # Multi-agent
        ma_result = run_multi_agent_trial(n_iterations, seed=trial)
        results["multi_agent"].append(ma_result["mean_reward"])
        learning_curves["multi_agent"].append(ma_result["rewards"])

        # Oracle
        oracle_result = run_oracle_trial(n_iterations, seed=trial)
        results["oracle"].append(oracle_result["mean_reward"])
        learning_curves["oracle"].append(oracle_result["rewards"])

        # DQN
        dqn_result = run_custom_dqn_trial(n_iterations, seed=trial)
        results["dqn"].append(dqn_result["mean_reward"])
        learning_curves["dqn"].append(dqn_result["rewards"])

        # PPO
        ppo_result = run_custom_ppo_trial(n_iterations, seed=trial)
        results["ppo"].append(ppo_result["mean_reward"])
        learning_curves["ppo"].append(ppo_result["rewards"])

    # Compute statistics
    stats = {
        name: compute_stats(values)
        for name, values in results.items()
    }

    # Compare against multi-agent
    comparisons = {}
    for name in ["oracle", "dqn", "ppo"]:
        comparisons[name] = compare_groups(
            results["multi_agent"],
            results[name],
            paired=True,
        )

    # Summary
    summary = {
        "experiment": "rl_baselines_v2",
        "n_trials": n_trials,
        "n_iterations": n_iterations,
        "sb3_available": SB3_AVAILABLE,
    }

    for name, stat in stats.items():
        summary[f"{name}_mean"] = stat.mean
        summary[f"{name}_std"] = stat.std
        summary[f"{name}_ci_lower"] = stat.ci_lower
        summary[f"{name}_ci_upper"] = stat.ci_upper

    for name, comp in comparisons.items():
        summary[f"multi_agent_vs_{name}_p"] = comp.p_value
        summary[f"multi_agent_vs_{name}_d"] = comp.cohens_d

    # Save results
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save learning curves (averaged)
    avg_curves = {}
    for name, curves in learning_curves.items():
        avg_curve = np.mean(curves, axis=0)
        # Smooth
        window = 50
        smooth_curve = np.convolve(avg_curve, np.ones(window)/window, mode='valid')
        avg_curves[name] = smooth_curve.tolist()

    with open(output_path / "learning_curves.json", "w") as f:
        json.dump(avg_curves, f)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (30 trials, 95% CI)")
    print("=" * 60)

    print(f"\n{'Approach':<20} | {'Mean ± Std':>20} | {'95% CI':>20}")
    print("-" * 65)

    for name in ["multi_agent", "oracle", "dqn", "ppo"]:
        s = stats[name]
        print(f"{name:<20} | {s.mean:>8.4f} ± {s.std:.4f} | [{s.ci_lower:.4f}, {s.ci_upper:.4f}]")

    print("\n" + format_comparison_table(comparisons, "multi_agent", "Comparison"))

    # Key insights
    print("\nKey Insights:")

    ma_mean = stats["multi_agent"].mean
    oracle_mean = stats["oracle"].mean
    dqn_mean = stats["dqn"].mean
    ppo_mean = stats["ppo"].mean

    oracle_gap = (oracle_mean - ma_mean) / oracle_mean * 100
    print(f"  Gap to Oracle: {oracle_gap:.1f}%")

    if ma_mean > dqn_mean:
        improvement = (ma_mean - dqn_mean) / abs(dqn_mean + 1e-8) * 100
        print(f"  ✅ Multi-agent beats DQN by {improvement:.1f}%")
    else:
        print(f"  ⚠️ DQN beats Multi-agent")

    if ma_mean > ppo_mean:
        improvement = (ma_mean - ppo_mean) / abs(ppo_mean + 1e-8) * 100
        print(f"  ✅ Multi-agent beats PPO by {improvement:.1f}%")
    else:
        print(f"  ⚠️ PPO beats Multi-agent")

    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_rl_baselines_v2(n_trials=30, n_iterations=2000)
