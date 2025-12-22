"""
Experiment: RL Baseline Comparison

Compares our multi-agent specialization system against standard
single-agent RL approaches (DQN and PPO).

This addresses reviewer concern #3: "Limited Baselines"

Key questions:
1. Does multi-agent specialization outperform single-agent RL?
2. Do RL agents learn meaningful policies?
3. What are the sample efficiency differences?
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from src.environment.synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from src.agents.niche_population import NichePopulation
from src.agents.inventory_v2 import METHOD_INVENTORY_V2
from src.baselines.dqn_agent import DQNTradingAgent
from src.baselines.ppo_agent import PPOTradingAgent


# Get method list
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


def run_rl_baselines_experiment(
    n_trials: int = 5,
    n_iterations: int = 2000,
    output_dir: str = "results/exp_rl_baselines",
):
    """
    Compare multi-agent vs single-agent RL.

    Args:
        n_trials: Number of trials per approach
        n_iterations: Training iterations per trial
        output_dir: Where to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RL BASELINES EXPERIMENT")
    print("=" * 60)

    results = {
        "diverse": {"rewards": [], "final_reward": []},
        "dqn": {"rewards": [], "final_reward": []},
        "ppo": {"rewards": [], "final_reward": []},
    }

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Create environment
        config = SyntheticMarketConfig(
            regime_duration_mean=50,
            regime_duration_std=20,
        )
        market = SyntheticMarketEnvironment(config)

        # Generate full dataset
        all_prices, all_regimes = market.generate(n_bars=n_iterations + 50, seed=trial)

        # Initialize agents
        diverse_pop = NichePopulation(n_agents=8, niche_bonus=0.5, seed=trial)
        dqn_agent = DQNTradingAgent(n_actions=N_METHODS, state_dim=10)
        ppo_agent = PPOTradingAgent(n_actions=N_METHODS, state_dim=10)

        # Training rewards
        diverse_rewards = []
        dqn_rewards = []
        ppo_rewards = []

        # Window size for price context
        window_size = 20

        # Run training
        for i in tqdm(range(n_iterations), desc=f"Trial {trial + 1}", leave=False):
            # Get price window and regime
            start_idx = i
            end_idx = i + window_size + 1
            prices = all_prices.iloc[start_idx:end_idx].copy()
            regime = all_regimes.iloc[end_idx - 1]

            # === Multi-Agent (Diverse) ===
            result = diverse_pop.run_iteration(prices, regime, compute_reward)
            diverse_reward = compute_reward([result["winner_method"]], prices)
            diverse_rewards.append(diverse_reward)

            # === DQN ===
            dqn_state = dqn_agent.extract_state(prices)
            dqn_action = dqn_agent.select_action(dqn_state, training=True)
            dqn_method = METHOD_NAMES[dqn_action]
            dqn_reward = compute_reward([dqn_method], prices)

            # Next state
            next_start_idx = i + 1
            next_end_idx = i + window_size + 2
            if next_end_idx <= len(all_prices):
                next_prices = all_prices.iloc[next_start_idx:next_end_idx].copy()
            else:
                next_prices = prices  # Use current if at end
            dqn_next_state = dqn_agent.extract_state(next_prices)
            done = (i == n_iterations - 1)

            dqn_agent.update(dqn_state, dqn_action, dqn_reward, dqn_next_state, done)
            dqn_rewards.append(dqn_reward)

            # === PPO ===
            ppo_state = ppo_agent.extract_state(prices)
            ppo_action = ppo_agent.select_action(ppo_state, training=True)
            ppo_method = METHOD_NAMES[ppo_action]
            ppo_reward = compute_reward([ppo_method], prices)

            ppo_agent.store_transition(ppo_state, ppo_action, ppo_reward, done)
            ppo_agent.update()  # Will only update when buffer is full
            ppo_rewards.append(ppo_reward)

        # Store results
        results["diverse"]["rewards"].append(diverse_rewards)
        results["diverse"]["final_reward"].append(np.mean(diverse_rewards[-200:]))

        results["dqn"]["rewards"].append(dqn_rewards)
        results["dqn"]["final_reward"].append(np.mean(dqn_rewards[-200:]))

        results["ppo"]["rewards"].append(ppo_rewards)
        results["ppo"]["final_reward"].append(np.mean(ppo_rewards[-200:]))

        print(f"  Diverse final: {results['diverse']['final_reward'][-1]:.4f}")
        print(f"  DQN final: {results['dqn']['final_reward'][-1]:.4f}")
        print(f"  PPO final: {results['ppo']['final_reward'][-1]:.4f}")

    # Compute summary statistics
    summary = {
        "experiment": "rl_baselines",
        "n_trials": n_trials,
        "n_iterations": n_iterations,

        "diverse_mean": float(np.mean(results["diverse"]["final_reward"])),
        "diverse_std": float(np.std(results["diverse"]["final_reward"])),

        "dqn_mean": float(np.mean(results["dqn"]["final_reward"])),
        "dqn_std": float(np.std(results["dqn"]["final_reward"])),

        "ppo_mean": float(np.mean(results["ppo"]["final_reward"])),
        "ppo_std": float(np.std(results["ppo"]["final_reward"])),
    }

    # Statistical tests
    diverse_vs_dqn = stats.ttest_rel(
        results["diverse"]["final_reward"],
        results["dqn"]["final_reward"],
    )
    diverse_vs_ppo = stats.ttest_rel(
        results["diverse"]["final_reward"],
        results["ppo"]["final_reward"],
    )

    summary["diverse_vs_dqn_p"] = float(diverse_vs_dqn[1])
    summary["diverse_vs_ppo_p"] = float(diverse_vs_ppo[1])

    # Learning curves (averaged across trials)
    diverse_curve = np.mean(results["diverse"]["rewards"], axis=0)
    dqn_curve = np.mean(results["dqn"]["rewards"], axis=0)
    ppo_curve = np.mean(results["ppo"]["rewards"], axis=0)

    # Smooth curves
    window = 50
    diverse_smooth = np.convolve(diverse_curve, np.ones(window)/window, mode='valid')
    dqn_smooth = np.convolve(dqn_curve, np.ones(window)/window, mode='valid')
    ppo_smooth = np.convolve(ppo_curve, np.ones(window)/window, mode='valid')

    learning_curves = {
        "diverse": diverse_smooth.tolist(),
        "dqn": dqn_smooth.tolist(),
        "ppo": ppo_smooth.tolist(),
    }

    # Save results
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_path / "learning_curves.json", "w") as f:
        json.dump(learning_curves, f)

    # Print summary
    print("\n" + "=" * 60)
    print("RL BASELINES RESULTS")
    print("=" * 60)

    print(f"\n{'Approach':<20} | {'Final Reward':>15} | {'vs Diverse':>12}")
    print("-" * 55)
    print(f"{'Diverse (Ours)':<20} | {summary['diverse_mean']:>8.4f}±{summary['diverse_std']:.4f} | {'---':>12}")

    dqn_diff = (summary['diverse_mean'] - summary['dqn_mean']) / abs(summary['dqn_mean'] + 1e-8) * 100
    ppo_diff = (summary['diverse_mean'] - summary['ppo_mean']) / abs(summary['ppo_mean'] + 1e-8) * 100

    dqn_sig = "*" if summary['diverse_vs_dqn_p'] < 0.05 else ""
    ppo_sig = "*" if summary['diverse_vs_ppo_p'] < 0.05 else ""

    print(f"{'DQN':<20} | {summary['dqn_mean']:>8.4f}±{summary['dqn_std']:.4f} | {dqn_diff:>+10.1f}%{dqn_sig}")
    print(f"{'PPO':<20} | {summary['ppo_mean']:>8.4f}±{summary['ppo_std']:.4f} | {ppo_diff:>+10.1f}%{ppo_sig}")

    print("\n* = statistically significant (p < 0.05)")

    # Key insights
    print("\nKey Insights:")
    if summary['diverse_mean'] > summary['dqn_mean'] and summary['diverse_mean'] > summary['ppo_mean']:
        print("  ✅ Multi-agent specialization outperforms both single-agent RL approaches")
    else:
        best = "DQN" if summary['dqn_mean'] > summary['ppo_mean'] else "PPO"
        print(f"  ⚠️ Single-agent {best} performs competitively")

    print(f"\nResults saved to {output_path}")

    return summary


if __name__ == "__main__":
    run_rl_baselines_experiment(n_trials=5, n_iterations=2000)
