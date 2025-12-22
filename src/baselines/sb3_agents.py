"""
Stable-Baselines3 RL Agents for Trading.

Uses battle-tested implementations of DQN, PPO, and A2C
for proper baseline comparisons.

Note: Requires stable-baselines3 to be installed:
    pip install stable-baselines3[extra]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

# Try to import SB3, fallback to custom implementations if not available
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Using fallback implementations.")


class TrainingProgressCallback(BaseCallback if SB3_AVAILABLE else object):
    """Callback to track training progress."""

    def __init__(self, verbose: int = 0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1

        # Check if episode ended
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True

    def get_learning_curve(self, window: int = 100) -> np.ndarray:
        """Get smoothed learning curve."""
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards)

        return np.convolve(
            self.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )


# Default hyperparameters (tuned for trading)
DQN_HYPERPARAMS = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "learning_starts": 1000,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [128, 128]},
}

PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [{"pi": [64, 64], "vf": [64, 64]}]},
}

A2C_HYPERPARAMS = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": [{"pi": [64, 64], "vf": [64, 64]}]},
}


class SB3AgentWrapper:
    """Wrapper for SB3 agents with consistent interface."""

    def __init__(
        self,
        algorithm: str = "DQN",
        env=None,
        hyperparams: Optional[Dict] = None,
        seed: int = 42,
    ):
        self.algorithm = algorithm
        self.seed = seed
        self.model = None
        self.callback = None

        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required for SB3AgentWrapper")

        # Select algorithm and hyperparams
        if algorithm == "DQN":
            AlgClass = DQN
            default_params = DQN_HYPERPARAMS.copy()
        elif algorithm == "PPO":
            AlgClass = PPO
            default_params = PPO_HYPERPARAMS.copy()
        elif algorithm == "A2C":
            AlgClass = A2C
            default_params = A2C_HYPERPARAMS.copy()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Merge with custom hyperparams
        if hyperparams:
            default_params.update(hyperparams)

        # Create model
        if env is not None:
            self.model = AlgClass(
                "MlpPolicy",
                env,
                seed=seed,
                verbose=0,
                **default_params,
            )

    def train(
        self,
        total_timesteps: int,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Train the agent."""
        self.callback = TrainingProgressCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            progress_bar=progress_bar,
        )

        return {
            "episode_rewards": self.callback.episode_rewards,
            "learning_curve": self.callback.get_learning_curve().tolist(),
        }

    def evaluate(
        self,
        env,
        n_eval_episodes: int = 100,
    ) -> Tuple[float, float]:
        """Evaluate the trained agent."""
        mean_reward, std_reward = evaluate_policy(
            self.model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        return mean_reward, std_reward

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Select action given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)

    def load(self, path: str, env=None):
        """Load a saved model."""
        if self.algorithm == "DQN":
            self.model = DQN.load(path, env=env)
        elif self.algorithm == "PPO":
            self.model = PPO.load(path, env=env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(path, env=env)


def hyperparameter_search(
    algorithm: str,
    env,
    param_grid: Dict[str, List],
    total_timesteps: int = 10000,
    n_eval_episodes: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Grid search over hyperparameters.

    Args:
        algorithm: "DQN", "PPO", or "A2C"
        env: Gym environment
        param_grid: Dict of param_name -> list of values to try
        total_timesteps: Training steps per configuration
        n_eval_episodes: Episodes for evaluation
        seed: Random seed

    Returns:
        Dict with best params and all results
    """
    if not SB3_AVAILABLE:
        return {"error": "SB3 not available"}

    from itertools import product

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []
    best_reward = -np.inf
    best_params = None

    for combo in combinations:
        params = dict(zip(param_names, combo))

        try:
            # Create and train agent
            agent = SB3AgentWrapper(
                algorithm=algorithm,
                env=env,
                hyperparams=params,
                seed=seed,
            )
            agent.train(total_timesteps=total_timesteps, progress_bar=False)

            # Evaluate
            mean_reward, std_reward = agent.evaluate(env, n_eval_episodes)

            results.append({
                "params": params,
                "mean_reward": mean_reward,
                "std_reward": std_reward,
            })

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_params = params

        except Exception as e:
            results.append({
                "params": params,
                "error": str(e),
            })

    return {
        "best_params": best_params,
        "best_reward": best_reward,
        "all_results": results,
    }


# Fallback implementations when SB3 is not available
class SimpleDQNFallback:
    """Simple DQN fallback when SB3 is not available."""

    def __init__(self, n_actions: int, state_dim: int = 10, seed: int = 42):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

        # Simple Q-table approximation
        self.q_table = np.zeros((100, n_actions))  # Discretized states
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.lr = 0.1
        self.gamma = 0.99

    def _discretize(self, state: np.ndarray) -> int:
        """Discretize state for Q-table."""
        # Use first feature (return) discretized to 100 bins
        val = np.clip(state[0], -1, 1)
        return int((val + 1) / 2 * 99)

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        s = self._discretize(state)
        return int(np.argmax(self.q_table[s]))

    def update(self, state, action, reward, next_state, done):
        s = self._discretize(state)
        ns = self._discretize(next_state)

        target = reward + self.gamma * np.max(self.q_table[ns]) * (1 - done)
        self.q_table[s, action] += self.lr * (target - self.q_table[s, action])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
