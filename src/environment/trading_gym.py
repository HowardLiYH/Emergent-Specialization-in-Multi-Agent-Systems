"""
Gym-compatible Trading Environment for Stable-Baselines3.

This wrapper enables proper RL baseline comparisons using
battle-tested implementations from SB3.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .synthetic_market import SyntheticMarketEnvironment, SyntheticMarketConfig
from ..agents.inventory_v2 import METHOD_INVENTORY_V2


# Method names for action mapping
METHOD_NAMES = list(METHOD_INVENTORY_V2.keys())
N_METHODS = len(METHOD_NAMES)


class TradingGymEnv(gym.Env):
    """
    Gym-compatible trading environment for RL agents.

    Observation: 10-dimensional feature vector
        - Recent returns (5)
        - Volatility (1)
        - Momentum (1)
        - Price position (1)
        - Volume trend (1)
        - RSI approximation (1)

    Action: Discrete selection of trading method

    Reward: Profit/loss from selected method
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[SyntheticMarketConfig] = None,
        episode_length: int = 1000,
        window_size: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.config = config or SyntheticMarketConfig(
            regime_duration_mean=50,
            regime_duration_std=20,
        )
        self.episode_length = episode_length
        self.window_size = window_size
        self.seed_value = seed

        # Create market
        self.market = SyntheticMarketEnvironment(self.config)

        # Spaces
        self.action_space = spaces.Discrete(N_METHODS)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10,),
            dtype=np.float32,
        )

        # State
        self.prices = None
        self.regimes = None
        self.current_step = 0
        self.episode_seed = seed or 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Generate new market data
        if seed is not None:
            self.episode_seed = seed
        else:
            self.episode_seed += 1

        self.prices, self.regimes = self.market.generate(
            n_bars=self.episode_length + self.window_size + 10,
            seed=self.episode_seed,
        )

        self.current_step = 0

        obs = self._get_observation()
        info = {"regime": self.regimes.iloc[self.window_size + self.current_step]}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Index of method to use

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current price window
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size + 1
        price_window = self.prices.iloc[start_idx:end_idx].copy()

        # Get selected method
        method_name = METHOD_NAMES[action]

        # Compute reward
        reward = self._compute_reward(method_name, price_window)

        # Get regime for info
        current_regime = self.regimes.iloc[end_idx - 1]

        # Advance step
        self.current_step += 1

        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length

        # Get next observation
        obs = self._get_observation()

        info = {
            "regime": current_regime,
            "method": method_name,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Extract observation from current state."""
        start_idx = self.current_step
        end_idx = self.current_step + self.window_size + 1

        if end_idx > len(self.prices):
            end_idx = len(self.prices)
            start_idx = max(0, end_idx - self.window_size - 1)

        price_window = self.prices.iloc[start_idx:end_idx]
        close = price_window["close"].values

        if len(close) < 20:
            return np.zeros(10, dtype=np.float32)

        # Returns
        returns = np.diff(np.log(close))[-5:]
        if len(returns) < 5:
            returns = np.pad(returns, (5 - len(returns), 0))

        # Volatility
        vol = np.std(returns) if len(returns) > 1 else 0.01

        # Momentum
        momentum = (close[-1] / close[-min(20, len(close))] - 1) if len(close) > 1 else 0

        # Position in range
        high = close[-20:].max()
        low = close[-20:].min()
        pos = (close[-1] - low) / (high - low + 1e-8)

        # Volume trend
        if "volume" in price_window.columns:
            vol_trend = price_window["volume"].iloc[-5:].mean() / (
                price_window["volume"].iloc[-20:].mean() + 1e-8
            )
        else:
            vol_trend = 1.0

        # RSI approximation
        gains = np.maximum(returns, 0).mean()
        losses = np.maximum(-returns, 0).mean()
        rsi = gains / (gains + losses + 1e-8)

        obs = np.array([
            *returns,  # 5
            vol,       # 1
            momentum,  # 1
            pos,       # 1
            vol_trend, # 1
            rsi,       # 1
        ], dtype=np.float32)

        return np.clip(obs, -10, 10)

    def _compute_reward(self, method_name: str, prices) -> float:
        """Compute reward for selected method."""
        if len(prices) < 2:
            return 0.0

        if method_name not in METHOD_INVENTORY_V2:
            return 0.0

        method = METHOD_INVENTORY_V2[method_name]
        result = method.execute(prices)

        signal = result['signal']
        confidence = result['confidence']

        price_return = (prices['close'].iloc[-1] / prices['close'].iloc[-2]) - 1
        reward = float(np.clip(signal * price_return * 10, -1, 1))

        return reward

    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            print(f"Step {self.current_step}, Regime: {self.regimes.iloc[self.window_size + self.current_step]}")

    def close(self):
        """Clean up resources."""
        pass


class VectorizedTradingEnv:
    """
    Vectorized wrapper for parallel environment execution.
    Compatible with SB3's vectorized environment interface.
    """

    def __init__(
        self,
        n_envs: int = 4,
        config: Optional[SyntheticMarketConfig] = None,
        episode_length: int = 1000,
        seed: int = 0,
    ):
        self.n_envs = n_envs
        self.envs = [
            TradingGymEnv(
                config=config,
                episode_length=episode_length,
                seed=seed + i * 1000,
            )
            for i in range(n_envs)
        ]

        # Expose spaces from first env
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self, seed=None):
        """Reset all environments."""
        obs_list = []
        info_list = []

        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)

        return np.array(obs_list), info_list

    def step(self, actions):
        """Step all environments."""
        obs_list = []
        reward_list = []
        term_list = []
        trunc_list = []
        info_list = []

        for env, action in zip(self.envs, actions):
            obs, reward, term, trunc, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)

            # Auto-reset if done
            if term or trunc:
                obs, info = env.reset()
                obs_list[-1] = obs
                info_list[-1] = info

        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(term_list),
            np.array(trunc_list),
            info_list,
        )

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def make_trading_env(
    config: Optional[SyntheticMarketConfig] = None,
    episode_length: int = 1000,
    seed: int = 0,
) -> TradingGymEnv:
    """Factory function to create trading environment."""
    return TradingGymEnv(
        config=config,
        episode_length=episode_length,
        seed=seed,
    )
