"""
DQN (Deep Q-Network) Trading Agent Baseline.

Standard single-agent RL approach for comparison with our
multi-agent specialization system.

Reference:
- Mnih et al. (2015). "Human-level control through deep reinforcement learning"
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class SimpleQNetwork:
    """
    Simple Q-Network implemented with numpy (no PyTorch dependency).

    Uses a 2-layer neural network approximation.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr: float = 0.001,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Initialize weights
        self.rng = np.random.default_rng(42)

        # Layer 1: state_dim -> hidden_dim
        self.W1 = self.rng.normal(0, 0.1, (state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Layer 2: hidden_dim -> n_actions
        self.W2 = self.rng.normal(0, 0.1, (hidden_dim, n_actions))
        self.b2 = np.zeros(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass to get Q-values."""
        # ReLU activation
        h = np.maximum(0, state @ self.W1 + self.b1)
        q_values = h @ self.W2 + self.b2
        return q_values

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Update network using gradient descent.

        Returns loss for monitoring.
        """
        batch_size = len(states)

        # Forward pass
        h = np.maximum(0, states @ self.W1 + self.b1)  # (batch, hidden)
        q_values = h @ self.W2 + self.b2  # (batch, n_actions)

        # Compute loss: only for taken actions
        q_taken = q_values[np.arange(batch_size), actions]
        loss = np.mean((q_taken - targets) ** 2)

        # Gradient of loss w.r.t. q_taken
        d_q = 2 * (q_taken - targets) / batch_size  # (batch,)

        # Backprop through output layer
        d_W2 = np.zeros_like(self.W2)
        d_b2 = np.zeros_like(self.b2)
        d_h = np.zeros((batch_size, self.hidden_dim))

        for i in range(batch_size):
            d_W2[:, actions[i]] += h[i] * d_q[i]
            d_b2[actions[i]] += d_q[i]
            d_h[i] = self.W2[:, actions[i]] * d_q[i]

        # Backprop through ReLU
        d_h = d_h * (h > 0)  # ReLU gradient

        # Backprop through input layer
        d_W1 = states.T @ d_h
        d_b1 = d_h.sum(axis=0)

        # Update weights
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

        return loss

    def copy_from(self, other: "SimpleQNetwork"):
        """Copy weights from another network."""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()


class DQNTradingAgent:
    """
    DQN agent for trading method selection.

    State: Price features (returns, volatility, etc.)
    Action: Select a trading method from inventory
    Reward: Trading reward based on method performance
    """

    def __init__(
        self,
        n_actions: int,
        state_dim: int = 10,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
    ):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Networks
        self.q_network = SimpleQNetwork(state_dim, n_actions, hidden_dim, lr)
        self.target_network = SimpleQNetwork(state_dim, n_actions, hidden_dim, lr)
        self.target_network.copy_from(self.q_network)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.step_count = 0
        self.rng = np.random.default_rng(42)

    def extract_state(self, prices) -> np.ndarray:
        """
        Extract state features from price window.

        Features:
        - Recent returns (5)
        - Volatility (1)
        - Momentum (1)
        - Price position in range (1)
        - Volume trend (1)
        - RSI approximation (1)
        """
        close = prices["close"].values

        if len(close) < 20:
            return np.zeros(self.state_dim)

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
        if "volume" in prices.columns:
            vol_trend = prices["volume"].iloc[-5:].mean() / (prices["volume"].iloc[-20:].mean() + 1e-8)
        else:
            vol_trend = 1.0

        # RSI approximation
        gains = np.maximum(returns, 0).mean()
        losses = np.maximum(-returns, 0).mean()
        rsi = gains / (gains + losses + 1e-8)

        state = np.array([
            *returns,  # 5
            vol,       # 1
            momentum,  # 1
            pos,       # 1
            vol_trend, # 1
            rsi,       # 1
        ])

        # Normalize
        state = np.clip(state, -10, 10)

        return state

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)

        q_values = self.q_network.forward(state)
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store experience and train."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Train if enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Compute targets
        next_q = self.target_network.forward(next_states)
        max_next_q = next_q.max(axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Update Q-network
        loss = self.q_network.update(states, actions, targets)

        # Update target network
        if self.step_count % self.target_update == 0:
            self.target_network.copy_from(self.q_network)

        return loss

    def get_action_distribution(self) -> np.ndarray:
        """Get softmax distribution over actions (for comparison)."""
        # Compute average Q-values from replay
        if len(self.replay_buffer) < 100:
            return np.ones(self.n_actions) / self.n_actions

        states, _, _, _, _ = self.replay_buffer.sample(100)
        q_values = np.array([self.q_network.forward(s) for s in states])
        mean_q = q_values.mean(axis=0)

        # Softmax
        exp_q = np.exp(mean_q - mean_q.max())
        return exp_q / exp_q.sum()
