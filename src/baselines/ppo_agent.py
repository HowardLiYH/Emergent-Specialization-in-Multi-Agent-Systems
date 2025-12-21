"""
PPO (Proximal Policy Optimization) Trading Agent Baseline.

Standard policy gradient RL approach for comparison with our
multi-agent specialization system.

Reference:
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class SimplePolicyNetwork:
    """
    Simple policy network with numpy (no PyTorch dependency).

    Actor-Critic architecture with shared features.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.rng = np.random.default_rng(42)

        # Shared feature layer
        self.W_shared = self.rng.normal(0, 0.1, (state_dim, hidden_dim))
        self.b_shared = np.zeros(hidden_dim)

        # Actor head (policy)
        self.W_actor = self.rng.normal(0, 0.1, (hidden_dim, n_actions))
        self.b_actor = np.zeros(n_actions)

        # Critic head (value)
        self.W_critic = self.rng.normal(0, 0.1, (hidden_dim, 1))
        self.b_critic = np.zeros(1)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass.

        Returns:
            action_probs: Softmax probabilities over actions
            value: State value estimate
        """
        # Shared features
        h = np.maximum(0, state @ self.W_shared + self.b_shared)

        # Actor: softmax policy
        logits = h @ self.W_actor + self.b_actor
        logits = logits - logits.max()  # Stability
        exp_logits = np.exp(logits)
        action_probs = exp_logits / (exp_logits.sum() + 1e-8)

        # Critic: value
        value = float(h @ self.W_critic + self.b_critic)

        return action_probs, value

    def get_action_and_value(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[int, float, float]:
        """
        Sample action and return value.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value
        """
        probs, value = self.forward(state)

        action = rng.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        return action, log_prob, value

    def evaluate(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate actions for PPO update.

        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Policy entropy (for exploration bonus)
        """
        batch_size = len(states)
        log_probs = np.zeros(batch_size)
        values = np.zeros(batch_size)
        entropies = np.zeros(batch_size)

        for i in range(batch_size):
            probs, value = self.forward(states[i])
            log_probs[i] = np.log(probs[actions[i]] + 1e-8)
            values[i] = value
            entropies[i] = -np.sum(probs * np.log(probs + 1e-8))

        return log_probs, values, entropies

    def update_actor(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: np.ndarray,
        clip_epsilon: float = 0.2,
    ) -> float:
        """
        PPO clipped objective update for actor.

        Returns actor loss.
        """
        batch_size = len(states)

        # Compute current log probs
        new_log_probs, _, _ = self.evaluate(states, actions)

        # Importance ratio
        ratio = np.exp(new_log_probs - old_log_probs)

        # Clipped objective
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        # PPO loss (negative because we want to maximize)
        loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))

        # Simple gradient update (approximate)
        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            adv = advantages[i]

            # Forward
            h = np.maximum(0, state @ self.W_shared + self.b_shared)
            logits = h @ self.W_actor + self.b_actor
            logits = logits - logits.max()
            exp_logits = np.exp(logits)
            probs = exp_logits / (exp_logits.sum() + 1e-8)

            # Gradient of log_prob w.r.t. logits (softmax gradient)
            grad_logits = -probs.copy()
            grad_logits[action] += 1
            grad_logits *= adv  # Scale by advantage

            # Update actor weights
            self.W_actor += self.lr_actor * np.outer(h, grad_logits) / batch_size
            self.b_actor += self.lr_actor * grad_logits / batch_size

        return loss

    def update_critic(
        self,
        states: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """
        Update critic using MSE loss.

        Returns critic loss.
        """
        batch_size = len(states)

        total_loss = 0.0
        for i in range(batch_size):
            state = states[i]
            target = returns[i]

            # Forward
            h = np.maximum(0, state @ self.W_shared + self.b_shared)
            value = float(h @ self.W_critic + self.b_critic)

            # MSE gradient
            error = value - target
            total_loss += error ** 2

            # Update critic
            grad_critic = 2 * error * h / batch_size
            self.W_critic -= self.lr_critic * grad_critic.reshape(-1, 1)
            self.b_critic -= self.lr_critic * 2 * error / batch_size

        return total_loss / batch_size


class PPOTradingAgent:
    """
    PPO agent for trading method selection.

    Uses actor-critic architecture with clipped objective.
    """

    def __init__(
        self,
        n_actions: int,
        state_dim: int = 10,
        hidden_dim: int = 64,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
        batch_size: int = 32,
        buffer_size: int = 2048,
    ):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Network
        self.network = SimplePolicyNetwork(
            state_dim, n_actions, hidden_dim, lr_actor, lr_critic
        )

        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

        self.rng = np.random.default_rng(42)
        self.step_count = 0

    def extract_state(self, prices) -> np.ndarray:
        """Extract state features from price window."""
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

        state = np.array([*returns, vol, momentum, pos, vol_trend, rsi])
        return np.clip(state, -10, 10)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using policy."""
        if training:
            action, log_prob, value = self.network.get_action_and_value(state, self.rng)
            self.log_probs.append(log_prob)
            self.values.append(value)
            return action
        else:
            probs, _ = self.network.forward(state)
            return int(np.argmax(probs))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ):
        """Store transition in buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.step_count += 1

    def update(self) -> Optional[Tuple[float, float]]:
        """
        Run PPO update if buffer is full.

        Returns (actor_loss, critic_loss) or None.
        """
        if len(self.states) < self.buffer_size:
            return None

        # Convert to arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        log_probs = np.array(self.log_probs)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Compute returns and advantages using GAE
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        last_value = 0
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            returns[t] = advantages[t] + values[t]
            last_advantage = advantages[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_actor_loss = 0
        total_critic_loss = 0

        for _ in range(self.n_epochs):
            # Shuffle and batch
            indices = self.rng.permutation(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                actor_loss = self.network.update_actor(
                    states[batch_idx],
                    actions[batch_idx],
                    advantages[batch_idx],
                    log_probs[batch_idx],
                    self.clip_epsilon,
                )

                critic_loss = self.network.update_critic(
                    states[batch_idx],
                    returns[batch_idx],
                )

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss

        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

        n_batches = self.n_epochs * (len(states) // self.batch_size + 1)
        return total_actor_loss / n_batches, total_critic_loss / n_batches

    def get_action_distribution(self) -> np.ndarray:
        """Get current policy distribution."""
        if len(self.states) < 10:
            return np.ones(self.n_actions) / self.n_actions

        # Average over recent states
        recent_states = np.array(self.states[-100:])
        probs = np.zeros(self.n_actions)

        for state in recent_states:
            p, _ = self.network.forward(state)
            probs += p

        return probs / len(recent_states)
