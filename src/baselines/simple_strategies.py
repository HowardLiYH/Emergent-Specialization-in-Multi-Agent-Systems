"""
Simple Trading Strategy Baselines.

These are classic quantitative strategies used as baselines:
1. Buy and Hold - Passive investment
2. Momentum - Buy recent winners
3. Mean Reversion - Buy losers, sell winners
"""

from typing import Dict, List
import numpy as np
import pandas as pd


class BuyAndHold:
    """
    Buy and Hold baseline.

    Simply holds a long position throughout the entire period.
    This is the simplest possible strategy and represents
    the passive investment baseline.
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size
        self.name = "BuyAndHold"

    def get_signal(self, prices: pd.DataFrame) -> Dict:
        """Always return long signal."""
        return {
            "signal": self.position_size,
            "confidence": 1.0,
        }

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series = None,
    ) -> Dict:
        """Run buy-and-hold through entire episode."""
        if len(prices) < 2:
            return {"total_return": 0.0, "sharpe": 0.0}

        # Simple return calculation
        start_price = prices["close"].iloc[0]
        end_price = prices["close"].iloc[-1]
        total_return = (end_price / start_price) - 1

        # Calculate Sharpe from daily returns
        returns = prices["close"].pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 / 6)  # Annualized for 4h bars

        return {
            "total_return": float(total_return),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "sharpe": float(sharpe),
            "n_steps": len(returns),
        }


class MomentumStrategy:
    """
    Momentum Strategy baseline.

    Classic 12-1 month momentum: buy recent winners, sell recent losers.

    Reference: Jegadeesh, N. & Titman, S. (1993). "Returns to Buying
              Winners and Selling Losers." Journal of Finance.
    """

    def __init__(self, lookback: int = 20, threshold: float = 0.0):
        """
        Args:
            lookback: Number of bars to look back for momentum
            threshold: Minimum return to trigger signal
        """
        self.lookback = lookback
        self.threshold = threshold
        self.name = "Momentum"

    def get_signal(self, prices: pd.DataFrame) -> Dict:
        """Get momentum signal."""
        if len(prices) < self.lookback:
            return {"signal": 0.0, "confidence": 0.3}

        # Calculate momentum (return over lookback period)
        momentum = (prices["close"].iloc[-1] / prices["close"].iloc[-self.lookback]) - 1

        # Convert to signal
        if momentum > self.threshold:
            signal = min(1.0, momentum * 5)
        elif momentum < -self.threshold:
            signal = max(-1.0, momentum * 5)
        else:
            signal = 0.0

        confidence = min(0.9, 0.5 + abs(momentum) * 2)

        return {"signal": float(signal), "confidence": float(confidence)}

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series = None,
    ) -> Dict:
        """Run momentum strategy through episode."""
        if len(prices) < self.lookback + 1:
            return {"total_return": 0.0, "sharpe": 0.0}

        rewards = []

        for i in range(self.lookback, len(prices) - 1):
            # Get signal
            window = prices.iloc[:i+1]
            signal_info = self.get_signal(window)
            signal = signal_info["signal"]

            # Calculate return for next bar
            next_return = (prices["close"].iloc[i+1] / prices["close"].iloc[i]) - 1

            # Reward is signal * return
            reward = signal * next_return
            rewards.append(reward)

        rewards = np.array(rewards)

        return {
            "total_return": float(np.sum(rewards)),
            "mean_return": float(np.mean(rewards)),
            "std_return": float(np.std(rewards)),
            "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-8)),
            "n_steps": len(rewards),
        }


class MeanReversionStrategy:
    """
    Mean Reversion Strategy baseline.

    Uses Bollinger Bands to identify overbought/oversold conditions.
    Sells when price is above upper band, buys when below lower band.
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Args:
            period: Bollinger band period
            std_dev: Number of standard deviations for bands
        """
        self.period = period
        self.std_dev = std_dev
        self.name = "MeanReversion"

    def get_signal(self, prices: pd.DataFrame) -> Dict:
        """Get mean reversion signal."""
        if len(prices) < self.period:
            return {"signal": 0.0, "confidence": 0.3}

        close = prices["close"]
        ma = close.rolling(self.period).mean().iloc[-1]
        std = close.rolling(self.period).std().iloc[-1]

        if std < 1e-8:
            return {"signal": 0.0, "confidence": 0.3}

        upper = ma + self.std_dev * std
        lower = ma - self.std_dev * std
        current = close.iloc[-1]

        # Z-score
        z_score = (current - ma) / std

        # Mean revert: sell high, buy low
        signal = float(np.clip(-z_score / self.std_dev, -1, 1))
        confidence = min(0.8, 0.4 + abs(z_score) * 0.2)

        return {"signal": signal, "confidence": float(confidence)}

    def run_episode(
        self,
        prices: pd.DataFrame,
        regimes: pd.Series = None,
    ) -> Dict:
        """Run mean reversion strategy through episode."""
        if len(prices) < self.period + 1:
            return {"total_return": 0.0, "sharpe": 0.0}

        rewards = []

        for i in range(self.period, len(prices) - 1):
            # Get signal
            window = prices.iloc[:i+1]
            signal_info = self.get_signal(window)
            signal = signal_info["signal"]

            # Calculate return for next bar
            next_return = (prices["close"].iloc[i+1] / prices["close"].iloc[i]) - 1

            # Reward is signal * return
            reward = signal * next_return
            rewards.append(reward)

        rewards = np.array(rewards)

        return {
            "total_return": float(np.sum(rewards)),
            "mean_return": float(np.mean(rewards)),
            "std_return": float(np.std(rewards)),
            "sharpe": float(np.mean(rewards) / (np.std(rewards) + 1e-8)),
            "n_steps": len(rewards),
        }
