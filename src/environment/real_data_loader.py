"""
Real Data Loader for Experiment 6.

Loads historical cryptocurrency data from Bybit CSVs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def load_bybit_data(
    symbol: str,
    data_dir: str = "../../MAS_Final_With_Agents/data/bybit",
) -> pd.DataFrame:
    """
    Load Bybit data for a symbol.

    Args:
        symbol: Coin symbol (BTC, ETH, SOL, etc.)
        data_dir: Path to bybit data directory

    Returns:
        DataFrame with OHLCV data
    """
    data_path = Path(data_dir) / f"Bybit_{symbol}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Parse timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_utc"])
        df = df.set_index("timestamp")

    # Ensure required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_multi_asset_data(
    symbols: List[str] = ["BTC", "ETH", "SOL"],
    data_dir: str = "../../MAS_Final_With_Agents/data/bybit",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple assets.

    Args:
        symbols: List of coin symbols
        data_dir: Path to bybit data directory
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Dict mapping symbol to DataFrame
    """
    data = {}

    for symbol in symbols:
        try:
            df = load_bybit_data(symbol, data_dir)

            # Apply date filters
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            data[symbol] = df
        except FileNotFoundError:
            print(f"Warning: Data not found for {symbol}")

    return data


def label_regimes_hmm(
    prices: pd.DataFrame,
    n_regimes: int = 4,
) -> pd.Series:
    """
    Label regimes using a simple HMM-like approach.

    This is a simplified version for the paper.
    Uses volatility + returns to classify regimes.

    Args:
        prices: Price DataFrame
        n_regimes: Number of regimes to detect

    Returns:
        Series with regime labels
    """
    close = prices["close"]

    # Compute features
    returns = close.pct_change()
    volatility = returns.rolling(20).std()
    momentum = close.pct_change(20)

    # Simple rule-based classification
    labels = []

    for i in range(len(prices)):
        if i < 20:
            labels.append("unknown")
            continue

        ret = momentum.iloc[i]
        vol = volatility.iloc[i]

        # Classify based on returns and volatility
        if vol > volatility.median() * 1.5:
            label = "volatile"
        elif ret > 0.05:
            label = "trend_up"
        elif ret < -0.05:
            label = "trend_down"
        else:
            label = "mean_revert"

        labels.append(label)

    return pd.Series(labels, index=prices.index, name="regime")


def prepare_experiment6_data(
    data_dir: str = "../../MAS_Final_With_Agents/data/bybit",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for Experiment 6.

    Returns train and test sets with regime labels.

    Args:
        data_dir: Path to data directory
        train_end: End of training period
        test_start: Start of test period

    Returns:
        Tuple of (train_prices, train_regimes, test_prices, test_regimes)
    """
    # Load BTC as primary asset
    df = load_bybit_data("BTC", data_dir)

    # Split train/test
    train_df = df[df.index <= train_end]
    test_df = df[df.index >= test_start]

    # Label regimes
    train_regimes = label_regimes_hmm(train_df)
    test_regimes = label_regimes_hmm(test_df)

    return train_df, train_regimes, test_df, test_regimes
