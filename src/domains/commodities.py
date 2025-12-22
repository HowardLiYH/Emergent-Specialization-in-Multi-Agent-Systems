"""
Commodities Domain - FRED commodity price prediction with market regimes.

Regimes: bull, bear, sideways, volatile
Methods: Persistence, MA5, MA20, Momentum, MeanReversion
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .base import DomainEnvironment, DomainMethod


COMMODITIES_REGIMES = ["bull", "bear", "sideways", "volatile"]
COMMODITIES_METHODS = ["Persistence", "MA5", "MA20", "Momentum", "MeanReversion"]


class CommodityMethod(DomainMethod):
    """Commodity price prediction method."""
    
    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes
        self._history = []
        self._long_term_mean = None
    
    def predict(self, current_price: float, history: np.ndarray) -> float:
        """Predict next price value."""
        if self.name == "Persistence":
            return current_price
        elif self.name == "MA5":
            if len(history) >= 5:
                return np.mean(history[-5:])
            return current_price
        elif self.name == "MA20":
            if len(history) >= 20:
                return np.mean(history[-20:])
            return current_price
        elif self.name == "Momentum":
            if len(history) >= 5:
                returns = np.diff(history[-5:]) / history[-5:-1]
                avg_return = np.mean(returns)
                return current_price * (1 + avg_return)
            return current_price
        elif self.name == "MeanReversion":
            if self._long_term_mean is None and len(history) > 0:
                self._long_term_mean = np.mean(history)
            
            if self._long_term_mean:
                # Predict reversion toward mean
                deviation = (current_price - self._long_term_mean) / self._long_term_mean
                reversion_rate = 0.1
                return current_price * (1 - reversion_rate * deviation)
            return current_price
        return current_price
    
    def execute(self, observation: np.ndarray) -> Dict:
        """Execute method on observation."""
        price = observation[0] if len(observation) > 0 else 50
        
        prediction = self.predict(price, np.array(self._history))
        self._history.append(price)
        if len(self._history) > 100:
            self._history = self._history[-100:]
            # Update long-term mean
            self._long_term_mean = np.mean(self._history)
        
        # Signal based on predicted direction
        if price > 0:
            pct_change = (prediction - price) / price
        else:
            pct_change = 0
        signal = np.clip(pct_change * 20, -1, 1)  # Scale for sensitivity
        return {"signal": signal, "prediction": prediction, "confidence": 0.5}


def load_commodities_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load FRED commodity price data from CSV."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "commodities" / "fred_prices.csv"
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_commodities_environment(
    n_bars: int = 2000,
    commodity: str = "WTI_Oil",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, CommodityMethod]]:
    """
    Create commodity price prediction environment from real data.
    
    State: [price, returns, price_ma5, price_ma20, vol_20]
    """
    df = load_commodities_data()
    
    # Filter by commodity
    comm_df = df[df['commodity'] == commodity].copy()
    if len(comm_df) == 0:
        commodity = df['commodity'].iloc[0]
        comm_df = df[df['commodity'] == commodity].copy()
    
    comm_df = comm_df.sort_values('date').reset_index(drop=True)
    
    # Sample if needed
    if len(comm_df) > n_bars:
        if seed is not None:
            np.random.seed(seed)
        start_idx = np.random.randint(0, len(comm_df) - n_bars)
        comm_df = comm_df.iloc[start_idx:start_idx + n_bars].reset_index(drop=True)
    
    # Extract state features
    state_df = pd.DataFrame({
        'price': comm_df['price'],
        'returns': comm_df['returns'],
        'price_ma5': comm_df['price_ma5'],
        'price_ma20': comm_df['price_ma20'],
        'vol_20': comm_df['vol_20'],
    })
    
    regimes = pd.Series(comm_df['regime'].values)
    
    # Create methods
    methods = {
        "Persistence": CommodityMethod("Persistence", ["sideways"]),
        "MA5": CommodityMethod("MA5", ["bull", "bear"]),
        "MA20": CommodityMethod("MA20", ["sideways", "bull"]),
        "Momentum": CommodityMethod("Momentum", ["bull", "volatile"]),
        "MeanReversion": CommodityMethod("MeanReversion", ["volatile", "bear"]),
    }
    
    return state_df, regimes, methods


class CommoditiesDomain:
    """Wrapper for commodities domain environment."""
    
    def __init__(self, n_bars: int = 2000, commodity: str = "WTI_Oil", seed: int = None):
        self.df, self.regimes, self.methods = create_commodities_environment(
            n_bars=n_bars, commodity=commodity, seed=seed
        )
    
    @property
    def regime_names(self):
        return COMMODITIES_REGIMES
    
    @property
    def method_names(self):
        return COMMODITIES_METHODS

