"""
Solar Domain - Solar irradiance (GHI) prediction with weather regimes.

Regimes: clear, partly_cloudy, overcast, storm
Methods: Persistence, MA6, ClearSkyModel, Seasonal
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .base import DomainEnvironment, DomainMethod


SOLAR_REGIMES = ["clear", "partly_cloudy", "overcast", "storm"]
SOLAR_METHODS = ["Persistence", "MA6", "ClearSkyModel", "Seasonal"]


class SolarMethod(DomainMethod):
    """Solar irradiance prediction method."""
    
    def __init__(self, name: str, optimal_regimes: List[str]):
        self.name = name
        self.optimal_regimes = optimal_regimes
        self._history = []
        self._hour_history = {}  # Track by hour for clear sky model
    
    def predict(self, current_ghi: float, history: np.ndarray, hour: int = 12, day_of_year: int = 180) -> float:
        """Predict next GHI value."""
        if self.name == "Persistence":
            return current_ghi
        elif self.name == "MA6":
            if len(history) >= 6:
                return np.mean(history[-6:])
            return current_ghi
        elif self.name == "ClearSkyModel":
            # Simplified clear sky model based on hour
            # Peak at noon, zero at night
            if 6 <= hour <= 18:
                # Solar elevation approximation
                solar_angle = np.sin(np.pi * (hour - 6) / 12)
                # Seasonal adjustment
                seasonal = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365)
                clear_sky_ghi = 1000 * solar_angle * seasonal
                return max(0, clear_sky_ghi)
            return 0
        elif self.name == "Seasonal":
            # Use same hour from yesterday
            if len(history) >= 24:
                return history[-24]
            return current_ghi
        return current_ghi
    
    def execute(self, observation: np.ndarray) -> Dict:
        """Execute method on observation."""
        ghi = observation[0] if len(observation) > 0 else 500
        hour = int(observation[1]) if len(observation) > 1 else 12
        doy = int(observation[2]) if len(observation) > 2 else 180
        
        prediction = self.predict(ghi, np.array(self._history), hour, doy)
        self._history.append(ghi)
        if len(self._history) > 48:
            self._history = self._history[-48:]
        
        # Signal based on prediction error
        if ghi > 0:
            error_pct = (prediction - ghi) / max(ghi, 100)
        else:
            error_pct = 0
        signal = np.clip(error_pct, -1, 1)
        return {"signal": signal, "prediction": prediction, "confidence": 0.5}


def load_solar_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load solar irradiance data from CSV."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "solar" / "nrel_irradiance.csv"
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def create_solar_environment(
    n_bars: int = 2000,
    location: str = "Phoenix_AZ",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, SolarMethod]]:
    """
    Create solar irradiance prediction environment from real data.
    
    State: [ghi, hour, day_of_year, ghi_ma6]
    """
    df = load_solar_data()
    
    # Filter by location
    loc_df = df[df['location'] == location].copy()
    if len(loc_df) == 0:
        location = df['location'].iloc[0]
        loc_df = df[df['location'] == location].copy()
    
    loc_df = loc_df.sort_values('datetime').reset_index(drop=True)
    
    # Sample if needed
    if len(loc_df) > n_bars:
        if seed is not None:
            np.random.seed(seed)
        start_idx = np.random.randint(0, len(loc_df) - n_bars)
        loc_df = loc_df.iloc[start_idx:start_idx + n_bars].reset_index(drop=True)
    
    # Extract state features
    state_df = pd.DataFrame({
        'ghi': loc_df['ghi'],
        'hour': loc_df['hour'],
        'day_of_year': loc_df['day_of_year'],
        'ghi_ma6': loc_df['ghi_ma6'],
    })
    
    regimes = pd.Series(loc_df['regime'].values)
    
    # Create methods
    methods = {
        "Persistence": SolarMethod("Persistence", ["clear"]),
        "MA6": SolarMethod("MA6", ["partly_cloudy"]),
        "ClearSkyModel": SolarMethod("ClearSkyModel", ["clear"]),
        "Seasonal": SolarMethod("Seasonal", ["clear", "partly_cloudy"]),
    }
    
    return state_df, regimes, methods


class SolarDomain:
    """Wrapper for solar domain environment."""
    
    def __init__(self, n_bars: int = 2000, location: str = "Phoenix_AZ", seed: int = None):
        self.df, self.regimes, self.methods = create_solar_environment(
            n_bars=n_bars, location=location, seed=seed
        )
    
    @property
    def regime_names(self):
        return SOLAR_REGIMES
    
    @property
    def method_names(self):
        return SOLAR_METHODS

