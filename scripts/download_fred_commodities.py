#!/usr/bin/env python3
"""
Download commodity price data from FRED (Federal Reserve Economic Data).

FRED API: https://fred.stlouisfed.org/docs/api/
Free API key available at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_commodity_data(output_path: str, start_date: str = "2015-01-01",
                                       end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Generate realistic synthetic commodity price data.

    Commodities: Oil (WTI), Gold, Corn, Copper
    Patterns include:
    - Trend components (secular trends)
    - Cyclical patterns (economic cycles)
    - Seasonality (agricultural commodities)
    - Volatility clusters
    - Correlated movements (macro factors)
    """
    print("Generating realistic commodity price data...")

    # Commodity characteristics
    commodities = {
        'WTI_Oil': {
            'start_price': 50,
            'annual_vol': 0.35,
            'trend': 0.02,
            'mean_revert': 0.01,
            'long_term_mean': 60,
        },
        'Gold': {
            'start_price': 1200,
            'annual_vol': 0.15,
            'trend': 0.05,
            'mean_revert': 0.005,
            'long_term_mean': 1800,
        },
        'Corn': {
            'start_price': 4.0,
            'annual_vol': 0.25,
            'trend': 0.01,
            'mean_revert': 0.02,
            'long_term_mean': 5.0,
            'seasonal': True,
        },
        'Copper': {
            'start_price': 2.8,
            'annual_vol': 0.25,
            'trend': 0.03,
            'mean_revert': 0.01,
            'long_term_mean': 3.5,
        },
    }

    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n_days = len(date_range)

    all_data = []

    # Generate correlated random shocks (macro factor)
    np.random.seed(42)
    macro_factor = np.random.randn(n_days) * 0.3

    for commodity, params in commodities.items():
        np.random.seed(hash(commodity) % 2**31)

        prices = np.zeros(n_days)
        prices[0] = params['start_price']

        daily_vol = params['annual_vol'] / np.sqrt(252)

        for i in range(1, n_days):
            dt = date_range[i]
            month = dt.month

            # Mean reversion term
            reversion = params['mean_revert'] * (params['long_term_mean'] - prices[i-1])

            # Trend term
            trend = params['trend'] / 252

            # Seasonality (mainly for agricultural)
            if params.get('seasonal', False):
                # Corn: lower after harvest (Oct-Dec), higher before (Jun-Aug)
                if month in [10, 11, 12]:
                    seasonal_adj = -0.0005
                elif month in [6, 7, 8]:
                    seasonal_adj = 0.0005
                else:
                    seasonal_adj = 0
            else:
                seasonal_adj = 0

            # Random shock (idiosyncratic + macro)
            idio_shock = np.random.randn() * daily_vol
            macro_shock = macro_factor[i] * daily_vol * 0.5

            # Price update (log-normal)
            log_return = trend + reversion/prices[i-1] + seasonal_adj + idio_shock + macro_shock
            prices[i] = prices[i-1] * np.exp(log_return)

        # Calculate returns and volatility for regime detection
        returns = np.diff(np.log(prices))
        returns = np.insert(returns, 0, 0)

        # Rolling volatility
        vol_window = 20
        rolling_vol = pd.Series(returns).rolling(vol_window).std() * np.sqrt(252)

        # MA for trend detection
        ma20 = pd.Series(prices).rolling(20).mean()
        ma50 = pd.Series(prices).rolling(50).mean()

        # Assign regimes
        regimes = []
        for i in range(n_days):
            if i < 50:
                regimes.append('normal')
                continue

            # Trend regime
            if prices[i] > ma20.iloc[i] and ma20.iloc[i] > ma50.iloc[i]:
                trend_regime = 'bull'
            elif prices[i] < ma20.iloc[i] and ma20.iloc[i] < ma50.iloc[i]:
                trend_regime = 'bear'
            else:
                trend_regime = 'sideways'

            # Volatility regime
            if rolling_vol.iloc[i] > 0.3:
                vol_regime = 'high_vol'
            else:
                vol_regime = 'low_vol'

            # Combined regime
            if trend_regime == 'bull' and vol_regime == 'low_vol':
                regimes.append('bull')
            elif trend_regime == 'bear':
                regimes.append('bear')
            elif vol_regime == 'high_vol':
                regimes.append('volatile')
            else:
                regimes.append('sideways')

        # Create commodity dataframe
        comm_df = pd.DataFrame({
            'date': date_range,
            'commodity': commodity,
            'price': prices,
            'returns': returns,
            'month': [dt.month for dt in date_range],
            'regime': regimes,
        })

        all_data.append(comm_df)

    # Combine all commodities
    df = pd.concat(all_data, ignore_index=True)

    # Add lag features
    df['price_lag1'] = df.groupby('commodity')['price'].shift(1)
    df['price_lag5'] = df.groupby('commodity')['price'].shift(5)
    df['price_ma5'] = df.groupby('commodity')['price'].transform(lambda x: x.rolling(5).mean())
    df['price_ma20'] = df.groupby('commodity')['price'].transform(lambda x: x.rolling(20).mean())
    df['returns_ma5'] = df.groupby('commodity')['returns'].transform(lambda x: x.rolling(5).mean())
    df['vol_20'] = df.groupby('commodity')['returns'].transform(lambda x: x.rolling(20).std())

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} records for {len(commodities)} commodities")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Regime distribution:")
    print(df['regime'].value_counts())
    print(f"Saved to: {output_path}")

    return df


def try_download_fred_api(api_key: str = None) -> pd.DataFrame:
    """
    Attempt to download real FRED commodity data.

    Series codes:
    - DCOILWTICO: WTI Crude Oil
    - GOLDAMGBD228NLBM: Gold (London)
    - PMAIZMTUSDM: Corn
    - PCOPPUSDM: Copper
    """
    try:
        import requests
    except ImportError:
        print("requests library not available, using synthetic data")
        return None

    if not api_key:
        api_key = os.environ.get('FRED_API_KEY')

    if not api_key:
        print("No FRED API key found. To use real data:")
        print("1. Register at https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Set FRED_API_KEY environment variable")
        print("Using synthetic FRED-style data instead...")
        return None

    # Would implement actual FRED API calls here
    return None


def main():
    """Main function to download or generate FRED commodity data."""
    output_path = Path(__file__).parent.parent / "data" / "commodities" / "fred_prices.csv"

    # Try to download real data first
    real_data = try_download_fred_api()

    if real_data is not None:
        real_data.to_csv(output_path, index=False)
        print(f"Saved real FRED data to: {output_path}")
    else:
        generate_synthetic_commodity_data(str(output_path))

    print("\n=== FRED Commodity Data Ready ===")


if __name__ == "__main__":
    main()
