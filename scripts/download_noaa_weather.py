#!/usr/bin/env python3
"""
Download NOAA Weather Data for Multi-Domain Validation.

Uses NOAA Climate Data Online (CDO) API or generates realistic synthetic data.
Data: Daily temperature, precipitation, wind for 3+ years.

Regime Labels (pattern-based):
- stable_warm: Low variance, high temp
- stable_cold: Low variance, low temp
- volatile_storm: High variance (storms)
- transition: Trend periods (season changes)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_realistic_weather_data(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    location: str = "NYC"
) -> pd.DataFrame:
    """
    Generate realistic synthetic weather data with clear regime structure.

    This creates weather data with patterns that have distinct optimal prediction strategies:
    - stable_warm: Low variance, predictable (use persistence)
    - stable_cold: Low variance, predictable (use persistence)
    - volatile_storm: High variance (use volatility-based methods)
    - transition: Trending (use trend-following)
    """
    print(f"Generating realistic weather data for {location}...")

    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)

    # Initialize arrays
    temps = np.zeros(n_days)
    precip = np.zeros(n_days)
    wind = np.zeros(n_days)
    regimes = []

    # Base temperature cycle (seasonal)
    day_of_year = np.array([d.timetuple().tm_yday for d in date_range])
    base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer

    for i in range(n_days):
        month = date_range[i].month

        # Determine regime based on season and random weather events
        if np.random.random() < 0.05:  # 5% chance of storm
            regime = 'volatile_storm'
            temp_noise = np.random.normal(0, 8)
            precip_val = np.random.exponential(20)
            wind_val = np.random.uniform(30, 60)
        elif month in [3, 4, 10, 11]:  # Transition months
            regime = 'transition'
            temp_noise = np.random.normal(0, 4)
            precip_val = np.random.exponential(5)
            wind_val = np.random.uniform(10, 25)
        elif month in [6, 7, 8]:  # Summer - stable warm
            regime = 'stable_warm'
            temp_noise = np.random.normal(0, 2)
            precip_val = np.random.exponential(3) if np.random.random() < 0.2 else 0
            wind_val = np.random.uniform(5, 15)
        else:  # Winter - stable cold
            regime = 'stable_cold'
            temp_noise = np.random.normal(0, 3)
            precip_val = np.random.exponential(4) if np.random.random() < 0.3 else 0
            wind_val = np.random.uniform(8, 20)

        temps[i] = base_temp[i] + temp_noise
        precip[i] = precip_val
        wind[i] = wind_val
        regimes.append(regime)

    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'temperature': temps,
        'precipitation': precip,
        'wind_speed': wind,
        'regime': regimes,
        'month': [d.month for d in date_range],
        'day_of_year': day_of_year
    })

    # Add features for prediction
    df['temp_lag1'] = df['temperature'].shift(1)
    df['temp_lag7'] = df['temperature'].shift(7)
    df['temp_ma7'] = df['temperature'].rolling(7).mean()
    df['temp_std7'] = df['temperature'].rolling(7).std()
    df['temp_trend7'] = df['temperature'].diff(7) / 7

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


def try_download_noaa_api(token: str = None) -> pd.DataFrame:
    """
    Attempt to download real NOAA data via API.

    Note: NOAA CDO API requires registration for a token.
    https://www.ncdc.noaa.gov/cdo-web/token
    """
    try:
        import requests
    except ImportError:
        print("requests library not available")
        return None

    if not token:
        token = os.environ.get('NOAA_TOKEN')

    if not token:
        print("No NOAA API token found. To use real data:")
        print("1. Register at https://www.ncdc.noaa.gov/cdo-web/token")
        print("2. Set NOAA_TOKEN environment variable")
        print("Using synthetic weather data instead...")
        return None

    # NOAA CDO API endpoint
    base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

    headers = {"token": token}
    params = {
        "datasetid": "GHCND",  # Global Historical Climatology Network Daily
        "stationid": "GHCND:USW00094728",  # Central Park, NYC
        "startdate": "2020-01-01",
        "enddate": "2023-12-31",
        "datatypeid": ["TMAX", "TMIN", "PRCP", "AWND"],
        "units": "metric",
        "limit": 1000
    }

    try:
        print("Attempting to download from NOAA API...")
        response = requests.get(base_url, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if 'results' in data:
            df = pd.DataFrame(data['results'])
            print(f"Downloaded {len(df)} records from NOAA API")
            return df
        else:
            print("Unexpected API response format")
            return None

    except Exception as e:
        print(f"NOAA API error: {e}")
        return None


def main():
    """Main function to download or generate weather data."""
    output_dir = Path(__file__).parent.parent / "data" / "weather" / "noaa"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "daily_weather.csv"

    # Try to download real data first
    real_data = try_download_noaa_api()

    if real_data is not None:
        # Process real data
        print("Processing real NOAA data...")
        real_data.to_csv(output_path, index=False)
        print(f"Saved real NOAA data to: {output_path}")
    else:
        # Generate synthetic data
        df = generate_realistic_weather_data()
        df.to_csv(output_path, index=False)

        print(f"\nGenerated {len(df)} days of weather data")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nRegime distribution:")
        print(df['regime'].value_counts())
        print(f"\nSaved to: {output_path}")

    print("\n=== Weather Data Ready ===")
    return output_path


if __name__ == "__main__":
    main()
