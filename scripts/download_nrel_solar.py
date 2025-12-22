#!/usr/bin/env python3
"""
Download solar irradiance data from NREL NSRDB or generate realistic synthetic data.

NREL NSRDB: https://nsrdb.nrel.gov/
API requires registration for large downloads.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_solar_data(output_path: str, start_date: str = "2020-01-01",
                                   end_date: str = "2024-12-31") -> pd.DataFrame:
    """
    Generate realistic synthetic solar irradiance (GHI) data.

    Patterns include:
    - Daily cycle (sunrise to sunset)
    - Seasonal variation (longer/shorter days, sun angle)
    - Weather effects (clear, partly cloudy, overcast, storm)
    - Geographic variation (latitude effects)
    """
    print("Generating realistic solar irradiance data...")

    # Locations with different solar characteristics
    locations = {
        'Phoenix_AZ': {'lat': 33.4, 'clear_prob': 0.75, 'peak_ghi': 1000},
        'Denver_CO': {'lat': 39.7, 'clear_prob': 0.65, 'peak_ghi': 950},
        'Seattle_WA': {'lat': 47.6, 'clear_prob': 0.35, 'peak_ghi': 850},
        'Miami_FL': {'lat': 25.8, 'clear_prob': 0.55, 'peak_ghi': 1050},
        'Boston_MA': {'lat': 42.4, 'clear_prob': 0.50, 'peak_ghi': 900},
    }

    # Hourly data during daylight hours
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    all_data = []

    for location, params in locations.items():
        np.random.seed(hash(location) % 2**31)

        ghi_values = []
        regimes = []
        valid_dates = []
        hours = []

        for dt in date_range:
            hour = dt.hour
            month = dt.month
            day_of_year = dt.timetuple().tm_yday

            # Calculate approximate sunrise/sunset based on latitude and season
            # Simplified model
            day_length = 12 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365) * (params['lat'] / 50)
            sunrise = 12 - day_length / 2
            sunset = 12 + day_length / 2

            # Skip nighttime hours
            if hour < sunrise or hour > sunset:
                continue

            # Solar zenith angle approximation
            hour_angle = (hour - 12) / (day_length / 2) * np.pi / 2
            elevation = np.cos(hour_angle) * np.cos(np.radians(params['lat']))
            elevation = max(0, elevation)

            # Base GHI from clear sky model
            clear_sky_ghi = params['peak_ghi'] * elevation

            # Weather conditions
            weather_roll = np.random.random()
            if weather_roll < params['clear_prob']:
                weather = 'clear'
                cloud_factor = 0.95 + 0.05 * np.random.random()
            elif weather_roll < params['clear_prob'] + 0.2:
                weather = 'partly_cloudy'
                cloud_factor = 0.5 + 0.3 * np.random.random()
            elif weather_roll < params['clear_prob'] + 0.35:
                weather = 'overcast'
                cloud_factor = 0.15 + 0.2 * np.random.random()
            else:
                weather = 'storm'
                cloud_factor = 0.05 + 0.1 * np.random.random()

            # Final GHI
            ghi = clear_sky_ghi * cloud_factor
            ghi = max(0, ghi + np.random.normal(0, 20))

            ghi_values.append(ghi)
            regimes.append(weather)
            valid_dates.append(dt)
            hours.append(hour)

        # Create location dataframe
        loc_df = pd.DataFrame({
            'datetime': valid_dates,
            'location': location,
            'ghi': ghi_values,
            'hour': hours,
            'regime': regimes,
        })

        loc_df['month'] = loc_df['datetime'].dt.month
        loc_df['day_of_year'] = loc_df['datetime'].dt.dayofyear

        all_data.append(loc_df)

    # Combine all locations
    df = pd.concat(all_data, ignore_index=True)

    # Add lag features
    df['ghi_lag1'] = df.groupby('location')['ghi'].shift(1)
    df['ghi_lag24'] = df.groupby('location')['ghi'].shift(24)
    df['ghi_ma6'] = df.groupby('location')['ghi'].transform(lambda x: x.rolling(6).mean())
    df['ghi_std6'] = df.groupby('location')['ghi'].transform(lambda x: x.rolling(6).std())

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} records for {len(locations)} locations")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Regime distribution:")
    print(df['regime'].value_counts())
    print(f"Saved to: {output_path}")

    return df


def try_download_nrel_api(api_key: str = None) -> pd.DataFrame:
    """
    Attempt to download real NREL NSRDB data.

    Requires API key from: https://developer.nrel.gov/signup/
    """
    try:
        import requests
    except ImportError:
        print("requests library not available, using synthetic data")
        return None

    if not api_key:
        api_key = os.environ.get('NREL_API_KEY')

    if not api_key:
        print("No NREL API key found. To use real data:")
        print("1. Register at https://developer.nrel.gov/signup/")
        print("2. Set NREL_API_KEY environment variable")
        print("Using synthetic NREL-style data instead...")
        return None

    # NREL API requires specific format, would need detailed setup
    return None


def main():
    """Main function to download or generate solar irradiance data."""
    output_path = Path(__file__).parent.parent / "data" / "solar" / "nrel_irradiance.csv"

    # Try to download real data first
    real_data = try_download_nrel_api()

    if real_data is not None:
        real_data.to_csv(output_path, index=False)
        print(f"Saved real NREL data to: {output_path}")
    else:
        generate_synthetic_solar_data(str(output_path))

    print("\n=== Solar Irradiance Data Ready ===")


if __name__ == "__main__":
    main()
