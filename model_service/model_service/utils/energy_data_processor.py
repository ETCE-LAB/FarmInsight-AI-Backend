"""
Energy Data Processor for FarmInsight

Processes JSON training data (capacity + power consumption) and fetches 
historical weather data from Open-Meteo Archive API (no API key required).
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

# Location: Clausthal-Zellerfeld area
DEFAULT_LATITUDE = 51.900994701794644
DEFAULT_LONGITUDE = 10.43181359767914
DEFAULT_ELEVATION = 289

# Training data paths
TRAINING_DATA_DIR = Path(__file__).parent.parent.parent.parent.parent / "Model Training Grounds"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_capacity_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load battery capacity data from JSON file.
    
    :param filepath: Path to JSON file (default: Anker Daten ALL Kapazität in Wh.json)
    :return: DataFrame with timestamp and capacity_wh columns
    """
    if filepath is None:
        filepath = TRAINING_DATA_DIR / "Anker Daten ALL Kapazität in Wh.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['measuredAt'], utc=True).dt.tz_localize(None)
    df['capacity_wh'] = df['value'].astype(float)
    df = df[['timestamp', 'capacity_wh']].sort_values('timestamp')
    df = df.set_index('timestamp')
    
    return df


def load_power_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load power consumption data from JSON file.
    
    :param filepath: Path to JSON file (default: Power Daten ALL Vebrauch in Watt.json)
    :return: DataFrame with timestamp and power_watts columns
    """
    if filepath is None:
        filepath = TRAINING_DATA_DIR / "Power Daten ALL Vebrauch in Watt.json"
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['measuredAt'], utc=True).dt.tz_localize(None)
    df['power_watts'] = df['value'].astype(float)
    df = df[['timestamp', 'power_watts']].sort_values('timestamp')
    df = df.set_index('timestamp')
    
    return df


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo Archive API (free, no API key).
    
    :param start_date: Start date in YYYY-MM-DD format
    :param end_date: End date in YYYY-MM-DD format
    :param latitude: Location latitude
    :param longitude: Location longitude
    :return: DataFrame with hourly weather data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "sunshine_duration",
            "precipitation"
        ],
        "timezone": "Europe/Berlin"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(times),
            "temperature_c": hourly.get("temperature_2m", []),
            "humidity_pct": hourly.get("relative_humidity_2m", []),
            "cloud_cover_pct": hourly.get("cloud_cover", []),
            "shortwave_radiation": hourly.get("shortwave_radiation", []),
            "direct_radiation": hourly.get("direct_radiation", []),
            "diffuse_radiation": hourly.get("diffuse_radiation", []),
            "sunshine_duration_s": hourly.get("sunshine_duration", []),
            "precipitation_mm": hourly.get("precipitation", [])
        })
        
        df = df.set_index("timestamp")
        return df
        
    except Exception as e:
        print(f"Error fetching historical weather: {e}")
        return pd.DataFrame()


def resample_to_hourly(df: pd.DataFrame, value_col: str, agg_func: str = 'mean') -> pd.DataFrame:
    """
    Resample time series data to hourly intervals.
    
    :param df: DataFrame with datetime index
    :param value_col: Column name to aggregate
    :param agg_func: Aggregation function ('mean', 'sum', 'last', etc.)
    :return: Hourly resampled DataFrame
    """
    return df.resample('h').agg({value_col: agg_func}).dropna()


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical temporal features to DataFrame.
    
    :param df: DataFrame with datetime index
    :return: DataFrame with added temporal features
    """
    df = df.copy()
    
    # Extract time components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    
    # Cyclical encoding for hour (0-23 cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week (0-6 cycle)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Cyclical encoding for month (1-12 cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def add_lagged_features(df: pd.DataFrame, value_col: str, lags: List[int] = None) -> pd.DataFrame:
    """
    Add lagged versions of a column as features.
    
    :param df: DataFrame
    :param value_col: Column to create lags from
    :param lags: List of lag values (hours). Default: [1, 2, 3, 6, 12, 24]
    :return: DataFrame with lagged features
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    
    df = df.copy()
    for lag in lags:
        df[f'{value_col}_lag_{lag}h'] = df[value_col].shift(lag)
    
    return df


def add_rolling_features(df: pd.DataFrame, value_col: str, windows: List[int] = None) -> pd.DataFrame:
    """
    Add rolling statistics as features.
    
    :param df: DataFrame
    :param value_col: Column to calculate rolling stats from
    :param windows: List of window sizes (hours). Default: [6, 12, 24]
    :return: DataFrame with rolling features
    """
    if windows is None:
        windows = [6, 12, 24]
    
    df = df.copy()
    for window in windows:
        df[f'{value_col}_rolling_mean_{window}h'] = df[value_col].rolling(window).mean()
        df[f'{value_col}_rolling_std_{window}h'] = df[value_col].rolling(window).std()
    
    return df


def prepare_energy_training_data() -> Tuple[pd.DataFrame, str]:
    """
    Prepare complete training dataset for energy model.
    
    :return: Tuple of (training DataFrame, output file path)
    """
    print("Loading capacity data...")
    capacity_df = load_capacity_data()
    print(f"  Loaded {len(capacity_df)} capacity records")
    
    print("Loading power data...")
    power_df = load_power_data()
    print(f"  Loaded {len(power_df)} power records")
    
    # Resample to hourly
    print("Resampling to hourly intervals...")
    capacity_hourly = resample_to_hourly(capacity_df, 'capacity_wh', 'last')
    power_hourly = resample_to_hourly(power_df, 'power_watts', 'mean')
    
    # Merge capacity and power data
    print("Merging energy data...")
    energy_df = capacity_hourly.join(power_hourly, how='outer')
    energy_df = energy_df.interpolate(method='time', limit=6)  # Fill small gaps
    
    # Get date range for weather data
    start_date = energy_df.index.min().strftime('%Y-%m-%d')
    end_date = energy_df.index.max().strftime('%Y-%m-%d')
    
    print(f"Fetching historical weather data ({start_date} to {end_date})...")
    weather_df = fetch_historical_weather(start_date, end_date)
    
    if weather_df.empty:
        print("  Warning: Could not fetch weather data, using synthetic features")
        # Create synthetic solar radiation pattern if weather unavailable
        energy_df['shortwave_radiation'] = 0
        energy_df['cloud_cover_pct'] = 50
        for idx in energy_df.index:
            hour = idx.hour
            if 6 <= hour <= 19:
                # Approximate solar pattern
                energy_df.loc[idx, 'shortwave_radiation'] = 500 * (1 - abs(hour - 12.5) / 6.5)
    else:
        print(f"  Fetched {len(weather_df)} weather records")
        # Merge weather data
        energy_df = energy_df.join(weather_df, how='left')
    
    # Add temporal features
    print("Adding temporal features...")
    energy_df = add_temporal_features(energy_df)
    
    # Add lagged capacity features
    print("Adding lagged and rolling features...")
    energy_df = add_lagged_features(energy_df, 'capacity_wh')
    energy_df = add_rolling_features(energy_df, 'capacity_wh')
    energy_df = add_rolling_features(energy_df, 'power_watts')
    
    # Create target variable: capacity change per hour
    energy_df['capacity_change'] = energy_df['capacity_wh'].diff()
    energy_df['capacity_next_hour'] = energy_df['capacity_wh'].shift(-1)
    
    # Drop rows with NaN (from lagging and rolling)
    energy_df = energy_df.dropna()
    
    # Save to CSV
    output_path = DATA_DIR / "energy_training_data.csv"
    os.makedirs(DATA_DIR, exist_ok=True)
    energy_df.to_csv(output_path)
    
    print(f"Training data saved to {output_path}")
    print(f"  Total records: {len(energy_df)}")
    print(f"  Features: {len(energy_df.columns)}")
    print(f"  Date range: {energy_df.index.min()} to {energy_df.index.max()}")
    
    return energy_df, str(output_path)


if __name__ == "__main__":
    df, path = prepare_energy_training_data()
    print(f"\nTraining data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
