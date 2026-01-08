"""
Energy Forecast Service for FarmInsight

Provides battery SoC and solar production forecasting with proactive action generation.
Uses trained ML model for predictions with rule-based fallback.
Fetches live weather forecasts from Open-Meteo (free API, no key required).
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import requests


# Thresholds for proactive action generation
GRID_CONNECT_THRESHOLD = 15  # Connect grid when predicted SoC < 15%
SHUTDOWN_THRESHOLD = 10  # Shutdown consumers when predicted SoC < 10%
BUFFER_HOURS = 2  # Schedule actions this many hours before threshold is hit

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "trained_models"

# Location defaults (Clausthal-Zellerfeld)
DEFAULT_LATITUDE = 51.900994701794644
DEFAULT_LONGITUDE = 10.43181359767914


def _load_ml_model():
    """
    Try to load the trained ML model.
    
    :return: Tuple of (model, scaler, metadata) or (None, None, None) if not available
    """
    try:
        from ..utils.energy_model_trainer import load_energy_model
        return load_energy_model()
    except Exception as e:
        print(f"ML model not available: {e}")
        return None, None, None


def fetch_weather_forecast(latitude: float, longitude: float, forecast_hours: int = 336) -> dict:
    """
    Fetch weather forecast from Open-Meteo (free API).
    
    :param latitude: Location latitude
    :param longitude: Location longitude
    :param forecast_hours: Number of hours to forecast (max 16 days = 384 hours)
    :return: Weather data dictionary
    """
    forecast_days = min(16, max(2, (forecast_hours // 24) + 1))
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "sunshine_duration"
        ],
        "timezone": "Europe/Berlin",
        "forecast_days": forecast_days
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return {}


def _prepare_ml_features(
    hour: int,
    day_of_week: int,
    month: int,
    power_watts: float,
    shortwave_radiation: float,
    cloud_cover_pct: float,
    capacity_history: List[float]
) -> np.ndarray:
    """
    Prepare feature vector for ML model prediction.
    
    :return: Feature array matching training features
    """
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Historical features (use available history or pad with current value)
    hist = capacity_history if len(capacity_history) >= 24 else [capacity_history[-1]] * 24
    
    cap_lag_1h = hist[-1] if len(hist) >= 1 else 800
    cap_lag_3h = hist[-3] if len(hist) >= 3 else cap_lag_1h
    cap_lag_6h = hist[-6] if len(hist) >= 6 else cap_lag_1h
    cap_lag_12h = hist[-12] if len(hist) >= 12 else cap_lag_1h
    cap_lag_24h = hist[-24] if len(hist) >= 24 else cap_lag_1h
    
    cap_rolling_6h = np.mean(hist[-6:]) if len(hist) >= 6 else cap_lag_1h
    cap_rolling_24h = np.mean(hist[-24:]) if len(hist) >= 24 else cap_lag_1h
    
    # Power rolling averages (simplified: use current value)
    power_rolling_6h = power_watts
    power_rolling_24h = power_watts
    
    features = np.array([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        month_sin, month_cos,
        power_watts,
        shortwave_radiation, cloud_cover_pct,
        cap_lag_1h, cap_lag_3h, cap_lag_6h, cap_lag_12h, cap_lag_24h,
        cap_rolling_6h, cap_rolling_24h,
        power_rolling_6h, power_rolling_24h
    ]).reshape(1, -1)
    
    return features


def predict_with_ml_model(
    model,
    scaler,
    weather_data: dict,
    initial_soc_wh: float,
    avg_consumption_watts: float,
    battery_max_wh: float,
    forecast_hours: int = 336
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate battery SoC predictions using ML model.
    
    :return: Tuple of (expected, optimistic, pessimistic) prediction lists
    """
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    radiations = hourly.get("shortwave_radiation", [0] * forecast_hours)
    cloud_covers = hourly.get("cloud_cover", [50] * forecast_hours)
    
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    # Scenario multipliers
    scenarios = {
        "expected": {"factor": 1.0},
        "optimistic": {"factor": 1.15},  # 15% better
        "pessimistic": {"factor": 0.85}  # 15% worse
    }
    
    results = {"expected": [], "optimistic": [], "pessimistic": []}
    
    for scenario_name, config in scenarios.items():
        soc_wh = initial_soc_wh
        capacity_history = [initial_soc_wh]
        predictions = []
        
        for i in range(min(forecast_hours, len(radiations))):
            forecast_time = now + timedelta(hours=i)
            
            # Get weather for this hour
            radiation = radiations[i] if i < len(radiations) else 0
            cloud = cloud_covers[i] if i < len(cloud_covers) else 50
            
            # Prepare features
            features = _prepare_ml_features(
                hour=forecast_time.hour,
                day_of_week=forecast_time.weekday(),
                month=forecast_time.month,
                power_watts=avg_consumption_watts,
                shortwave_radiation=radiation,
                cloud_cover_pct=cloud,
                capacity_history=capacity_history[-24:]
            )
            
            # Scale and predict
            try:
                features_scaled = scaler.transform(features)
                predicted_soc = model.predict(features_scaled)[0]
                
                # Apply scenario factor
                if scenario_name == "optimistic":
                    predicted_soc = min(battery_max_wh, predicted_soc * config["factor"])
                elif scenario_name == "pessimistic":
                    predicted_soc = max(0, predicted_soc * config["factor"])
                
                soc_wh = max(0, min(battery_max_wh, predicted_soc))
            except Exception as e:
                # Fallback: simple calculation
                print(f"ML prediction error at hour {i}: {e}")
                solar_factor = (1 - cloud / 100) * (radiation / 1000) if radiation > 0 else 0
                solar_output = 600 * solar_factor * (0.5 if 6 <= forecast_time.hour <= 19 else 0)
                net_power = solar_output - avg_consumption_watts
                soc_wh = max(0, min(battery_max_wh, soc_wh + net_power))
            
            capacity_history.append(soc_wh)
            predictions.append({
                "timestamp": forecast_time.isoformat() + "Z",
                "value": round(soc_wh, 0)
            })
        
        results[scenario_name] = predictions
    
    return results["expected"], results["optimistic"], results["pessimistic"]


def predict_solar_production(
    weather_data: dict,
    max_solar_output_watts: float,
    hour_offset: int = 0,
    hours: int = 336
) -> List[Dict[str, Any]]:
    """
    Predict solar energy production based on weather data.
    
    :param weather_data: Weather forecast from Open-Meteo
    :param max_solar_output_watts: Maximum solar panel output in watts
    :param hour_offset: Starting hour offset in the weather data
    :param hours: Number of hours to predict
    :return: List of hourly predictions
    """
    hourly = weather_data.get("hourly", {})
    cloud_covers = hourly.get("cloud_cover", [50] * hours)
    radiations = hourly.get("shortwave_radiation", [400] * hours)
    
    predictions = []
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    for i in range(hours):
        forecast_time = now + timedelta(hours=i)
        weather_idx = min(hour_offset + i, len(cloud_covers) - 1)
        
        # Solar output factor based on radiation and cloud cover
        radiation = radiations[weather_idx] if weather_idx < len(radiations) else 400
        cloud = cloud_covers[weather_idx] if weather_idx < len(cloud_covers) else 50
        
        # Normalize radiation (typical max ~1000 W/m²) and apply cloud factor
        radiation_factor = min(1.0, radiation / 1000.0)
        cloud_factor = 1.0 - (cloud / 100.0 * 0.8)  # Clouds reduce output by up to 80%
        
        # Time of day factor (solar panel efficiency varies with sun angle)
        hour = forecast_time.hour
        if 6 <= hour <= 19:
            time_factor = 0.5 + 0.5 * (1 - abs(hour - 12.5) / 6.5)
        else:
            time_factor = 0.0  # No solar at night
        
        output = max_solar_output_watts * radiation_factor * cloud_factor * time_factor
        
        predictions.append({
            "timestamp": forecast_time.isoformat() + "Z",
            "value": round(output, 1)
        })
    
    return predictions


def predict_battery_soc_rule_based(
    solar_predictions: List[Dict],
    consumption_watts: float,
    initial_soc_wh: float,
    battery_max_wh: float,
    hours: int = 336
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Predict battery state of charge over time using rule-based approach.
    Returns three scenarios: expected, optimistic, pessimistic.
    
    :param solar_predictions: Hourly solar production predictions
    :param consumption_watts: Average consumption in watts
    :param initial_soc_wh: Initial battery state of charge in Wh
    :param battery_max_wh: Maximum battery capacity in Wh
    :param hours: Number of hours to predict
    :return: Tuple of (expected, optimistic, pessimistic) prediction lists
    """
    expected = []
    optimistic = []
    pessimistic = []
    
    # Scenario multipliers
    scenarios = {
        "expected": {"solar": 1.0, "consumption": 1.0},
        "optimistic": {"solar": 1.25, "consumption": 0.85},
        "pessimistic": {"solar": 0.6, "consumption": 1.2}
    }
    
    for scenario_name, factors in scenarios.items():
        soc_wh = initial_soc_wh
        predictions = []
        
        for i, solar_pred in enumerate(solar_predictions[:hours]):
            # Net power = production - consumption
            solar_output = solar_pred["value"] * factors["solar"]
            consumption = consumption_watts * factors["consumption"]
            net_power = solar_output - consumption
            
            # Update SoC (assume 1-hour intervals)
            soc_wh = soc_wh + net_power
            soc_wh = max(0, min(battery_max_wh, soc_wh))
            
            predictions.append({
                "timestamp": solar_pred["timestamp"],
                "value": round(soc_wh, 0)
            })
        
        if scenario_name == "expected":
            expected = predictions
        elif scenario_name == "optimistic":
            optimistic = predictions
        else:
            pessimistic = predictions
    
    return expected, optimistic, pessimistic


def generate_proactive_actions(
    soc_predictions: List[Dict[str, Any]],
    battery_max_wh: float
) -> List[Dict[str, Any]]:
    """
    Generate proactive action schedule based on SoC predictions.
    
    :param soc_predictions: Battery SoC predictions (expected scenario)
    :param battery_max_wh: Maximum battery capacity in Wh
    :return: List of scheduled actions (only connect_grid)
    """
    actions = []
    grid_connected = False
    
    for pred in soc_predictions:
        soc_wh = pred["value"]
        soc_percent = (soc_wh / battery_max_wh) * 100
        timestamp = pred["timestamp"]
        
        # Calculate buffer timestamp (schedule action BUFFER_HOURS before)
        try:
            pred_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            action_time = pred_time - timedelta(hours=BUFFER_HOURS)
            action_timestamp = action_time.isoformat().replace("+00:00", "Z")
        except:
            action_timestamp = timestamp
        
        # Check if we need to connect grid
        if soc_percent < GRID_CONNECT_THRESHOLD and not grid_connected:
            actions.append({
                "timestamp": action_timestamp,
                "action": "connect_grid",
                "value": True
            })
            grid_connected = True
        
        # Check if we can disconnect grid (battery recovered)
        if soc_percent > 50 and grid_connected:
            actions.append({
                "timestamp": action_timestamp,
                "action": "connect_grid",
                "value": False
            })
            grid_connected = False
    
    return actions


def energy_forecast(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    forecast_hours: int = 336,
    max_solar_output_watts: float = 600,
    avg_consumption_watts: float = 50,
    initial_soc_wh: float = 800,
    battery_max_wh: float = 1600
) -> Dict[str, Any]:
    """
    Generate complete energy forecast with proactive actions.
    
    Uses ML model if available, falls back to rule-based approach.
    
    :return: Dictionary with forecasts and actions
    """
    # Fetch weather data
    weather_data = fetch_weather_forecast(latitude, longitude, forecast_hours)
    weather_source = "live" if weather_data else "fallback"
    
    # Try to use ML model
    model, scaler, metadata = _load_ml_model()
    use_ml = model is not None and scaler is not None
    
    if use_ml:
        print("Using ML model for predictions")
        expected, optimistic, pessimistic = predict_with_ml_model(
            model, scaler, weather_data,
            initial_soc_wh, avg_consumption_watts, battery_max_wh, forecast_hours
        )
    else:
        print("Using rule-based predictions (ML model not available)")
        # Predict solar production first
        solar_predictions = predict_solar_production(
            weather_data, max_solar_output_watts, hours=forecast_hours
        )
        # Then predict battery SoC
        expected, optimistic, pessimistic = predict_battery_soc_rule_based(
            solar_predictions, avg_consumption_watts, initial_soc_wh, 
            battery_max_wh, hours=forecast_hours
        )
    
    # Always calculate solar predictions for output
    solar_predictions = predict_solar_production(
        weather_data, max_solar_output_watts, hours=forecast_hours
    )
    
    # Generate proactive actions based on expected scenario
    actions = generate_proactive_actions(expected, battery_max_wh)
    
    return {
        "forecasts": [
            {
                "name": "Battery State of Charge",
                "values": [
                    {"name": "expected", "value": expected},
                    {"name": "optimistic", "value": optimistic},
                    {"name": "pessimistic", "value": pessimistic}
                ]
            },
            {
                "name": "Solar Energy Production",
                "values": [
                    {"name": "expected", "value": solar_predictions},
                    {"name": "optimistic", "value": [
                        {"timestamp": p["timestamp"], "value": round(p["value"] * 1.25, 1)}
                        for p in solar_predictions
                    ]},
                    {"name": "pessimistic", "value": [
                        {"timestamp": p["timestamp"], "value": round(p["value"] * 0.6, 1)}
                        for p in solar_predictions
                    ]}
                ]
            }
        ],
        "actions": actions,
        "weather_source": weather_source,
        "prediction_method": "ml_model" if use_ml else "rule_based",
        "forecast_hours": forecast_hours
    }
