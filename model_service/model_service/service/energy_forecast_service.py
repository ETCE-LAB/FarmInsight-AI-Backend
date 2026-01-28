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

# Import site-specific configuration
from ..utils.site_config import (
    SITE_SHADING_FACTOR,
    EFFECTIVE_SOLAR_START_HOUR,
    EFFECTIVE_SOLAR_END_HOUR,
    BATTERY_MIN_WH,
    BATTERY_MAX_WH,
    TYPICAL_CONSUMPTION_WATTS,
    MAX_SOLAR_OUTPUT_WATTS,
    SCENARIO_CONSUMPTION_FACTORS,
    SCENARIO_SOLAR_FACTORS,
    get_seasonal_factor,
    get_effective_solar_fraction,
)


# Thresholds for proactive action generation
GRID_CONNECT_THRESHOLD = 15  # Connect grid when predicted SoC < 15%
SHUTDOWN_THRESHOLD = 10  # Shutdown consumers when predicted SoC < 10%
BUFFER_HOURS = 2  # Schedule actions this many hours before threshold is hit

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "trained_models"

# Location defaults (Clausthal-Zellerfeld)
DEFAULT_LATITUDE = 51.900994701794644
DEFAULT_LONGITUDE = 10.43181359767914

# Charging rates from historical data (Wh gained per hour during solar window)
# These are GROSS charging rates (before consumption)
# Measured from actual hourly capacity increases
CHARGING_RATES_BY_MONTH = {
    1: 30,    # January - estimated from trend
    2: 40,    # February
    3: 50,    # March
    4: 60,    # April
    5: 70,    # May
    6: 80,    # June
    7: 56,    # July - measured 55.9 Wh/hour
    8: 91,    # August - measured 91.0 Wh/hour
    9: 96,    # September - measured 95.5 Wh/hour
    10: 61,   # October - measured 60.7 Wh/hour
    11: 24,   # November - measured 24.0 Wh/hour
    12: 20,   # December - estimated
}

# Consumption rates from historical data (Wh lost per hour)
# Different rates for different times of day
CONSUMPTION_NIGHT = 5     # 00-07: ~5 Wh/hour (low activity)
CONSUMPTION_DAY = 20      # 13-23: ~20 Wh/hour (equipment running)


def _simple_battery_forecast(
    weather_data: dict,
    initial_soc_wh: float,
    forecast_hours: int,
    consumption_wh_per_hour: float,  # Not used - we use time-based consumption
    battery_max_wh: float
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Generate battery SoC forecast using physics-based model calibrated from historical data.
    
    Historical analysis shows:
    - Charging window: 9:00-12:00 local time (peak at 10-11)
    - Night consumption: ~5 Wh/hour (00:00-08:00)
    - Day consumption: ~20 Wh/hour (13:00-23:00)
    - Most days run at minimum (160 Wh) with peaks during charging
    
    Returns:
        Tuple of (expected, optimistic, pessimistic, solar_production)
    """
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    cloud_covers = hourly.get("cloud_cover", [])
    
    # Scenario multipliers
    scenarios = {
        "expected": {"solar_mult": 1.0, "consumption_mult": 1.0},
        "optimistic": {"solar_mult": 1.5, "consumption_mult": 0.7},
        "pessimistic": {"solar_mult": 0.4, "consumption_mult": 1.4},
    }
    
    results = {}
    solar_predictions = []
    
    for scenario_name, mults in scenarios.items():
        soc = initial_soc_wh
        predictions = []
        
        for i in range(forecast_hours):
            # Get LOCAL time from weather API (already in Europe/Berlin)
            if i < len(times):
                try:
                    local_time = datetime.fromisoformat(times[i])
                except:
                    local_time = datetime.now() + timedelta(hours=i)
            else:
                local_time = datetime.now() + timedelta(hours=i)
            
            local_hour = local_time.hour
            month = local_time.month
            
            # Get base charging rate for this month (from historical data)
            base_charge_rate = CHARGING_RATES_BY_MONTH.get(month, 50)
            
            # Adjust for weather (cloud cover reduces charging)
            if i < len(cloud_covers):
                cloud = cloud_covers[i]
                weather_factor = 1.0 - (cloud / 100.0 * 0.8)  # Clouds reduce by up to 80%
            else:
                weather_factor = 0.6  # Assume cloudy if no data
            
            # Solar window: 9-12 local time (from historical analysis)
            solar_start = EFFECTIVE_SOLAR_START_HOUR  # 9
            solar_end = EFFECTIVE_SOLAR_END_HOUR      # 12
            
            # Determine consumption based on time of day (from historical data)
            if 0 <= local_hour < 8:
                consumption_wh = CONSUMPTION_NIGHT * mults["consumption_mult"]  # ~5 Wh/h
            else:
                consumption_wh = CONSUMPTION_DAY * mults["consumption_mult"]    # ~20 Wh/h
            
            # Calculate energy change this hour
            if solar_start <= local_hour <= solar_end:
                # During solar window: charging happens (gross rate, not net)
                charge_wh = base_charge_rate * weather_factor * mults["solar_mult"]
                net_change = charge_wh  # Charging is GROSS - already accounts for some consumption
                
                # Track solar production (only for first scenario)
                if scenario_name == "expected":
                    solar_predictions.append({
                        "timestamp": local_time.isoformat(),
                        "value": round(charge_wh, 1)
                    })
            else:
                # Outside solar window: consumption only
                net_change = -consumption_wh
                
                if scenario_name == "expected":
                    solar_predictions.append({
                        "timestamp": local_time.isoformat(),
                        "value": 0.0
                    })
            
            # Update SoC with limits
            soc = soc + net_change
            soc = max(BATTERY_MIN_WH, min(battery_max_wh, soc))
            
            predictions.append({
                "timestamp": local_time.isoformat(),
                "value": round(soc, 0)
            })
        
        results[scenario_name] = predictions
    
    return results["expected"], results["optimistic"], results["pessimistic"], solar_predictions


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


def _load_multi_horizon_models():
    """
    Load the new multi-horizon ML models trained with backtest validation.
    
    :return: Dictionary with models, scaler, feature names, and metadata
             or None if not available
    """
    import pickle
    models_path = MODELS_DIR / "energy_forecast_models.pkl"
    
    if not models_path.exists():
        print(f"Multi-horizon models not found at {models_path}")
        return None
    
    try:
        with open(models_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded multi-horizon models: horizons={list(data['models'].keys())}")
        return data
    except Exception as e:
        print(f"Error loading multi-horizon models: {e}")
        return None


def _prepare_features_for_new_model(
    capacity_wh: float,
    power_watts: float,
    local_hour: int,
    month: int,
    shortwave_radiation: float,
    cloud_cover: float,
    sunshine_minutes: float,
    capacity_history: List[float]
) -> np.ndarray:
    """
    Prepare features matching the new multi-horizon model training.
    
    Features: ['capacity_wh', 'power_watts', 'hour_sin', 'hour_cos', 'month_sin',
               'month_cos', 'shortwave_radiation', 'cloud_cover', 'sunshine_minutes',
               'effective_sunshine', 'in_solar_window', 'capacity_lag_1h',
               'capacity_lag_3h', 'capacity_lag_6h', 'capacity_rolling_6h',
               'capacity_rolling_24h']
    """
    # Cyclical encoding for hour and month
    hour_sin = np.sin(2 * np.pi * local_hour / 24)
    hour_cos = np.cos(2 * np.pi * local_hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Solar window (7-11 AM based on historical data)
    in_solar_window = 1.0 if 7 <= local_hour <= 11 else 0.0
    effective_sunshine = sunshine_minutes * in_solar_window
    
    # Lag features from capacity history
    hist = capacity_history if len(capacity_history) >= 24 else [capacity_wh] * 24
    capacity_lag_1h = hist[-1] if len(hist) >= 1 else capacity_wh
    capacity_lag_3h = hist[-3] if len(hist) >= 3 else capacity_wh
    capacity_lag_6h = hist[-6] if len(hist) >= 6 else capacity_wh
    
    # Rolling averages
    capacity_rolling_6h = np.mean(hist[-6:]) if len(hist) >= 6 else capacity_wh
    capacity_rolling_24h = np.mean(hist[-24:]) if len(hist) >= 24 else capacity_wh
    
    # Feature array matching training order EXACTLY
    features = np.array([
        capacity_wh,
        power_watts,
        hour_sin, hour_cos,
        month_sin, month_cos,
        shortwave_radiation,
        cloud_cover,
        sunshine_minutes,
        effective_sunshine,
        in_solar_window,
        capacity_lag_1h,
        capacity_lag_3h,
        capacity_lag_6h,
        capacity_rolling_6h,
        capacity_rolling_24h
    ]).reshape(1, -1)
    
    return features


def _ml_battery_forecast(
    weather_data: dict,
    initial_soc_wh: float,
    forecast_hours: int,
    battery_max_wh: float
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Generate battery SoC forecast using the validated multi-horizon ML model.
    
    Uses iterative prediction: predict 1 hour ahead, use that as input for next hour.
    This matches how the model was trained and validated in the backtest.
    """
    # Load the multi-horizon models
    model_data = _load_multi_horizon_models()
    if model_data is None:
        print("ML models not available, falling back to physics-based")
        return _simple_battery_forecast(
            weather_data, initial_soc_wh, forecast_hours, 12.0, battery_max_wh
        )
    
    models = model_data['models']
    scalers = model_data['scalers']  # Dict: {horizon: scaler}
    
    # Get weather data
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    radiations = hourly.get("shortwave_radiation", [0] * forecast_hours)
    cloud_covers = hourly.get("cloud_cover", [80] * forecast_hours)
    sunshine_durations = hourly.get("sunshine_duration", [0] * forecast_hours)
    
    # Scenario adjustments - affect both weather input AND prediction confidence
    # The ML model is heavily weighted on capacity_wh (96%), so we add a post-prediction adjustment
    scenarios = {
        "expected": {
            "solar_mult": 1.0,
            "cloud_adj": 0,      # No cloud adjustment
            "sunshine_mult": 1.0,
            "charge_bonus": 0,   # Wh bonus/penalty during solar hours
        },
        "optimistic": {
            "solar_mult": 1.4,
            "cloud_adj": -20,    # Less clouds (more sun)
            "sunshine_mult": 1.3,
            "charge_bonus": 30,  # Extra Wh per solar hour
        },
        "pessimistic": {
            "solar_mult": 0.5,
            "cloud_adj": 30,     # More clouds
            "sunshine_mult": 0.5,
            "charge_bonus": -20, # Less charging per solar hour
        },
    }
    
    results = {}
    solar_predictions = []
    
    for scenario_name, config in scenarios.items():
        capacity_wh = initial_soc_wh
        capacity_history = [initial_soc_wh] * 24  # Initialize history
        predictions = []
        
        for i in range(forecast_hours):
            # Get local time from weather API (already Europe/Berlin)
            if i < len(times):
                try:
                    local_time = datetime.fromisoformat(times[i])
                except:
                    local_time = datetime.now() + timedelta(hours=i)
            else:
                local_time = datetime.now() + timedelta(hours=i)
            
            local_hour = local_time.hour
            month = local_time.month
            
            # Get weather for this hour with scenario adjustments
            base_radiation = radiations[i] if i < len(radiations) else 0
            radiation = base_radiation * config["solar_mult"]
            
            base_cloud = cloud_covers[i] if i < len(cloud_covers) else 80
            cloud = max(0, min(100, base_cloud + config["cloud_adj"]))
            
            base_sunshine = (sunshine_durations[i] / 60.0) if i < len(sunshine_durations) else 0
            sunshine = base_sunshine * config["sunshine_mult"]
            
            # Estimate power consumption (from historical: ~12 Wh/hour average)
            power_watts = 12.0
            
            # Prepare features
            features = _prepare_features_for_new_model(
                capacity_wh=capacity_wh,
                power_watts=power_watts,
                local_hour=local_hour,
                month=month,
                shortwave_radiation=radiation,
                cloud_cover=cloud,
                sunshine_minutes=sunshine,
                capacity_history=capacity_history
            )
            
            # Scale features (use scaler for 1h horizon)
            features_scaled = scalers[1].transform(features)
            
            # Predict next hour using +1h model
            predicted_capacity = models[1].predict(features_scaled)[0]
            
            # Apply scenario-specific adjustment during solar hours
            # This creates meaningful differences between scenarios
            if 7 <= local_hour <= 11:
                predicted_capacity += config["charge_bonus"]
            
            # Apply battery limits
            predicted_capacity = max(BATTERY_MIN_WH, min(battery_max_wh, predicted_capacity))
            
            # Update for next iteration
            capacity_wh = predicted_capacity
            capacity_history.append(capacity_wh)
            
            predictions.append({
                "timestamp": local_time.isoformat(),
                "value": round(capacity_wh, 0)
            })
            
            # Track solar production for expected scenario
            if scenario_name == "expected":
                # Estimate solar production from capacity change
                if 7 <= local_hour <= 11 and radiation > 0:
                    solar_est = radiation * 0.35 * 0.6  # Rough estimate with shading
                else:
                    solar_est = 0
                solar_predictions.append({
                    "timestamp": local_time.isoformat(),
                    "value": round(solar_est, 1)
                })
        
        results[scenario_name] = predictions
    
    return results["expected"], results["optimistic"], results["pessimistic"], solar_predictions


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
    sunshine_minutes: float,
    capacity_history: List[float],
    radiation_history: List[float] = None
) -> np.ndarray:
    """
    Prepare feature vector for ML model prediction.
    
    Features must match the training order from train_enhanced_model.py:
    - hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    - power_watts
    - shortwave_radiation, cloud_cover, sunshine_minutes
    - effective_sunshine, in_solar_window
    - capacity_lag_1h, capacity_lag_3h, capacity_lag_6h, capacity_lag_12h, capacity_lag_24h
    - capacity_rolling_6h, capacity_rolling_24h
    - power_rolling_6h, power_rolling_24h
    - radiation_rolling_6h
    
    :return: Feature array matching training features
    """
    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Solar window flag (7-11 AM at this location)
    in_solar_window = 1.0 if EFFECTIVE_SOLAR_START_HOUR <= hour <= EFFECTIVE_SOLAR_END_HOUR else 0.0
    
    # Effective sunshine - sunshine during solar window
    effective_sunshine = sunshine_minutes * in_solar_window
    
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
    
    # Radiation rolling average
    if radiation_history and len(radiation_history) >= 6:
        radiation_rolling_6h = np.mean(radiation_history[-6:])
    else:
        radiation_rolling_6h = shortwave_radiation
    
    # Feature array matching training order EXACTLY
    features = np.array([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        month_sin, month_cos,
        power_watts,
        shortwave_radiation, cloud_cover_pct, sunshine_minutes,
        effective_sunshine, in_solar_window,
        cap_lag_1h, cap_lag_3h, cap_lag_6h, cap_lag_12h, cap_lag_24h,
        cap_rolling_6h, cap_rolling_24h,
        power_rolling_6h, power_rolling_24h,
        radiation_rolling_6h
    ]).reshape(1, -1)
    
    return features


def predict_with_ml_model(
    model,
    scaler,
    weather_data: dict,
    initial_soc_wh: float,
    avg_consumption_watts: float,
    battery_max_wh: float,
    forecast_hours: int = 336,
    max_solar_output_watts: float = 600
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate battery SoC predictions using ML model with physics-based constraints.
    
    The ML model provides a prediction, but we apply physical limits:
    - Battery cannot lose more than consumption allows per hour
    - Battery cannot gain more than solar production allows per hour
    
    :return: Tuple of (expected, optimistic, pessimistic) prediction lists
    """
    hourly = weather_data.get("hourly", {})
    times = hourly.get("time", [])
    radiations = hourly.get("shortwave_radiation", [0] * forecast_hours)
    cloud_covers = hourly.get("cloud_cover", [50] * forecast_hours)
    sunshine_durations = hourly.get("sunshine_duration", [0] * forecast_hours)  # Minutes of sunshine
    
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    # Seasonal factor from site configuration
    month = now.month
    seasonal_factor = get_seasonal_factor(month)
    
    # Scenario configuration using site-specific values
    scenarios = {
        "expected": {
            "consumption_factor": SCENARIO_CONSUMPTION_FACTORS["expected"], 
            "solar_factor": SCENARIO_SOLAR_FACTORS["expected"],
            "use_ml": True,
            "ml_weight": 0.6  # Increased ML weight since model is better now
        },
        "optimistic": {
            "consumption_factor": SCENARIO_CONSUMPTION_FACTORS["optimistic"],
            "solar_factor": SCENARIO_SOLAR_FACTORS["optimistic"],
            "use_ml": True,
            "ml_weight": 0.4
        },
        "pessimistic": {
            "consumption_factor": SCENARIO_CONSUMPTION_FACTORS["pessimistic"],
            "solar_factor": SCENARIO_SOLAR_FACTORS["pessimistic"],
            "use_ml": True,
            "ml_weight": 0.3
        }
    }
    
    results = {"expected": [], "optimistic": [], "pessimistic": []}
    
    for scenario_name, config in scenarios.items():
        soc_wh = initial_soc_wh
        capacity_history = [initial_soc_wh] * 24  # Initialize with current value
        radiation_history = [0.0] * 6  # Track radiation for rolling average
        predictions = []
        
        for i in range(min(forecast_hours, max(len(radiations), 336))):
            forecast_time = now + timedelta(hours=i)
            
            # Get weather for this hour
            radiation = radiations[i] if i < len(radiations) else 0
            cloud = cloud_covers[i] if i < len(cloud_covers) else 80  # Assume cloudy if no data
            sunshine = sunshine_durations[i] if i < len(sunshine_durations) else 0
            
            # Track radiation history
            radiation_history.append(radiation)
            
            # Use LOCAL time for solar window (API returns Europe/Berlin times)
            # Parse local time from weather API if available
            if i < len(times):
                try:
                    local_time = datetime.fromisoformat(times[i])
                    local_hour = local_time.hour
                except:
                    local_hour = (forecast_time.hour + 1) % 24  # Approximate CET
            else:
                local_hour = (forecast_time.hour + 1) % 24
            
            # Get effective solar fraction for this hour (accounts for terrain shading)
            solar_window_factor = get_effective_solar_fraction(local_hour)
            
            if solar_window_factor > 0 and radiation > 0:
                # Normalize radiation (typical peak ~1000 W/m²)
                radiation_factor = min(1.0, radiation / 1000.0)
                cloud_factor = 1.0 - (cloud / 100.0 * 0.85)
                
                # Apply SITE SHADING FACTOR
                solar_output_wh = (max_solar_output_watts * radiation_factor * cloud_factor * 
                                   solar_window_factor * config["solar_factor"] * 
                                   seasonal_factor * SITE_SHADING_FACTOR)
            else:
                solar_output_wh = 0
            
            # Consumption for this hour (Watts = Wh over 1 hour)
            consumption_wh = avg_consumption_watts * config["consumption_factor"]
            
            # Net energy change (positive = charging, negative = discharging)
            net_energy_wh = solar_output_wh - consumption_wh
            
            # Calculate pure physics SoC with realistic battery minimum
            physics_soc = soc_wh + net_energy_wh
            physics_soc = max(BATTERY_MIN_WH, min(battery_max_wh, physics_soc))
            
            # Try ML prediction
            ml_soc = None
            if config["use_ml"]:
                try:
                    features = _prepare_ml_features(
                        hour=local_hour,
                        day_of_week=forecast_time.weekday(),
                        month=forecast_time.month,
                        power_watts=avg_consumption_watts * config["consumption_factor"],
                        shortwave_radiation=radiation,
                        cloud_cover_pct=cloud,
                        sunshine_minutes=sunshine,
                        capacity_history=capacity_history[-24:],
                        radiation_history=radiation_history[-6:]
                    )
                    features_scaled = scaler.transform(features)
                    ml_soc = model.predict(features_scaled)[0]
                    ml_soc = max(BATTERY_MIN_WH, min(battery_max_wh, ml_soc))
                except Exception as e:
                    ml_soc = None
                
                # Hybrid blend with time-decay (less ML over time)
                if ml_soc is not None:
                    time_decay = max(0.3, 1.0 - (i / 336))
                    effective_ml_weight = config["ml_weight"] * time_decay
                    soc_wh = effective_ml_weight * ml_soc + (1 - effective_ml_weight) * physics_soc
                else:
                    soc_wh = physics_soc
            else:
                soc_wh = physics_soc
            
            # Apply hard limits with realistic minimum
            soc_wh = max(BATTERY_MIN_WH, min(battery_max_wh, soc_wh))
            
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
    times = hourly.get("time", [])  # Local time strings from API
    
    predictions = []
    
    # Use weather API times directly (they're in Europe/Berlin timezone)
    for i in range(min(hours, len(times) if times else hours)):
        # Parse local time from weather API or calculate from UTC
        if i < len(times):
            try:
                local_time = datetime.fromisoformat(times[i])
            except:
                # Fallback: estimate local time (UTC+1 in winter, UTC+2 in summer)
                utc_now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                local_time = utc_now + timedelta(hours=i+1)  # Approximate CET
        else:
            utc_now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            local_time = utc_now + timedelta(hours=i+1)
        
        weather_idx = min(hour_offset + i, len(cloud_covers) - 1)
        
        # Solar output factor based on radiation and cloud cover
        radiation = radiations[weather_idx] if weather_idx < len(radiations) else 400
        cloud = cloud_covers[weather_idx] if weather_idx < len(cloud_covers) else 50
        
        # Normalize radiation (typical max ~1000 W/m²) and apply cloud factor
        radiation_factor = min(1.0, radiation / 1000.0)
        cloud_factor = 1.0 - (cloud / 100.0 * 0.8)
        
        # Use site-specific solar window and shading factor
        # IMPORTANT: Use LOCAL hour for solar window calculation
        local_hour = local_time.hour
        solar_window_factor = get_effective_solar_fraction(local_hour)
        seasonal_factor = get_seasonal_factor(local_time.month)
        
        # Apply site shading factor - location receives only ~35% of theoretical sunshine
        output = (max_solar_output_watts * radiation_factor * cloud_factor * 
                  solar_window_factor * seasonal_factor * SITE_SHADING_FACTOR)
        
        predictions.append({
            "timestamp": local_time.isoformat() + "Z",
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
    
    # Use scenario multipliers from site config
    scenarios = {
        "expected": {"solar": SCENARIO_SOLAR_FACTORS["expected"], 
                     "consumption": SCENARIO_CONSUMPTION_FACTORS["expected"]},
        "optimistic": {"solar": SCENARIO_SOLAR_FACTORS["optimistic"], 
                       "consumption": SCENARIO_CONSUMPTION_FACTORS["optimistic"]},
        "pessimistic": {"solar": SCENARIO_SOLAR_FACTORS["pessimistic"], 
                        "consumption": SCENARIO_CONSUMPTION_FACTORS["pessimistic"]}
    }
    
    for scenario_name, factors in scenarios.items():
        soc_wh = initial_soc_wh
        predictions = []
        
        for i, solar_pred in enumerate(solar_predictions[:hours]):
            # Net power = production - consumption
            solar_output = solar_pred["value"] * factors["solar"]
            consumption = consumption_watts * factors["consumption"]
            net_power = solar_output - consumption
            
            # Update SoC (assume 1-hour intervals) with realistic minimum
            soc_wh = soc_wh + net_power
            soc_wh = max(BATTERY_MIN_WH, min(battery_max_wh, soc_wh))
            
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
    
    Returns actions in format expected by Dashboard Backend validator:
    [{"name": "expected", "value": [{"timestamp": ..., "value": ..., "action": ...}]}]
    
    :param soc_predictions: Battery SoC predictions (expected scenario)
    :param battery_max_wh: Maximum battery capacity in Wh
    :return: List of action scenario groups with scheduled actions
    """
    action_entries = []
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
            action_entries.append({
                "timestamp": action_timestamp,
                "action": "connect_grid",
                "value": True
            })
            grid_connected = True
        
        # Check if we can disconnect grid (battery recovered)
        if soc_percent > 50 and grid_connected:
            action_entries.append({
                "timestamp": action_timestamp,
                "action": "connect_grid",
                "value": False
            })
            grid_connected = False
    
    # Wrap in scenario format expected by Dashboard Backend validator
    # Format: [{"name": "scenario_name", "value": [actions]}]
    return [
        {"name": "expected", "value": action_entries},
        {"name": "optimistic", "value": []},
        {"name": "pessimistic", "value": []}
    ]


def energy_forecast(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    forecast_hours: int = 336,
    max_solar_output_watts: float = None,  # Uses site config if None
    avg_consumption_watts: float = None,    # Uses site config if None
    initial_soc_wh: float = 800,
    battery_max_wh: float = None            # Uses site config if None
) -> Dict[str, Any]:
    """
    Generate complete energy forecast with proactive actions.
    
    Uses ML model if available, falls back to rule-based approach.
    
    :return: Dictionary with forecasts and actions
    """
    # Apply site configuration defaults
    if max_solar_output_watts is None:
        max_solar_output_watts = MAX_SOLAR_OUTPUT_WATTS
    if avg_consumption_watts is None:
        avg_consumption_watts = TYPICAL_CONSUMPTION_WATTS
    if battery_max_wh is None:
        battery_max_wh = BATTERY_MAX_WH
    
    # Fetch weather data
    weather_data = fetch_weather_forecast(latitude, longitude, forecast_hours)
    weather_source = "live" if weather_data else "fallback"
    
    # Try to use the new multi-horizon ML model (trained with backtest validation)
    model_data = _load_multi_horizon_models()
    use_ml = model_data is not None
    
    if use_ml:
        print("Using ML-based predictions (validated multi-horizon model)")
        expected, optimistic, pessimistic, solar_predictions = _ml_battery_forecast(
            weather_data=weather_data,
            initial_soc_wh=initial_soc_wh,
            forecast_hours=forecast_hours,
            battery_max_wh=battery_max_wh
        )
        prediction_method = "ml_multi_horizon"
    else:
        # Fallback to physics-based approach
        print("Using physics-based predictions (ML model not available)")
        expected, optimistic, pessimistic, solar_predictions = _simple_battery_forecast(
            weather_data=weather_data,
            initial_soc_wh=initial_soc_wh,
            forecast_hours=forecast_hours,
            consumption_wh_per_hour=avg_consumption_watts,
            battery_max_wh=battery_max_wh
        )
        prediction_method = "physics_based"
    
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
                        {"timestamp": p["timestamp"], "value": round(p["value"] * 1.3, 1)}
                        for p in solar_predictions
                    ]},
                    {"name": "pessimistic", "value": [
                        {"timestamp": p["timestamp"], "value": round(p["value"] * 0.5, 1)}
                        for p in solar_predictions
                    ]}
                ]
            }
        ],
        "actions": actions,
        "weather_source": weather_source,
        "prediction_method": prediction_method,
        "forecast_hours": forecast_hours
    }
