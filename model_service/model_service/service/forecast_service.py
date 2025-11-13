from typing import Dict, List, Tuple

import requests

from ..utils.greenhouse_calculator import compute_greenhouse_roof_rain
from ..utils.simulation_loop import run_forecast_period

try:
    # ML_MODEL = joblib.load("model.pkl")

    from ..utils.predictor import MockTankModel

    ML_MODEL = MockTankModel(water_in_tank_loss_factor=0.005)

except FileNotFoundError:
    print("WARNING: Missing ML-model. Use mock model")
    ML_MODEL = MockTankModel(water_in_tank_loss_factor=0.01)

FACE_AZIMUTH_DEG = 30.0
FACE_AREA_M2 = 6.668
SLOPE_DEG = 30.26

# estimated plant area
A_SOIL_M2 = 8.0

MAX_IRRIGATION_L_DAY = 1.5


def model_forecast(
        latitude: float,
        longitude: float,
        forecast_days: int,
        initial_water_level: float,
        tank_capacity_liters: float,
        soil_threshold: float,
        scenarios: list,
        start_soil_moisture: float,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&daily=rain_sum,sunshine_duration,weather_code,wind_speed_10m_max,wind_direction_10m_dominant,"
        f"wind_gusts_10m_max,temperature_2m_min,temperature_2m_max,sunrise,sunset,"
        f"precipitation_sum,precipitation_probability_max"
        f"&timezone=Europe%2FBerlin&forecast_days={forecast_days}"
    )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    forecast = resp.json()

    inflow_data = compute_greenhouse_roof_rain(
        forecast,
        face_azimuth_deg=FACE_AZIMUTH_DEG,
        face_area_m2=FACE_AREA_M2,
        slope_deg=SLOPE_DEG,
        precip_key="rain_sum",
        wind_bias_strength=0.35,
        wind_exposure_factors=(0.8, 0.9),
        runoff_coeff=1.0,  # for ML-model, factor learned by model
        first_flush_loss=0.0  # for ML-model, factor learned by model
    )
    # print("\n data: ", inflow_data)

    tank_results, soil_results = run_forecast_period(
        forecast=forecast,
        inflow_data=inflow_data,
        ml_model=ML_MODEL,

        initial_water_level=initial_water_level,
        initial_moisture=start_soil_moisture,
        tank_capacity_liters=tank_capacity_liters,
        soil_threshold=soil_threshold,

        max_irrigation_l_day=MAX_IRRIGATION_L_DAY,
        plant_area=A_SOIL_M2,
        scenarios=scenarios
    )

    return tank_results, soil_results
