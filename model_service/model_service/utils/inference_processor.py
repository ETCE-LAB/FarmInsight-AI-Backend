import os
from typing import Tuple, Dict, List, Any

import pandas as pd
import requests

from ..utils.greenhouse_calculator import compute_greenhouse_roof_rain


def prepare_data_for_prediction(
        latitude: float,
        longitude: float,
        forecast_days: int,
        *,
        runoff_coeff: float = 0.9,
        first_flush_loss: float = 0.03,
        face_azimuth_deg: float = 30.0,
        face_area_m2: float = 6.668,
        slope_deg: float = 30.26,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&daily=rain_sum,sunshine_duration,weather_code,wind_speed_10m_max,wind_direction_10m_dominant,"
        f"wind_gusts_10m_max,temperature_2m_min,temperature_2m_max,sunrise,sunset,"
        f"precipitation_sum,precipitation_probability_max"
        f"&timezone=Europe%2FBerlin&forecast_days={forecast_days}"
    )

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    forecast = response.json()

    inflow_data = compute_greenhouse_roof_rain(
        forecast,
        runoff_coeff=runoff_coeff,
        first_flush_loss=first_flush_loss,
        face_azimuth_deg=face_azimuth_deg,
        face_area_m2=face_area_m2,
        slope_deg=slope_deg,
    )

    df_inflow = pd.DataFrame(inflow_data)
    df_inflow["date"] = pd.to_datetime(df_inflow["date"])

    daily = forecast.get("daily", {})
    weather_features = pd.DataFrame({
        "date": daily.get("time", []),
        "Tmax_c": daily.get("temperature_2m_max", []),
        "sunshine_duration": daily.get("sunshine_duration", []),
        "rain_sum": daily.get("rain_sum", []),
    })
    weather_features["date"] = pd.to_datetime(weather_features["date"])

    df_forecast = weather_features.merge(
        df_inflow[["date", "total_m3", "applied_loss_global"]],
        on="date",
        how="inner",
    )

    return forecast, inflow_data, df_forecast
