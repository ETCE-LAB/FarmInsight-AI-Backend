import os
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..utils.greenhouse_calculator import compute_greenhouse_roof_rain

from django.conf import settings

BASE_DIR = settings.BASE_DIR
DATA_DIR = BASE_DIR / "model_service" / "data"


def load_dwd_rain_sun_txt(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=";",
        na_values=[-999, -999.0],
        dtype={"MESS_DATUM": str},
    )
    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d")
    df["rain_mm"] = df["RS"].astype(float)
    df["sunshine_duration"] = df["SH_TAG"].astype(float) * 3600.0

    return (
        df[["date", "rain_mm", "sunshine_duration"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_dwd_10min_temperature(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=";",
        na_values=[-999],
        dtype={"MESS_DATUM": str},
    )

    df.columns = df.columns.str.strip()

    df["datetime"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H%M")
    df["date"] = df["datetime"].dt.date

    # TT_10 in 0.1 °C -> °C
    df["T_c"] = df["TT_10"].astype(float) / 10.0

    daily = (
        df.groupby("date", as_index=False)["T_c"]
        .max()
        .rename(columns={"T_c": "Tmax_c"})
    )

    daily["date"] = pd.to_datetime(daily["date"])
    return daily[["date", "Tmax_c"]].sort_values("date").reset_index(drop=True)


def add_inflow_from_greenhouse(df_driver: pd.DataFrame) -> pd.DataFrame:
    historical_forecast = {
        "daily": {
            "time": df_driver["date"].dt.strftime("%Y-%m-%d").tolist(),
            "rain_sum": df_driver["rain_mm"].tolist(),
            "temperature_2m_max": df_driver["Tmax_c"].tolist(),
            "sunshine_duration": df_driver["sunshine_duration"].tolist(),
        }
    }

    computed_inflow = compute_greenhouse_roof_rain(
        historical_forecast,
        runoff_coeff=0.9,
        first_flush_loss=0.03,
        face_azimuth_deg=30.0,
        face_area_m2=6.668,
        slope_deg=30.26,
    )

    df_inflow = pd.DataFrame(computed_inflow)
    df_inflow["date"] = pd.to_datetime(df_inflow["date"])

    df = df_driver.merge(df_inflow[["date", "total_m3"]], on="date", how="inner")
    return df.sort_values("date").reset_index(drop=True)


def build_synthetic_time_series_v2(
        dwd_df: pd.DataFrame,
        tank_capacity_l: float,
        plant_area_m2: float,
        base_loss_l_per_day: float,
        initial_water_level: float,
        initial_moisture: float,
        k_irrig: float = 3.0,
        k_rain: float = 0.15,
        k_evap: float = 0.25,
        soil_min_target: float = 55.0,
        soil_max_target: float = 75.0,
        exploration_prob: float = 0.7,
        inflow_l_per_m3: float = 300.0,
) -> pd.DataFrame:
    dwd_df = dwd_df.sort_values("date").reset_index(drop=True)

    water_level = initial_water_level
    moisture = initial_moisture

    rows: List[Dict] = []

    liters_per_pump = 1.5

    for _, row in dwd_df.iterrows():
        date = row["date"]
        rain_mm = float(row["rain_mm"])
        temp = float(row["Tmax_c"])
        sun_sec = float(row["sunshine_duration"])
        total_m3 = float(row["total_m3"])

        inflow_l = total_m3 * inflow_l_per_m3

        sun_factor = sun_sec / 86400.0
        et0_proxy = 0.08 * temp + 0.15 * sun_factor * temp

        if moisture < soil_min_target:
            base_pump_usage = 2
        elif moisture > soil_max_target:
            base_pump_usage = 0
        else:
            base_pump_usage = 1

        if random.random() < exploration_prob:
            pump_usage = random.choice([0, 1, 2])
        else:
            pump_usage = base_pump_usage

        irrigation_l = min(pump_usage * liters_per_pump, water_level)
        irrigation_mm = irrigation_l / plant_area_m2 if plant_area_m2 > 0 else 0.0

        water_level = water_level + inflow_l - irrigation_l - base_loss_l_per_day
        water_level = max(0.0, min(tank_capacity_l, water_level))

        moisture = (
                moisture
                + k_irrig * irrigation_mm
                + k_rain * rain_mm
                - k_evap * et0_proxy
        )

        if moisture > 80.0:
            moisture -= 0.1 * (moisture - 80.0)

        moisture = max(0.0, min(100.0, moisture))

        rows.append(
            {
                "date": date,
                "rain_mm": rain_mm,
                "Tmax_c": temp,
                "sunshine_duration": sun_sec,
                "total_m3": total_m3,
                "calculated_total_l": inflow_l,
                "irrigation_l": irrigation_l,
                "pump_usage": pump_usage,
                "water_level": water_level,
                "soil_moisture": moisture,
            }
        )

    return pd.DataFrame(rows)


def build_ml_dataset_from_series(df_sim: pd.DataFrame, h: int = 7) -> pd.DataFrame:
    df = df_sim.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["water_level_prev"] = df["water_level"].shift(1)
    df["soil_moisture_prev"] = df["soil_moisture"].shift(1)

    df["water_level_tomorrow"] = df["water_level"].shift(-1)
    df["soil_moisture_tomorrow"] = df["soil_moisture"].shift(-1)

    df["delta_water_level"] = df["water_level_tomorrow"] - df["water_level"]
    df["delta_soil_moisture"] = df["soil_moisture_tomorrow"] - df["soil_moisture"]

    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year

    df["rain_today"] = df["rain_mm"]
    df["rain_tomorrow"] = df["rain_mm"].shift(-1)
    df["temp_today"] = df["Tmax_c"]
    df["temp_tomorrow"] = df["Tmax_c"].shift(-1)

    df["irrigation_today"] = df["irrigation_l"]

    for i in range(1, h + 1):
        df[f"irrigation_last{i}_days"] = df["irrigation_l"].shift(i)

    df["irrigation_last_h_days"] = df[
        [f"irrigation_last{i}_days" for i in range(1, h + 1)]
    ].sum(axis=1)

    df = df.dropna().reset_index(drop=True)
    return df


def build_synthetic_ml_dataset_from_dwd_txt_v2(
        path_rain: str | Path, path_temp: str | Path
) -> pd.DataFrame:
    path_rain = Path(path_rain)
    path_temp = Path(path_temp)

    df_rain = load_dwd_rain_sun_txt(path_rain)
    df_temp = load_dwd_10min_temperature(path_temp)

    df_driver = df_rain.merge(df_temp, on="date", how="inner")
    df_driver = add_inflow_from_greenhouse(df_driver)

    df_sim = build_synthetic_time_series_v2(
        dwd_df=df_driver,
        tank_capacity_l=210.0,
        plant_area_m2=10.0,
        base_loss_l_per_day=2.0,
        initial_water_level=80.0,
        initial_moisture=60.0,
        k_irrig=3.0,
        k_rain=0.15,
        k_evap=0.25,
        soil_min_target=55.0,
        soil_max_target=75.0,
        exploration_prob=0.7,
        inflow_l_per_m3=300.0,
    )

    df_ml = build_ml_dataset_from_series(df_sim, h=7)
    return df_ml


def prepare_synthetic_training_data():
    dwd_path_rain = DATA_DIR / "dwd_data_rain.txt"
    dwd_path_temp = DATA_DIR / "dwd_data_temp.txt"

    df_ml = build_synthetic_ml_dataset_from_dwd_txt_v2(dwd_path_rain, dwd_path_temp)
    out_path = DATA_DIR / "synthetic_training_data.csv"
    df_ml.to_csv(out_path, index=False)
