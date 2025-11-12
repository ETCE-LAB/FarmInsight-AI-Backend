import pandas as pd
import requests

from model_service.model_service.utils.greenhouse_calculator import compute_greenhouse_roof_rain

def prepare_data_for_training():

    rain = pd.read_csv("../data/rain_amount.csv", parse_dates=["measuredAt"])
    soil = pd.read_csv("../data/soil_moisture.csv", parse_dates=["measuredAt"])
    water = pd.read_csv("../data/water_level.csv", parse_dates=["measuredAt"])

    rain = rain.rename(columns={"value": "rain_amount"})
    soil = soil.rename(columns={"value": "soil_moisture"})
    water = water.rename(columns={"value": "water_level"})

    rain["date"] = rain["measuredAt"].dt.date
    soil["date"] = soil["measuredAt"].dt.date
    water["date"] = water["measuredAt"].dt.date

    daily_rain = rain.groupby("date", as_index=False)["rain_amount"].sum()
    daily_soil = soil.groupby("date", as_index=False)["soil_moisture"].mean()
    daily_water = water.groupby("date", as_index=False)["water_level"].mean()

    merged = (
        daily_rain
        .merge(daily_soil, on="date", how="outer")
        .merge(daily_water, on="date", how="outer")
        .sort_values("date")
    )

    start_date = pd.to_datetime("2025-10-01")
    end_date = pd.to_datetime("2025-10-25")

    merged["date"] = pd.to_datetime(merged["date"])

    merged = merged[(merged["date"] >= start_date) & (merged["date"] <= end_date)]

    latitude = 51.9
    longitude = 10.42

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}"
        f"&daily=rain_sum,wind_direction_10m_dominant,temperature_2m_max,sunshine_duration"
        f"&timezone=Europe%2FBerlin"
    )

    response = requests.get(url, timeout=15)
    historical_forecast = response.json()

    # for ML, it will learn this factors, for now, they're estimated
    IS_TRAINING_ML = False

    if IS_TRAINING_ML:
        first_flush_loss_ML = 0.0
        runoff_coeff_ML = 1.0
    else:
        first_flush_loss_ML = 0.03
        runoff_coeff_ML = 0.9

    face_azimuth_deg = 30.0
    face_area_m2 = 6.668
    slope_deg = 30.26

    computed_inflow = compute_greenhouse_roof_rain(
        historical_forecast,
        runoff_coeff=runoff_coeff_ML,
        first_flush_loss=first_flush_loss_ML,
        face_azimuth_deg=face_azimuth_deg,
        face_area_m2=face_area_m2,
        slope_deg=slope_deg
    )

    df_inflow = pd.DataFrame(computed_inflow)
    df_inflow['date'] = pd.to_datetime(df_inflow['date'])

    weather_features = pd.DataFrame({
        'date': historical_forecast['daily']['time'],
        'Tmax_c': historical_forecast['daily']['temperature_2m_max'],
        'sunshine_duration': historical_forecast['daily']['sunshine_duration'],
    })
    weather_features['date'] = pd.to_datetime(weather_features['date'])

    df_train = pd.merge(
        merged,
        df_inflow[['date', 'total_m3', 'applied_loss_global']],
        on='date',
        how='inner'
    )
    df_train = pd.merge(df_train, weather_features, on='date', how='inner')

    df_train['water_level_prev'] = df_train['water_level'].shift(1)

    df_train["calculated_total_l"] = df_train["total_m3"] * 1000

    df_train = df_train.dropna()

    df_train.to_csv("../data/optimized_data_hist_computed.csv", index=False)

    # wann die Pumpe aktiv war, muss ergänzt werden, damit das Modell weiß, warum der Füllstand weniger wurde
