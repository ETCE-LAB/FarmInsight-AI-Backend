from os import makedirs

import pandas as pd
import requests

from model_service.model_service.utils.greenhouse_calculator import compute_greenhouse_roof_rain

import calendar


def days_in_year(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def check_timeline_in_data(series: pd.Series):
    series = pd.to_datetime(series).sort_values().reset_index(drop=True)
    for i in range(0, len(series) - 1):
        t0 = series.iloc[i]
        t1 = series.iloc[i + 1]

        day0 = t0.day_of_year
        day1 = t1.day_of_year

        year_of_day0 = series.iloc[i].year
        days_in_year_of_day0 = days_in_year(year_of_day0)

        if t1.year - t0.year > 1:
            return False

        if day0 == days_in_year_of_day0:
            if day1 != 1:
                return False
        else:
            if day1 != day0 + 1:
                return False

    return True


def prepare_data_for_training():
    rain = pd.read_csv("../data/rain_amount.csv", parse_dates=["measuredAt"])
    soil = pd.read_csv("../data/soil_moisture.csv", parse_dates=["measuredAt"])
    water = pd.read_csv("../data/water_level.csv", parse_dates=["measuredAt"])

    rain = rain.rename(columns={"value": "rain_amount"})
    soil = soil.rename(columns={"value": "soil_moisture"})
    water = water.rename(columns={"value": "water_level"})

    rain["rain_amount"] = rain["rain_amount"] * 0.25  # Auflösung von Sensor checken

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

    timeline_correct = check_timeline_in_data(merged["date"])

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
    df_train["soil_moisture_prev"] = df_train["soil_moisture"].shift(1)

    df_train['water_level_tomorrow'] = df_train['water_level'].shift(-1)
    df_train["soil_moisture_tomorrow"] = df_train["soil_moisture"].shift(-1)

    df_train["calculated_total_l"] = df_train["total_m3"] * 1000

    df_train["tank_capacity"] = 150
    df_train["month"] = df_train["date"].dt.month
    df_train["day_of_year"] = df_train["date"].dt.day_of_year

    pump_usage = [1 if i % 2 == 0 else 0 for i in range(0, len(df_train))]

    df_train["pump_usage"] = pump_usage
    df_train["water_usage"] = df_train["pump_usage"] * 1.5

    h = 7

    for i in range(1, h + 1):
        df_train[f"rain_after_{i}_days"] = df_train["rain_amount"].shift(-i)  # negativer Shift = Zukunft
        df_train[f"temp_after_{i}_days"] = df_train["Tmax_c"].shift(-i)
        df_train[f"irrigation_last{i}_days"] = df_train["water_usage"].shift(i)  # positiver Shift = Vergangenheit

    df_train["rain_today"] = df_train["rain_amount"]
    df_train["rain_tomorrow"] = df_train["rain_amount"].shift(-1)

    df_train["temp_today"] = df_train["Tmax_c"]
    df_train["temp_tomorrow"] = df_train["Tmax_c"].shift(-1)

    df_train["sum_rain_next_h_days"] = df_train[[f"rain_after_{i}_days" for i in range(1, h + 1)]].sum(axis=1)
    df_train["mean_temp_next_h_days"] = df_train[[f"temp_after_{i}_days" for i in range(1, h + 1)]].mean(axis=1)
    df_train["irrigation_last_h_days"] = df_train[[f"irrigation_last{i}_days" for i in range(1, h + 1)]].sum(axis=1)

    df_train = df_train.dropna()

    makedirs("../data", exist_ok=True)
    df_train.to_csv("../data/training_data.csv", index=False)

    # wann die Pumpe aktiv war, muss ergänzt werden, damit das Modell weiß, warum der Füllstand weniger wurde

    # überprüfen, ob die tage forlaufen sind,
    # wenn Unterschied der Tage größer als 1 -> raise Error

    # FÜr die Modellaufrufe: plans = itertools.product([0,1,2], repeat=7)

    return timeline_correct
