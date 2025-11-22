from typing import Dict, List, Tuple, Any

import pandas as pd

from .model_trainer import X_feature_cols


def case_value_multiplicator(scenario: str, value: float, value_name: str) -> float:
    if scenario == "average-case":
        return value

    is_best = scenario == "best-case"

    if value_name in ("temp_max", "temp_today", "temp_tomorrow"):
        return value * (0.9 if is_best else 1.1)

    if value_name in ("rain", "rain_today", "rain_tomorrow"):
        return value * (1.1 if is_best else 0.9)

    if value_name in ("inflow_l", "calculated_total_l"):
        return value * (1.1 if is_best else 0.9)

    if value_name in ("soil_moisture_previous", "water_level_previous"):
        return value * (1.05 if is_best else 0.95)

    return value


def run_forecast_period(
        df_forecast: pd.DataFrame,
        forecast_days: int,
        ml_model: Any,
        tank_capacity: float,
        initial_water_level: float,
        initial_moisture: float,
        plant_area: float,
        scenario: str,
        plan: Tuple[int, ...]
) -> tuple[list[dict], list[dict]]:
    tank_results: List[Dict] = []
    soil_results: List[Dict] = []

    df_forecast = df_forecast.sort_values("date").reset_index(drop=True)

    water_level_today = initial_water_level
    water_level_previous = initial_water_level
    water_level_previous = case_value_multiplicator(scenario, water_level_previous, "water_level_previous")
    moisture_today = initial_moisture
    moisture_previous = initial_moisture
    moisture_previous = case_value_multiplicator(scenario, moisture_previous, "soil_moisture_previous")

    irrigation_history: List[float] = [0.0] * forecast_days

    water_level = initial_water_level
    moisture = initial_moisture

    for i, row in df_forecast.iterrows():
        date = row["date"].strftime("%Y-%m-%d")
        temp_today = float(row["Tmax_c"])
        rain_today = float(row["rain_sum"])
        water_inflow = float(row["total_m3"] * 1000.0)

        if (i + 1) < len(df_forecast):
            temp_tomorrow = float(df_forecast.iloc[i + 1]["Tmax_c"])
            rain_tomorrow = float(df_forecast.iloc[i + 1]["rain_sum"])
        else:
            temp_tomorrow = temp_today
            rain_tomorrow = rain_today

        temp_today = case_value_multiplicator(scenario, temp_today, "temp_today")
        rain_today = case_value_multiplicator(scenario, rain_today, "rain_today")
        temp_tomorrow = case_value_multiplicator(scenario, temp_tomorrow, "temp_tomorrow")
        rain_tomorrow = case_value_multiplicator(scenario, rain_tomorrow, "rain_tomorrow")
        water_inflow = case_value_multiplicator(scenario, water_inflow, "inflow_l")

        day_of_year = row["date"].timetuple().tm_yday
        month = row["date"].month

        if i < len(plan):
            pumps_today = plan[i]
        else:
            pumps_today = plan[-1]

        Qout_l = pumps_today * 1.5
        Qout_l_final = min(Qout_l, water_level_previous + water_inflow)
        Qout_mm = Qout_l_final / plant_area

        irrigation_history = irrigation_history[1:] + [Qout_l_final]
        irrigation_last_h_days = sum(irrigation_history)

        pump_usage = pumps_today

        irrigation_today = Qout_l_final

        feature_row = {
            "soil_moisture": moisture_today,
            "water_level": water_level_today,
            "soil_moisture_prev": moisture_previous,
            "water_level_prev": water_level_previous,
            "day_of_year": day_of_year,
            "month": month,
            "tank_capacity": tank_capacity,
            "temp_today": temp_today,
            "rain_today": rain_today,
            "temp_tomorrow": temp_tomorrow,
            "rain_tomorrow": rain_tomorrow,
            "irrigation_last_h_days": irrigation_last_h_days,
            "pump_usage": pump_usage,
            "calculated_total_l": water_inflow,
            "irrigation_today": irrigation_today,
        }

        X_infer = pd.DataFrame([feature_row])[X_feature_cols]
        pred = ml_model.predict(X_infer)
        water_level += float(pred[0][0])
        moisture += float(pred[0][1])
        # this will be used when data is better,
        # then the model learns the relationship between pump_usage and soil_moisture

        k_irrig = 1.5  # wie stark 1 mm Bewässerung die Bodenfeuchte erhöht
        k_rain = 0.3  # wie stark sich 1 mm Regen auswirken
        k_evap = 0.4  # Stärke der Verdunstung

        temp_factor = max(0.0, temp_today / 30.0)
        M_t_phys = (
                moisture_today
                + k_irrig * Qout_mm  # Bewässerung
                + k_rain * rain_today  # Regen
                - k_evap * temp_factor  # Verdunstung
        )

        # M_t_ml = max(0.0, min(100.0, M_t_phys))

        W_t_final = max(0.0, min(tank_capacity, water_level))
        overflow_l = max(0.0, water_level - tank_capacity)

        moisture = max(0.0, min(100.00, moisture))

        tank_result = {
            "date": date,
            "tank_l": W_t_final,
            "qin_l": water_inflow,
            "qout_l": Qout_l_final,
            "overflow_l": overflow_l,
            "pump_usage": pump_usage
        }
        soil_result = {
            "date": date,
            "soil_mm": moisture,
            "irrigation_mm": Qout_mm,
        }

        tank_results.append(tank_result)
        soil_results.append(soil_result)

        water_level_previous = water_level_today
        moisture_previous = moisture_today

        water_level_today = W_t_final
        moisture_today = moisture

    return tank_results, soil_results
