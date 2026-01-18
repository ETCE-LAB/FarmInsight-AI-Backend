"""
Simulation loop for water management forecast.

Refactored to use physics-first calculations with bounded ML residuals.
"""
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    ML_RESIDUAL_MAX_FRACTION, USE_CONFORMAL_UNCERTAINTY,
    SOIL_K_IRRIG, SOIL_K_RAIN, SOIL_K_EVAP,
    LITERS_PER_PUMP_LEVEL,
)
from .model_trainer import X_feature_cols
from .physics_model import SoilPhysics, TankPhysics
from .uncertainty import ConformalUncertainty, case_value_multiplicator, get_uncertainty_estimator


# Initialize physics models
tank_physics = TankPhysics()
soil_physics = SoilPhysics(
    k_irrig=SOIL_K_IRRIG,
    k_rain=SOIL_K_RAIN,
    k_evap=SOIL_K_EVAP,
)


def run_forecast_period(
    df_forecast: pd.DataFrame,
    forecast_days: int,
    ml_model: Any,
    tank_capacity: float,
    initial_water_level: float,
    initial_moisture: float,
    plant_area: float,
    scenario: str,
    plan: Tuple[int, ...],
    use_conformal_uncertainty: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    tank_results: List[Dict] = []
    soil_results: List[Dict] = []

    # Rolling irrigation history (liters)
    irrigation_history: List[float] = []

    df_forecast = df_forecast.sort_values("date").reset_index(drop=True)

    # State (today = current step, previous = t-1 for features)
    water_level_today = float(initial_water_level)
    moisture_today = float(initial_moisture)

    if use_conformal_uncertainty:
        water_level_previous = float(initial_water_level)
        moisture_previous = float(initial_moisture)
    else:
        water_level_previous = float(case_value_multiplicator(scenario, initial_water_level, "water_level_previous"))
        moisture_previous = float(case_value_multiplicator(scenario, initial_moisture, "soil_moisture_previous"))

    uncertainty = get_uncertainty_estimator(use_conformal=use_conformal_uncertainty)

    # Iterate over forecast (cap to forecast_days if df is longer)
    for i, row in df_forecast.iloc[:forecast_days].iterrows():
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

        # Apply legacy scenario multipliers ONLY when conformal is OFF
        if not use_conformal_uncertainty:
            temp_today = float(case_value_multiplicator(scenario, temp_today, "temp_today"))
            rain_today = float(case_value_multiplicator(scenario, rain_today, "rain_today"))
            temp_tomorrow = float(case_value_multiplicator(scenario, temp_tomorrow, "temp_tomorrow"))
            rain_tomorrow = float(case_value_multiplicator(scenario, rain_tomorrow, "rain_tomorrow"))
            water_inflow = float(case_value_multiplicator(scenario, water_inflow, "inflow_l"))

        day_of_year = row["date"].timetuple().tm_yday
        month = row["date"].month

        pumps_today = int(plan[i]) if i < len(plan) else (int(plan[-1]) if plan else 0)

        # Outflow constraint uses CURRENT tank level (today)
        Qout_l = pumps_today * LITERS_PER_PUMP_LEVEL
        max_available = tank_physics.compute_max_outflow(water_level_today, water_inflow)
        Qout_l_final = min(Qout_l, max_available)
        Qout_mm = Qout_l_final / plant_area

        # Feature: sum of last H days (excluding "today", because we append after building features)
        irrigation_last_h_days = sum(irrigation_history)

        # Build feature row (stable feature interface)
        feature_row = {
            "soil_moisture": moisture_today,
            "water_level": water_level_today,
            "soil_moisture_prev": moisture_previous,
            "water_level_prev": water_level_previous,
            "day_of_year": day_of_year,
            "month": month,
            "temp_today": temp_today,
            "rain_today": rain_today,
            "temp_tomorrow": temp_tomorrow,
            "rain_tomorrow": rain_tomorrow,
            "irrigation_last_h_days": irrigation_last_h_days,
            "pump_usage": pumps_today,
            "calculated_total_l": water_inflow,
            "irrigation_today": Qout_l_final,
        }

        X_infer = pd.DataFrame([feature_row])[X_feature_cols]
        pred = ml_model.predict(X_infer)
        ml_tank_res = float(pred[0][0])
        ml_soil_res = float(pred[0][1])

        # Clamp ML residuals
        max_tank_corr = ML_RESIDUAL_MAX_FRACTION * tank_capacity
        clamped_tank_res = max(-max_tank_corr, min(max_tank_corr, ml_tank_res))

        # Physics-first tank update uses CURRENT tank level (today)
        raw_tank = water_level_today + water_inflow - Qout_l_final
        new_tank_val = max(0.0, min(tank_capacity, raw_tank + clamped_tank_res))
        overflow_l = max(0.0, raw_tank - tank_capacity)

        # Physics-first soil update uses CURRENT soil moisture (today)
        soil_base = soil_physics.update(
            soil_moisture_prev=moisture_today,
            irrigation_mm=Qout_mm,
            rain_mm=rain_today,
            temp_c=temp_today,
        )
        max_soil_corr = ML_RESIDUAL_MAX_FRACTION * max(1.0, abs(soil_base))
        clamped_soil_res = max(-max_soil_corr, min(max_soil_corr, ml_soil_res))
        moisture_pred = max(0.0, min(100.0, soil_base + clamped_soil_res))

        # Apply uncertainty ONLY when conformal is ON
        if use_conformal_uncertainty:
            water_level_next = uncertainty.apply_to_tank_prediction(new_tank_val, scenario, tank_capacity)
            moisture_next = uncertainty.apply_to_soil_prediction(moisture_pred, scenario)
        else:
            water_level_next = new_tank_val
            moisture_next = moisture_pred

        tank_result = {
            "date": date,
            "tank_l": float(water_level_next),
            "qin_l": float(water_inflow),
            "qout_l": float(Qout_l_final),
            "overflow_l": float(overflow_l),
            "pump_usage": int(pumps_today),
        }
        soil_result = {
            "date": date,
            "soil_mm": float(moisture_next),
            "irrigation_mm": float(Qout_mm),
        }

        tank_results.append(tank_result)
        soil_results.append(soil_result)

        # Update irrigation history window AFTER using it for features
        irrigation_history.append(float(Qout_l_final))
        if len(irrigation_history) > forecast_days:
            irrigation_history.pop(0)

        # Shift state: today becomes previous, next becomes today
        water_level_previous = water_level_today
        moisture_previous = moisture_today
        water_level_today = float(water_level_next)
        moisture_today = float(moisture_next)

    return tank_results, soil_results


def simulate_single_day(
    tank_level_prev: float,
    moisture_prev: float,
    tank_level_pp: float,
    moisture_pp: float,
    action: int,
    day_data: pd.Series,
    next_day_data: Optional[pd.Series],
    ml_model: Any,
    tank_capacity: float,
    plant_area: float,
    scenario: str,
    use_conformal_uncertainty: bool = True,
    uncertainty_estimator: Optional[ConformalUncertainty] = None,
    irrigation_history: Optional[List[float]] = None,
) -> Tuple[Dict, Dict, float, float]:
    """
    Simulate a single day for beam search.
    
    This is a lighter-weight version of run_forecast_period for single-day
    simulation during beam search expansion.
    
    Returns:
        (tank_result, soil_result, new_tank_level, new_soil_moisture)
    """
    date = day_data["date"].strftime("%Y-%m-%d") if hasattr(day_data["date"], "strftime") else str(day_data["date"])
    temp_today = float(day_data["Tmax_c"])
    rain_today = float(day_data["rain_sum"])
    water_inflow = float(day_data["total_m3"] * 1000.0)

    if next_day_data is not None:
        temp_tomorrow = float(next_day_data["Tmax_c"])
        rain_tomorrow = float(next_day_data["rain_sum"])
    else:
        temp_tomorrow = temp_today
        rain_tomorrow = rain_today

    # Scenario Adjustments (Legacy Multiplier approach)
    # ONLY if conformal uncertainty is NOT used
    if not use_conformal_uncertainty:
        rain_today = case_value_multiplicator(scenario, rain_today, "rain_today")
        temp_today = case_value_multiplicator(scenario, temp_today, "temp_today")
        temp_tomorrow = case_value_multiplicator(scenario, temp_tomorrow, "temp_tomorrow")
        rain_tomorrow = case_value_multiplicator(scenario, rain_tomorrow, "rain_tomorrow")
        water_inflow = case_value_multiplicator(scenario, water_inflow, "inflow_l")

    day_of_year = day_data["date"].timetuple().tm_yday if hasattr(day_data["date"], "timetuple") else 1
    month = day_data["date"].month if hasattr(day_data["date"], "month") else 1

    # Calculate outflow
    Qout_l = action * LITERS_PER_PUMP_LEVEL
    max_available = tank_physics.compute_max_outflow(tank_level_prev, water_inflow)
    Qout_l_final = min(Qout_l, max_available)
    Qout_mm = Qout_l_final / plant_area

    # Dynamic features for the model
    # Use fixed window H=7
    hist = irrigation_history or []
    irrigation_last_h_days = sum(hist[-7:])

    # Build feature row
    feature_row = {
        "soil_moisture": moisture_prev,
        "water_level": tank_level_prev,
        "soil_moisture_prev": moisture_pp,
        "water_level_prev": tank_level_pp,
        "day_of_year": day_of_year,
        "month": month,
        "temp_today": temp_today,
        "rain_today": rain_today,
        "temp_tomorrow": temp_tomorrow,
        "rain_tomorrow": rain_tomorrow,
        "irrigation_last_h_days": irrigation_last_h_days,
        "pump_usage": action,
        "calculated_total_l": water_inflow,
        "irrigation_today": Qout_l_final,
    }
    # ML Predictions (Residuals)
    X_infer = pd.DataFrame([feature_row])[X_feature_cols]
    pred = ml_model.predict(X_infer)
    ml_tank_res = float(pred[0][0])
    ml_soil_res = float(pred[0][1])

    # ML Tank residual application with clamping
    max_tank_corr = ML_RESIDUAL_MAX_FRACTION * tank_capacity
    clamped_tank_res = max(-max_tank_corr, min(max_tank_corr, ml_tank_res))

    # Physics-first tank calculation (Mass Balance)
    raw_tank = tank_level_prev + water_inflow - Qout_l_final
    new_tank = max(0.0, min(tank_capacity, raw_tank + clamped_tank_res))
    overflow_l = max(0.0, raw_tank - tank_capacity)

    # Physics-first soil calculation with bounded ML residual
    soil_base = soil_physics.update(
        soil_moisture_prev=moisture_prev,
        irrigation_mm=Qout_mm,
        rain_mm=rain_today,
        temp_c=temp_today
    )
    
    max_soil_corr = ML_RESIDUAL_MAX_FRACTION * max(1.0, abs(soil_base))
    clamped_soil_res = max(-max_soil_corr, min(max_soil_corr, ml_soil_res))
    new_soil = max(0.0, min(100.0, soil_base + clamped_soil_res))

    # Apply uncertainty adjustments
    uncertainty = uncertainty_estimator or get_uncertainty_estimator(use_conformal=use_conformal_uncertainty)
    
    # We apply uncertainty only to output predictions if conformal is used
    if use_conformal_uncertainty:
        new_tank = uncertainty.apply_to_tank_prediction(new_tank, scenario, tank_capacity)
        new_soil = uncertainty.apply_to_soil_prediction(new_soil, scenario)

    tank_result = {
        "date": date,
        "tank_l": new_tank,
        "qin_l": water_inflow,
        "qout_l": Qout_l_final,
        "overflow_l": overflow_l,
        "pump_usage": action,
    }

    soil_result = {
        "date": date,
        "soil_mm": new_soil,
        "irrigation_mm": Qout_mm,
    }

    return tank_result, soil_result, new_tank, new_soil
