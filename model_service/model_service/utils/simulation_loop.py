"""
Simulation loop for water management forecast.

Refactored to use physics-first calculations with bounded ML residuals.
"""
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import (
    SOIL_K_IRRIG,
    SOIL_K_RAIN,
    SOIL_K_EVAP,
    ML_RESIDUAL_MAX_FRACTION,
    LITERS_PER_PUMP_LEVEL,
)
from .model_trainer import X_feature_cols
from .physics_model import TankPhysics, SoilPhysics
from .uncertainty import case_value_multiplicator, get_uncertainty_estimator


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
    """
    Run forecast simulation for a given irrigation plan.
    
    Args:
        df_forecast: Weather forecast DataFrame
        forecast_days: Number of days to forecast
        ml_model: Trained ML model
        tank_capacity: Tank capacity in liters
        initial_water_level: Starting water level
        initial_moisture: Starting soil moisture
        plant_area: Plant area in m²
        scenario: Scenario name (best-case, average-case, worst-case)
        plan: Tuple of pump levels for each day (0, 1, or 2)
        use_conformal_uncertainty: Whether to use conformal prediction for scenarios
        
    Returns:
        (tank_results, soil_results) lists of daily dictionaries
    """
    tank_results: List[Dict] = []
    soil_results: List[Dict] = []

    df_forecast = df_forecast.sort_values("date").reset_index(drop=True)

    # Initialize state with scenario adjustments
    water_level_today = initial_water_level
    water_level_previous = case_value_multiplicator(
        scenario, initial_water_level, "water_level_previous"
    )
    moisture_today = initial_moisture
    moisture_previous = case_value_multiplicator(
        scenario, initial_moisture, "soil_moisture_previous"
    )

    irrigation_history: List[float] = [0.0] * forecast_days

    # Get uncertainty estimator
    uncertainty = get_uncertainty_estimator(use_conformal=use_conformal_uncertainty)

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

        # Apply scenario adjustments to inputs (legacy behavior, kept for compatibility)
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
            pumps_today = plan[-1] if plan else 0

        # Calculate outflow with physics constraint
        Qout_l = pumps_today * LITERS_PER_PUMP_LEVEL
        max_available = tank_physics.compute_max_outflow(water_level_previous, water_inflow)
        Qout_l_final = min(Qout_l, max_available)
        Qout_mm = Qout_l_final / plant_area

        irrigation_history = irrigation_history[1:] + [Qout_l_final]
        irrigation_last_h_days = sum(irrigation_history)

        pump_usage = pumps_today
        irrigation_today = Qout_l_final

        # Build feature row for ML model (unchanged feature interface!)
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
            "pump_usage": pump_usage,
            "calculated_total_l": water_inflow,
            "irrigation_today": irrigation_today,
        }

        X_infer = pd.DataFrame([feature_row])[X_feature_cols]
        pred = ml_model.predict(X_infer)
        
        # ML predictions (used as residuals now)
        ml_tank_delta = float(pred[0][0])
        ml_soil_residual = float(pred[0][1])

        # =====================================================================
        # PHYSICS-FIRST TANK CALCULATION (Mass Balance)
        # =====================================================================
        W_t_final, overflow_l = tank_physics.update(
            tank_level_prev=water_level_previous,
            inflow_l=water_inflow,
            outflow_l=Qout_l_final,
            tank_capacity=tank_capacity,
        )
        
        # Apply conformal uncertainty adjustment to tank prediction
        W_t_final = uncertainty.apply_to_tank_prediction(
            W_t_final, scenario, tank_capacity
        )

        # =====================================================================
        # PHYSICS-FIRST SOIL CALCULATION (Bucket/ET0-proxy with bounded ML)
        # =====================================================================
        moisture = soil_physics.update(
            soil_moisture_prev=moisture_previous,
            irrigation_mm=Qout_mm,
            rain_mm=rain_today,
            temp_c=temp_today,
            ml_residual=ml_soil_residual,
            max_residual_fraction=ML_RESIDUAL_MAX_FRACTION,
        )
        
        # Apply conformal uncertainty adjustment to soil prediction
        moisture = uncertainty.apply_to_soil_prediction(
            moisture, scenario
        )

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

        # Update state for next iteration
        water_level_previous = water_level_today
        moisture_previous = moisture_today

        water_level_today = W_t_final
        moisture_today = moisture

    return tank_results, soil_results


def simulate_single_day(
        tank_level_prev: float,
        moisture_prev: float,
        action: int,
        day_data: pd.Series,
        next_day_data: pd.Series,
        ml_model: Any,
        tank_capacity: float,
        plant_area: float,
        scenario: str,
        irrigation_history: List[float],
        use_conformal_uncertainty: bool = True,
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

    # Apply scenario adjustments
    temp_today = case_value_multiplicator(scenario, temp_today, "temp_today")
    rain_today = case_value_multiplicator(scenario, rain_today, "rain_today")
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

    irrigation_last_h_days = sum(irrigation_history[-7:]) if irrigation_history else 0.0

    # Build feature row
    feature_row = {
        "soil_moisture": moisture_prev,
        "water_level": tank_level_prev,
        "soil_moisture_prev": moisture_prev,
        "water_level_prev": tank_level_prev,
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

    X_infer = pd.DataFrame([feature_row])[X_feature_cols]
    pred = ml_model.predict(X_infer)
    ml_soil_residual = float(pred[0][1])

    # Physics-first tank calculation
    new_tank, overflow_l = tank_physics.update(
        tank_level_prev=tank_level_prev,
        inflow_l=water_inflow,
        outflow_l=Qout_l_final,
        tank_capacity=tank_capacity,
    )

    # Physics-first soil calculation with bounded ML residual
    new_soil = soil_physics.update(
        soil_moisture_prev=moisture_prev,
        irrigation_mm=Qout_mm,
        rain_mm=rain_today,
        temp_c=temp_today,
        ml_residual=ml_soil_residual,
        max_residual_fraction=ML_RESIDUAL_MAX_FRACTION,
    )

    # Apply uncertainty adjustments
    uncertainty = get_uncertainty_estimator(use_conformal=use_conformal_uncertainty)
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
