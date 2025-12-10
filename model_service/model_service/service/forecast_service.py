import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any

from joblib import load

from ..utils.inference_processor import prepare_data_for_prediction
from ..utils.simulation_loop import run_forecast_period

BASE_DIR = Path(__file__).resolve().parent
TRAINED_MODELS_DIR = BASE_DIR.parent / "trained_models"

FACE_AZIMUTH_DEG = 30.0
FACE_AREA_M2 = 6.668
SLOPE_DEG = 30.26

# estimated plant area
A_SOIL_M2 = 8.0

MAX_IRRIGATION_L_DAY = 1.5


def score_plan(
        tank_results: List[Dict],
        soil_results: List[Dict],
        soil_threshold: float,
        scenario: str,
        alpha: float = 10.0,
        beta: float = 0.1,
        gamma: float = 0.3,
        delta: float = 1.0,
) -> float:

    moisture_deficit = 0.0
    total_irrigation = 0.0
    total_overflow = 0.0

    for t, s in zip(tank_results, soil_results):
        soil_mm = s["soil_mm"]
        irrigation_mm = s["irrigation_mm"]
        overflow_l = t["overflow_l"]

        moisture_deficit += max(0.0, soil_threshold - soil_mm)
        total_irrigation += irrigation_mm
        total_overflow += overflow_l

    final_tank = tank_results[-1]["tank_l"] if soil_results else 0.0

    return (
            - alpha * moisture_deficit
            - beta * total_irrigation
            - delta * total_overflow
            + gamma * final_tank
    )


def model_forecast(
        latitude: float,
        longitude: float,
        forecast_days: int,
        initial_water_level: float,
        tank_capacity_liters: float,
        soil_threshold: float,
        scenarios: list,
        start_soil_moisture: float,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    ml_model = load(TRAINED_MODELS_DIR / "synthetic_model.pkl")

    forecast, inflow_data, df_forecast = prepare_data_for_prediction(
        latitude=latitude,
        longitude=longitude,
        forecast_days=forecast_days
    )

    all_best_tank_results: Dict[str, List[Dict]] = {}
    all_best_soil_results: Dict[str, List[Dict]] = {}

    plans = list(itertools.product([0, 1, 2], repeat=forecast_days))

    for scenario in scenarios:
        best_score: float | None = None
        best_tank_results: Dict[str, List[Dict]] | None = None
        best_soil_results: Dict[str, List[Dict]] | None = None

        for plan in plans:
            tank_results, soil_results = run_forecast_period(
                df_forecast=df_forecast,
                ml_model=ml_model,
                forecast_days=forecast_days,
                initial_water_level=initial_water_level,
                initial_moisture=start_soil_moisture,
                plant_area=A_SOIL_M2,
                scenario=scenario,
                tank_capacity=tank_capacity_liters,
                plan=plan
            )

            score = score_plan(tank_results, soil_results, soil_threshold, scenario=scenario)
            if best_score is None or score > best_score:
                best_score = score
                best_tank_results = tank_results
                best_soil_results = soil_results

        if best_tank_results is not None and best_soil_results is not None:
            all_best_tank_results[scenario] = best_tank_results
            all_best_soil_results[scenario] = best_soil_results

    return all_best_tank_results, all_best_soil_results
