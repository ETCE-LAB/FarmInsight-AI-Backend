from typing import Dict, List, Tuple, Any

from ..utils.simulation_step import run_forecast_day


def case_value_multiplicator(scenario: str, value: float, value_name: str) -> float:
    if scenario == "best-case":
        if value_name == "temp_max":
            return value * 0.9
        else:
            return value * 1.1
    elif scenario == "worst-case":
        if value_name == "temp_max":
            return value * 1.1
        else:
            return value * 0.9
    else:
        return value


def run_forecast_period(
        forecast: Dict,
        inflow_data: List[Dict],
        ml_model: Any,

        initial_water_level: float,
        initial_moisture: float,
        tank_capacity_liters: float,
        soil_threshold: float,

        max_irrigation_l_day: float = 1.5,
        plant_area: float = 20.0,
        scenarios: list = []
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    tank_results: Dict[str, List[Dict]] = {}
    soil_results: Dict[str, List[Dict]] = {}

    daily = forecast.get("daily", {})
    times = daily.get("time", [])
    temp_max_list = daily.get("temperature_2m_max", [])
    if not (len(times) == len(temp_max_list) == len(inflow_data)):
        raise ValueError("Daten-Listen (time, Tmax, inflow) haben unterschiedliche Längen.")

    for scenario in scenarios:
        tank_results[scenario] = []
        soil_results[scenario] = []

        water_level_previous = initial_water_level
        water_level_previous = case_value_multiplicator(scenario, water_level_previous, "water_level_previous")
        moisture_previous = initial_moisture
        moisture_previous = case_value_multiplicator(scenario, moisture_previous, "moisture_previous")
        for i, date in enumerate(times):
            temp_max = temp_max_list[i] if temp_max_list[i] is not None else 10.0
            temp_max = case_value_multiplicator(scenario, temp_max, "temp_max")

            water_inflow = inflow_data[i]['total_m3'] * 1000.0
            water_inflow = case_value_multiplicator(scenario, water_inflow, "water_inflow")
            water_level, moisture, tank_result, soil_result = run_forecast_day(
                water_level_previous=water_level_previous,
                moisture_previous=moisture_previous,
                date=date,
                max_temp=temp_max,
                water_inflow=water_inflow,
                ml_model=ml_model,
                tank_capacity_liters=tank_capacity_liters,
                soil_threshold=soil_threshold,
                max_irrigation_l_day=max_irrigation_l_day,
                plant_area=plant_area
            )

            tank_results[scenario].append(tank_result)
            soil_results[scenario].append(soil_result)

            water_level_previous = water_level
            moisture_previous = moisture

    return tank_results, soil_results
