from typing import Dict, List, Tuple, Any

from ..utils.simulation_step import run_forecast_day


def run_forecast_period(
        forecast: Dict,
        inflow_data: List[Dict],
        ML_model: Any,

        W0_l: float,
        M0_mm: float,
        tank_capacity_liters: float,
        soil_threshold: float,

        max_irrigation_l_day: float = 15.0,
        A_soil_m2: float = 20.0,
        max_moisture: float = 300.0,
        min_moisture: float = 50.0,
        scenarios: list = []
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:

    tank_results: Dict[str, List[Dict]] = {}
    soil_results: Dict[str, List[Dict]] = {}

    daily = forecast.get("daily", {})
    times = daily.get("time", [])
    print(times, type(times))
    temp_max_list = daily.get("temperature_2m_max", [])

    if not (len(times) == len(temp_max_list) == len(inflow_data)):
        raise ValueError("Daten-Listen (time, Tmax, inflow) haben unterschiedliche Längen.")

    for scenario in scenarios:
        tank_results[scenario] = []
        soil_results[scenario] = []

        W_prev = W0_l
        M_prev = M0_mm
        for i, date in enumerate(times):
            Tmax_c = temp_max_list[i] if temp_max_list[i] is not None else 10.0

            Qin_l = inflow_data[i]['total_m3'] * 1000.0

            W_t_final, M_t_final, tank_result, soil_result = run_forecast_day(
                W_prev=W_prev,
                M_prev=M_prev,
                date=date,
                Tmax_c=Tmax_c,
                Qin_l=Qin_l,
                ML_model=ML_model,
                tank_capacity_liters=tank_capacity_liters,
                soil_threshold=soil_threshold,
                max_irrigation_l_day=max_irrigation_l_day,
                A_soil_m2=A_soil_m2,
                min_moisture=min_moisture,
                max_moisture=max_moisture
            )

            tank_results[scenario].append(tank_result)
            soil_results[scenario].append(soil_result)

            W_prev = W_t_final
            M_prev = M_t_final

    return tank_results, soil_results
