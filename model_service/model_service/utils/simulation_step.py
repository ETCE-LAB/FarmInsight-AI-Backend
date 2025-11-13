from typing import Any, Dict, Tuple


def run_forecast_day(
        water_level_previous: float,
        moisture_previous: float,

        date: str,
        max_temp: float,
        water_inflow: float,

        ml_model: Any,
        tank_capacity_liters: float,
        soil_threshold: float,
        max_irrigation_l_day: float,
        plant_area: float,
        # min_moisture: float,
        # max_moisture: float
) -> Tuple[float, float, Dict, Dict]:
    Qout_l = 0.0
    if moisture_previous < soil_threshold:
        Qout_l = max_irrigation_l_day

    Qout_l_final = min(Qout_l, water_level_previous + water_inflow)
    Qout_mm = Qout_l_final / plant_area

    features = [water_level_previous, moisture_previous, water_inflow, Qout_l_final, max_temp]

    predictions = ml_model.predict([features])
    W_t_ml = predictions[0][0]
    M_t_ml_final = predictions[0][1]

    W_t_final = max(0.0, min(tank_capacity_liters, W_t_ml))
    overflow_l = max(0.0, W_t_ml - tank_capacity_liters)

    tank_result = {
        "date": date, "tank_l": W_t_final, "qin_l": water_inflow,
        "qout_l": Qout_l_final, "overflow_l": overflow_l
    }
    soil_result = {
        "date": date, "soil_mm": M_t_ml_final,  # "soil_percent": M_percent,
        "irrigation_mm": Qout_mm,  # "et_mm_day": ET_mm,
        # "drainage_mm": drainage_mm
    }

    return W_t_final, M_t_ml_final, tank_result, soil_result
