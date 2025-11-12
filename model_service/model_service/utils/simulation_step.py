from typing import Any, Dict, Tuple

from ..utils.soil_moisture_calculator import update_soil_moisture, _calculate_et


def run_forecast_day(
        W_prev: float,
        M_prev: float,

        date: str,
        Tmax_c: float,
        Qin_l: float,

        ML_model: Any,
        tank_capacity_liters: float,
        soil_threshold: float,
        max_irrigation_l_day: float,
        A_soil_m2: float,
        min_moisture: float,
        max_moisture: float
) -> Tuple[float, float, Dict, Dict]:

    ET_mm = _calculate_et(
        Tmax_c, M_prev,
        min_moisture, max_moisture
    )
    M_after_et = M_prev - ET_mm

    Qout_l = 0.0
    if M_after_et < soil_threshold:
        Qout_l = max_irrigation_l_day

    Qout_l_final = min(Qout_l, W_prev + Qin_l)
    Qout_mm = Qout_l_final / A_soil_m2

    features = [W_prev, Qin_l, Qout_l_final, Tmax_c]

    W_t_ml_raw = ML_model.predict([features])
    W_t_ml = W_t_ml_raw[0]

    W_t_final = max(0.0, min(tank_capacity_liters, W_t_ml))
    overflow_l = max(0.0, W_t_ml - tank_capacity_liters)

    M_t_final, drainage_mm = update_soil_moisture(
        M_after_et,
        Qout_mm,
        min_moisture=min_moisture,
        max_moisture=max_moisture
    )
    M_percent = ((M_t_final - min_moisture) / (max_moisture - min_moisture)) * 100.0

    tank_result = {
        "date": date, "tank_l": W_t_final, "qin_l": Qin_l,
        "qout_l": Qout_l_final, "overflow_l": overflow_l
    }
    soil_result = {
        "date": date, "soil_mm": M_t_final, "soil_percent": M_percent,
        "et_mm_day": ET_mm, "irrigation_mm": Qout_mm,
        "drainage_mm": drainage_mm
    }

    return W_t_final, M_t_final, tank_result, soil_result
