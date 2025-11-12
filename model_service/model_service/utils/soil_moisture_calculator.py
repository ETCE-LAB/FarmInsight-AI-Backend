from typing import Tuple


def update_soil_moisture(
        M_after_et: float,
        Qout_mm: float,
        *,
        min_moisture: float = 60.0,
        max_moisture: float = 135.0
) -> Tuple[float, float]:
    """
    params:
        M_after_et: in mm
        Qout_mm: in mm
        min_moisture_mm: Welkepunkt (physikalische Untergrenze).
        max_moisture_mm: Feldkapazität (physikalische Obergrenze).

    Returns:
        M_t_final: final soil moisture at the end of the day in mm
        drainage: drainage in mm
    """

    M_raw_next = M_after_et + Qout_mm

    drainage = max(0.0, M_raw_next - max_moisture)

    M_t_final = M_raw_next - drainage
    M_t_final = max(min_moisture, M_t_final)

    return M_t_final, drainage


def _calculate_et(
        Tmax_c: float,
        M_prev: float,
        min_moisture_mm: float,
        max_moisture_mm: float,
        ET_MAX_COLD_SEASON: float = 3.0,
        ET_T_FACTOR: float = 0.15
) -> float:
    ET_mm = Tmax_c * ET_T_FACTOR
    ET_mm = min(ET_MAX_COLD_SEASON, ET_mm)
    ET_mm = max(0.0, ET_mm)

    available_M_ratio = (M_prev - min_moisture_mm) / (max_moisture_mm - min_moisture_mm)
    available_M_ratio = max(0.0, min(1.0, available_M_ratio))
    ET_mm *= available_M_ratio

    return ET_mm
