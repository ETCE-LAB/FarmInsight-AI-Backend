from math import radians, cos, nan
from typing import Dict, List, Tuple, Optional, Union

import requests

latitude = 52.52
longitude = 13.40

face_area_m2 = 10.0
slope_deg = 30.0
precip_key = "rain_sum"

# in liter
W0_l = 250.0
capacity_l = 1000.0
daily_outflow_l = 15.0

# Constants for the model
ET_BASE_FACTOR = 0.15
MAX_MOISTURE_MM = 300.0
MIN_MOISTURE_MM = 50.0
ROOT_DEPTH_M = 0.3


def compute_greenhouse_roof_rain(
        forecast: Dict,
        *,
        face_azimuth_deg: float = 30.26, # Dach A looks to NorthEastNorth
        face_area_m2: float,
        slope_deg: float,
        precip_key: str = "rain_sum",
        wind_bias_strength: float = 0.35,
        wind_exposure_factors: Tuple[float, float] = (0.8, 0.9),
        runoff_coeff: float = 0.9,
        first_flush_loss: float = 0.03
) -> List[Dict]:
    daily = forecast.get("daily", {})
    print(daily)
    times: List[str] = daily.get("time") or []

    if not times:
        raise ValueError("Forecast JSON is missing the 'time' array.")

    num_days = len(times)

    rains: List[Optional[float]] = daily.get(precip_key) or []
    if len(rains) != num_days:
        if len(rains) < num_days:
            rains.extend([None] * (num_days - len(rains)))
        else:
            raise ValueError(f"Rain array length ({len(rains)}) exceeds time array length ({num_days}).")

    wind_from: List[Optional[float]] = daily.get("wind_direction_10m_dominant") or []
    if len(wind_from) != num_days:
        if len(wind_from) < num_days:
            wind_from.extend([None] * (num_days - len(wind_from)))
        else:
            raise ValueError(f"Wind direction array length ({len(wind_from)}) exceeds time array length ({num_days}).")

    alpha = radians(slope_deg)
    base = cos(alpha)
    beta_A = radians(face_azimuth_deg % 360)
    beta_B = radians((face_azimuth_deg + 180.0) % 360)

    exposure_A, exposure_B = wind_exposure_factors
    w = max(0.0, min(1.0, float(wind_bias_strength)))

    loss_factor_global = runoff_coeff * (1.0 - first_flush_loss)

    out: List[Dict] = []

    for i, date in enumerate(times):
        rain_mm_day = 0.0 if rains[i] is None else float(rains[i])
        phi_from = wind_from[i]

        if phi_from is None:
            cos_A = cos_B = 0.0
            w_eff = 0.0
            phi_from_val, phi_to_deg = nan, nan
        else:
            phi_from_val = float(phi_from)
            phi_to_deg = (phi_from_val + 180.0) % 360.0
            phi_to = radians(phi_to_deg)
            cos_A = cos(beta_A - phi_to)
            cos_B = cos(beta_B - phi_to)
            w_eff = w

        raw_share_A = max(0.0, 1.0 + w_eff * cos_A)
        raw_share_B = max(0.0, 1.0 + w_eff * cos_B)
        s_sum = raw_share_A + raw_share_B

        if s_sum > 1e-6:
            share_A = raw_share_A / s_sum
            share_B = raw_share_B / s_sum
        else:
            share_A = 0.5
            share_B = 0.5

        scale_A_out = share_A
        scale_B_out = share_B

        total_mm_on_roof = rain_mm_day * 2.0 * base

        mm_A_raw = total_mm_on_roof * share_A
        mm_B_raw = total_mm_on_roof * share_B

        mm_A = mm_A_raw * loss_factor_global * exposure_A
        mm_B = mm_B_raw * loss_factor_global * exposure_B

        # in m³
        m3_A = (mm_A * face_area_m2) / 1000.0
        m3_B = (mm_B * face_area_m2) / 1000.0

        out.append({
            "date": date,
            "rain_mm": rain_mm_day,
            "mm_on_A": mm_A,
            "mm_on_B": mm_B,
            "m3_on_A": m3_A,
            "m3_on_B": m3_B,
            "total_m3": m3_A + m3_B,
            "scale_A": scale_A_out,
            "scale_B": scale_B_out,
            "phi_from_deg": phi_from_val,
            "phi_to_deg": phi_to_deg,
            "model_type": "keep_mass",
            "cos_base": base,
            "applied_loss_global": loss_factor_global,
        })
    return out
