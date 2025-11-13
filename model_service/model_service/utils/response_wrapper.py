from __future__ import annotations

import pandas as pd


def response_wrapper(plans: dict) -> dict:
    print("plans:", plans)

    def add_time_delta(date):
        timestamp = pd.to_datetime(date)
        timestamp = timestamp.tz_localize("Europe/Berlin")
        timestamp = timestamp + pd.Timedelta(hours=9)
        timestamp = timestamp.tz_convert("UTC")
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def series_points(value_key: str, case_key: str, field: str):
        return [
            {"timestamp": add_time_delta(row["date"]), "value": float(row[field])}
            for row in plans[value_key][case_key]
        ]

    def action_points(value_key: str, case_key: str):
        pts = []
        for row in plans[value_key][case_key]:
            data = {
                "timestamp": add_time_delta(row["date"]),
                "value": float(row["qout_l"])  # amount of water
            }

            if float(row["qout_l"]) > 0.0:
                data["action"] = "watering"
            print("data:", data)
            pts.append(data)
        return pts

    forecasts = [
        {
            "name": "tank_level",
            "values": [
                {"name": "best_case", "value": series_points("tank_forecast", "best-case", "tank_l")},
                {"name": "average_case", "value": series_points("tank_forecast", "average-case", "tank_l")},
                {"name": "worst_case", "value": series_points("tank_forecast", "worst-case", "tank_l")},
            ],
        },
        {
            "name": "soil_moisture",
            "values": [
                {"name": "best_case", "value": series_points("soil_forecast", "best-case", "soil_mm")},
                {"name": "average_case", "value": series_points("soil_forecast", "average-case", "soil_mm")},
                {"name": "worst_case", "value": series_points("soil_forecast", "worst-case", "soil_mm")},
            ],
        },
    ]

    actions = [
        {"name": "best_case", "value": action_points("tank_forecast", "best-case")},
        {"name": "average_case", "value": action_points("tank_forecast", "average-case")},
        {"name": "worst_case", "value": action_points("tank_forecast", "worst-case")},
    ]

    return {"forecasts": forecasts, "actions": actions}
