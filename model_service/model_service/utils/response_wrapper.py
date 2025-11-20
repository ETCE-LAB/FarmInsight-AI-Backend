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
            if float(row["qout_l"]) > 0.0:
                data = {
                    "timestamp": add_time_delta(row["date"]),
                    "value": float(row["qout_l"]),  # amount of water
                    "action": "watering"
                }
                print("data:", data)
                pts.append(data)
            else:
                continue
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


{"forecasts": [{"name": "tank_level", "values": [{"name": "best_case", "value": [
    {"timestamp": "2025-11-20T08:00:00Z", "value": 70.51430706282729},
    {"timestamp": "2025-11-21T08:00:00Z", "value": 70.51430706282729}]}, {"name": "average_case", "value": [
    {"timestamp": "2025-11-20T08:00:00Z", "value": 64.05891551166116},
    {"timestamp": "2025-11-21T08:00:00Z", "value": 64.05891551166116}]}, {"name": "worst_case", "value": [
    {"timestamp": "2025-11-20T08:00:00Z", "value": 56.11252396049503},
    {"timestamp": "2025-11-21T08:00:00Z", "value": 54.61252396049503}]}]}, {"name": "soil_moisture", "values": [
    {"name": "best_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 57.119},
                                    {"timestamp": "2025-11-21T08:00:00Z", "value": 57.119}]}, {"name": "average_case",
                                                                                               "value": [{
                                                                                                   "timestamp": "2025-11-20T08:00:00Z",
                                                                                                   "value": 51.91},
                                                                                                   {
                                                                                                       "timestamp": "2025-11-21T08:00:00Z",
                                                                                                       "value": 51.91}]},
    {"name": "worst_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 46.851000000000006},
                                     {"timestamp": "2025-11-21T08:00:00Z", "value": 47.001000000000005}]}]}],
 "actions": [{"name": "best_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 0.0},
                                             {"timestamp": "2025-11-21T08:00:00Z", "value": 0.0}]},
             {"name": "average_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 0.0},
                                                {"timestamp": "2025-11-21T08:00:00Z", "value": 0.0}]},
             {"name": "worst_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 1.5, "action": "watering"},
                                              {"timestamp": "2025-11-21T08:00:00Z", "value": 1.5,
                                               "action": "watering"}]}]

{"forecasts": [{"name": "tank_level", "values": [{"name": "best_case", "value": [
    {"timestamp": "2025-11-20T08:00:00Z", "value": 70.51430706282729},
    {"timestamp": "2025-11-21T08:00:00Z", "value": 70.51430706282729}]}, {"name": "average_case", "value": [
    {"timestamp": "2025-11-20T08:00:00Z", "value": 64.05891551166116},
    {"timestamp": "2025-11-21T08:00:00Z", "value": 64.05891551166116}]}, {"name": "worst_case", "value": [
    {"timestamp": "2025-11-20T08:00:00Z", "value": 56.11252396049503},
    {"timestamp": "2025-11-21T08:00:00Z", "value": 54.61252396049503}]}]}, {"name": "soil_moisture", "values": [
    {"name": "best_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 57.119},
                                    {"timestamp": "2025-11-21T08:00:00Z", "value": 57.119}]}, {"name": "average_case",
                                                                                               "value": [{
                                                                                                             "timestamp": "2025-11-20T08:00:00Z",
                                                                                                             "value": 51.91},
                                                                                                         {
                                                                                                             "timestamp": "2025-11-21T08:00:00Z",
                                                                                                             "value": 51.91}]},
    {"name": "worst_case", "value": [{"timestamp": "2025-11-20T08:00:00Z", "value": 46.851000000000006},
                                     {"timestamp": "2025-11-21T08:00:00Z", "value": 47.001000000000005}]}]}],


 "actions": [{"name": "best_case", "value": []}, {"name": "average_case", "value": []}, {"name": "worst_case",
                                                                                         "value": [{
                                                                                                       "timestamp": "2025-11-20T08:00:00Z",
                                                                                                       "value": 1.5,
                                                                                                       "action": "watering"},
                                                                                                   {
                                                                                                       "timestamp": "2025-11-21T08:00:00Z",
                                                                                                       "value": 1.5,
                                                                                                       "action": "watering"}]}]
