# model_trainer.py  (ersetzt/ergänzt deine train_model-Funktion)
import os.path
from os import makedirs

import numpy as np
import pandas as pd
from django.conf import settings
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_pinball_loss, make_scorer
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

BASE_DIR = settings.BASE_DIR
DATA_DIR = BASE_DIR / "model_service"

X_feature_cols = [
    "soil_moisture",
    "water_level",
    "soil_moisture_prev",
    "water_level_prev",
    "day_of_year",
    "month",
    "temp_today",
    "rain_today",
    "temp_tomorrow",
    "rain_tomorrow",
    "irrigation_last_h_days",
    "pump_usage",
    "calculated_total_l",
    "irrigation_today",
]

y_feature_cols = [
    "delta_water_level",
    "delta_soil_moisture",
]


def get_split_number(len_data, predict_horizon=7):
    min_split_size = predict_horizon * 2
    max_splits = len_data // min_split_size - 1
    max_splits = max(2, max_splits)

    if len_data >= 600:
        return min(10, max_splits)
    elif len_data >= 300:
        return min(8, max_splits)
    elif len_data >= 120:
        return min(5, max_splits)
    else:
        return min(3, max_splits)


def _compute_residual_targets(X_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    """
    residual = true_next_absolute - physics_next_absolute
    """
    from .physics_model import TankPhysics, SoilPhysics
    from .config import SOIL_K_IRRIG, SOIL_K_RAIN, SOIL_K_EVAP, DEFAULT_TANK_CAPACITY, DEFAULT_PLANT_AREA

    tank_phys = TankPhysics()
    soil_phys = SoilPhysics(k_irrig=SOIL_K_IRRIG, k_rain=SOIL_K_RAIN, k_evap=SOIL_K_EVAP)

    res_tank = []
    res_soil = []

    for (i, x), (_, y) in zip(X_df.iterrows(), y_df.iterrows()):
        p_tank_next, _ = tank_phys.update(
            tank_level_prev=float(x["water_level"]),
            inflow_l=float(x["calculated_total_l"]),
            outflow_l=float(x["irrigation_today"]),
            tank_capacity=float(DEFAULT_TANK_CAPACITY),
        )

        p_soil_next = soil_phys.compute_base(
            soil_moisture_prev=float(x["soil_moisture"]),
            irrigation_mm=float(x["irrigation_today"]) / float(DEFAULT_PLANT_AREA),
            rain_mm=float(x["rain_today"]),
            temp_c=float(x["temp_today"]),
        )
        p_soil_next = max(0.0, min(100.0, p_soil_next))

        t_tank_next = float(x["water_level"]) + float(y["delta_water_level"])
        t_soil_next = float(x["soil_moisture"]) + float(y["delta_soil_moisture"])

        res_tank.append(t_tank_next - p_tank_next)
        res_soil.append(t_soil_next - p_soil_next)

    return pd.DataFrame(
        {"residual_water_level": res_tank, "residual_soil_moisture": res_soil},
        index=X_df.index,
    )


def _pinball_score_func(y_true, y_pred, alpha):
    return -mean_pinball_loss(y_true, y_pred, alpha=alpha)


def train_quantile_models(use_synthetic_data: bool, quantiles=(0.1, 0.5, 0.9)):
    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "data/synthetic_training_data.csv" if use_synthetic_data else "data/training_data.csv",
        )
    )

    X = df[X_feature_cols]
    y = df[y_feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    y_train_res = _compute_residual_targets(X_train, y_train)
    y_test_res = _compute_residual_targets(X_test, y_test)

    n_splits = get_split_number(len(X_train))
    tss = TimeSeriesSplit(n_splits=n_splits)

    base_params = dict(
        boosting_type="gbdt",
        n_estimators=600,
        learning_rate=0.02,
        max_depth=-1,
        n_jobs=1,
        subsample_freq=1,
        max_bin=255,
        min_split_gain=0.0,
        min_child_weight=1e-3,
    )

    param_grid = {
        "num_leaves": [31],
        "min_child_samples": [20],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [0.0],
        "reg_alpha": [0.0],
    }

    makedirs(os.path.join(DATA_DIR, "trained_models"), exist_ok=True)

    results = {
        "rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "len_x": len(X_feature_cols),
        "len_y": len(y_feature_cols),
        "model_type": "MultiOutputRegressor(LightGBM)",
        "quantiles": {},
    }

    for q in quantiles:
        base = LGBMRegressor(objective="quantile", alpha=float(q), metric="quantile", **base_params)

        grid = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            cv=tss,
            scoring=make_scorer(_pinball_score_func, alpha=float(q)),
        )

        model = MultiOutputRegressor(grid)
        model.fit(X_train, y_train_res)

        y_pred = model.predict(X_test)

        # Pinball Loss pro Target
        pl_tank = mean_pinball_loss(y_test_res["residual_water_level"].values, y_pred[:, 0], alpha=float(q))
        pl_soil = mean_pinball_loss(y_test_res["residual_soil_moisture"].values, y_pred[:, 1], alpha=float(q))

        suffix = "synthetic" if use_synthetic_data else "real"
        out_path = os.path.join(DATA_DIR, "trained_models", f"{suffix}_quantile_q{int(q*100):02d}.pkl")
        dump(model, out_path)

        results["quantiles"][str(q)] = {
            "model_path": out_path,
            "pinball_loss_water_level": float(pl_tank),
            "pinball_loss_soil_moisture": float(pl_soil),
            "best_params_water_level": model.estimators_[0].best_params_,
            "best_params_soil_moisture": model.estimators_[1].best_params_,
        }

        if q == 0.5:
            q_res = results["quantiles"][str(q)]
            results["best_params_water_level"] = q_res["best_params_water_level"]
            results["best_params_soil_moisture"] = q_res["best_params_soil_moisture"]
            results["median_pinball_loss"] = (q_res["pinball_loss_water_level"] + q_res["pinball_loss_soil_moisture"]) / 2

    return results
