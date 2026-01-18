import os.path
from os import makedirs

import numpy as np
import pandas as pd
from django.conf import settings
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = settings.BASE_DIR

DATA_DIR = BASE_DIR / "model_service"

X_feature_cols = [
    "soil_moisture",
    "water_level",
    "soil_moisture_prev",
    "water_level_prev",

    "day_of_year",
    "month",

    # "tank_capacity",

    "temp_today",
    "rain_today",

    "temp_tomorrow",
    "rain_tomorrow",

    # für Horizon h
    # "sum_rain_next_h_days",
    # "mean_temp_next_h_days",
    "irrigation_last_h_days",
    # "eto_next_h_days",

    # Plan
    # "planned_irrigation_next_h_days",
    "pump_usage",
    "calculated_total_l",
    "irrigation_today"
]

y_feature_cols = [
    "delta_water_level",
    "delta_soil_moisture",
    # "water_level_tomorrow", for production
    # "soil_moisture_tomorrow" for production
]


def get_model_type_and_params(model_type: str, scaler_unnecessary: bool):
    if scaler_unnecessary:
        model = LGBMRegressor(objective="regression", boosting_type="gbdt", metric="rmse", min_child_weight=1e-3,
                              subsample_freq=1, max_bin=255, n_jobs=1, min_split_gain=0.0)

        gpu = False
        if not gpu:
            params = {
                "n_estimators": [300],
                "learning_rate": [0.01],
                "num_leaves": [31, 63],
                "max_depth": [-1],
                "min_child_samples": [10],
                "subsample": [1.0],
                "colsample_bytree": [1.0],
                "reg_lambda": [0.0, 1.0],  # L2
                "reg_alpha": [0.0, 0.1],  # L1
            }
        else:
            params = {
                "n_estimators": [300, 600],
                "learning_rate": [0.01, 0.03, 0.1],
                "num_leaves": [31, 63],
                "max_depth": [-1, 10],
                "min_child_samples": [10, 30],
                "subsample": [0.7, 1.0],
                "colsample_bytree": [0.7, 1.0],
                "reg_lambda": [0.0, 1.0],  # L2
                "reg_alpha": [0.0, 0.1],  # L1
            }
        pipeline = model
    else:
        if model_type == "ElasticNet":
            model = ElasticNet(max_iter=10000, tol=1e-4)
            params = {
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "model__alpha": np.logspace(-4, 1, 10)  # logspace is better than arange for non-linear parameters
            }
        else:
            model = BayesianRidge(max_iter=300, compute_score=True)
            params = {
                "model__tol": np.logspace(-6, -1, 3),
                "model__alpha_1": np.logspace(-6, -1, 3),
                "model__alpha_2": np.logspace(-6, -1, 3),
                "model__lambda_1": np.logspace(-6, -1, 3),
                "model__lambda_2": np.logspace(-6, -1, 3)
            }

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    return pipeline, params


def get_model(len_dataset: int):
    n_splits = get_split_number(len_dataset)
    tss = TimeSeriesSplit(n_splits=n_splits)

    if len_dataset >= 300:
        model_type = "LGBMRegressor"
        model, params = get_model_type_and_params(model_type, True)
    else:
        if 50 < len_dataset < 300:
            model_type = "ElasticNet"
            model, params = get_model_type_and_params(model_type, False)
        else:
            model_type = "BayesianRidge"
            model, params = get_model_type_and_params(model_type, False)

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=tss,
        scoring="neg_mean_squared_error"
    )

    return model_type, grid


def training_info(len_x: int, len_y: int, model_type: str, best_params_water_level,
                  best_params_soil_moisture, final_r2_score: float):
    return {
        "len_x": len_x,
        "len_y": len_y,
        "model_type": model_type,
        "best_params_water_level": best_params_water_level,
        "best_params_soil_moisture": best_params_soil_moisture,
        "final_r2_score": final_r2_score
    }


def get_split_number(len_data, predict_horizon=7):
    # size of splits should be twice the size of horizon, default: horizon 7 days
    # -> because model should predict a time series of e.g. 7 days
    # and there are two types of validation error: 1. initial error (t -> t1), 2. cumulative error (t -> t7)
    # -> cumulative error shows the model drift
    min_split_size = predict_horizon * 2

    # TimeSeriesSplits splits data in k + 1 blocks with k=n_split, so make sure that max_splits are - 1
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


def train_model(use_synthetic_data: bool):
    df = pd.read_csv(
        os.path.join(DATA_DIR, "data/synthetic_training_data.csv" if use_synthetic_data else "data/training_data.csv"))
    X = df[X_feature_cols]
    y = df[y_feature_cols]

    len_X = len(X)
    len_y = len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Calculate residuals for training: residual = true_next - physics_next
    from .physics_model import TankPhysics, SoilPhysics
    from .config import (
        SOIL_K_IRRIG, SOIL_K_RAIN, SOIL_K_EVAP, 
        DEFAULT_TANK_CAPACITY, DEFAULT_PLANT_AREA
    )

    tank_phys = TankPhysics()
    soil_phys = SoilPhysics(k_irrig=SOIL_K_IRRIG, k_rain=SOIL_K_RAIN, k_evap=SOIL_K_EVAP)

    def compute_residuals(X_df, y_df):
        res_tank = []
        res_soil = []
        for (i, row), (_, y_row) in zip(X_df.iterrows(), y_df.iterrows()):
            # Physics next values
            p_tank_next, _ = tank_phys.update(
                tank_level_prev=row["water_level"],
                inflow_l=row["calculated_total_l"],
                outflow_l=row["irrigation_today"],
                tank_capacity=DEFAULT_TANK_CAPACITY
            )
            p_soil_next = soil_phys.compute_base(
                soil_moisture_prev=row["soil_moisture"],
                irrigation_mm=row["irrigation_today"] / DEFAULT_PLANT_AREA,
                rain_mm=row["rain_today"],
                temp_c=row["temp_today"]
            )
            # Clip soil physics to Match inference 0..100
            p_soil_next = max(0.0, min(100.0, p_soil_next))

            # True absolute next values
            t_tank_next = row["water_level"] + y_row["delta_water_level"]
            t_soil_next = row["soil_moisture"] + y_row["delta_soil_moisture"]

            res_tank.append(t_tank_next - p_tank_next)
            res_soil.append(t_soil_next - p_soil_next)
        
        return pd.DataFrame({
            "residual_water_level": res_tank,
            "residual_soil_moisture": res_soil
        }, index=X_df.index)

    y_train_res = compute_residuals(X_train, y_train)
    y_test_res = compute_residuals(X_test, y_test)

    model_type, model = get_model(len(X_train))

    multi_model = MultiOutputRegressor(model)

    multi_model.fit(X_train, y_train_res)

    best_params_water_level = multi_model.estimators_[0].best_params_
    best_params_soil_moisture = multi_model.estimators_[1].best_params_

    makedirs(os.path.join(DATA_DIR, "trained_models"), exist_ok=True)
    dump(multi_model, os.path.join(DATA_DIR,
                                   "trained_models/synthetic_model.pkl" if use_synthetic_data else "trained_models/model.pkl"))

    y_pred = multi_model.predict(X_test)
    final_r2_score = r2_score(y_test_res, y_pred, multioutput='uniform_average')

    return training_info(len_X, len_y, model_type, best_params_water_level, best_params_soil_moisture, final_r2_score)
