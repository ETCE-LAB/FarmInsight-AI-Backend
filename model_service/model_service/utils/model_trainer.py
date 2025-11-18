from os import makedirs

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_service.model_service.utils.data_processor import prepare_data_for_training

X_feature_cols = [
    # Aktueller Zustand
    "soil_moisture",
    "water_level",
    "soil_moisture_prev",
    "water_level_prev",

    # Zeit
    "day_of_year",
    "month",

    # Daten des Standortes
    "tank_capacity",

    # Heute:
    "temp_today",
    "rain_today",

    # Morgen:
    "temp_tomorrow",
    "rain_tomorrow",

    # Wettervorhersage (für Horizon h)
    # "sum_rain_next_h_days",
    # "mean_temp_next_h_days",
    "irrigation_last_h_days",
    # "sum_ET0_next_h_days",

    # Bewässerungsplan
    # "planned_irrigation_next_h_days",
    "pump_usage"
]
# rain_d1, rain_d2, …, rain_dh
# temp_mean_d1, temp_mean_d2, ...

y_feature_cols = [
    # "soil_moisture_in_h_days",
    # "water_level_in_h_days",
    "water_level_tomorrow",
    "soil_moisture_tomorrow"
]


def get_model_type_and_params(model_type: str, scaler_unnecessary: bool):
    pipeline = None
    if scaler_unnecessary:
        model = LGBMRegressor(objective="regression", boosting_type="gbdt", metric="rmse", min_child_weight=1e-3,
                              subsample_freq=1, max_bin=255, n_jobs=1)
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


def train_model():
    prepare_data_for_training()

    df = pd.read_csv("../data/training_data.csv")
    X = df[X_feature_cols]
    y = df[y_feature_cols]

    len_X = len(X)
    len_y = len(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model_type, model = get_model(len(X_train))

    multi_model = MultiOutputRegressor(model)

    multi_model.fit(X_train, y_train)

    best_params_water_level = multi_model.estimators_[0].best_params_
    best_params_soil_moisture = multi_model.estimators_[1].best_params_

    makedirs("../trained_models", exist_ok=True)
    dump(multi_model, "../trained_models/final_model.pkl")

    y_pred = multi_model.predict(X_test)
    final_r2_score = r2_score(y_test, y_pred, multioutput='uniform_average')

    print("Training completed:",
          training_info(len_X, len_y, model_type, best_params_water_level, best_params_soil_moisture, final_r2_score)
          )

    return len_X, len_y, model_type, best_params_water_level, best_params_soil_moisture, final_r2_score


train_model()
