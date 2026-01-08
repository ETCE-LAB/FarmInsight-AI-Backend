"""
Energy Model Trainer for FarmInsight

Trains an ML model (XGBoost or Random Forest) to predict battery state of charge
based on power consumption, weather conditions, and temporal features.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "trained_models"

# Feature columns for training
FEATURE_COLUMNS = [
    # Temporal features
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'month_sin', 'month_cos',
    # Current state
    'power_watts',
    # Weather features
    'shortwave_radiation', 'cloud_cover_pct',
    # Lagged capacity features
    'capacity_wh_lag_1h', 'capacity_wh_lag_3h', 
    'capacity_wh_lag_6h', 'capacity_wh_lag_12h', 'capacity_wh_lag_24h',
    # Rolling features
    'capacity_wh_rolling_mean_6h', 'capacity_wh_rolling_mean_24h',
    'power_watts_rolling_mean_6h', 'power_watts_rolling_mean_24h'
]

# Target column
TARGET_COLUMN = 'capacity_next_hour'


def load_training_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load prepared training data from CSV.
    
    :param filepath: Path to training CSV
    :return: Training DataFrame
    """
    if filepath is None:
        filepath = DATA_DIR / "energy_training_data.csv"
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def get_available_features(df: pd.DataFrame) -> list:
    """
    Get list of available feature columns from DataFrame.
    
    :param df: Training DataFrame
    :return: List of available feature column names
    """
    available = []
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            available.append(col)
        else:
            print(f"  Warning: Feature '{col}' not found in data")
    return available


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare feature matrix and target vector.
    
    :param df: Training DataFrame
    :return: Tuple of (X, y, feature_names)
    """
    available_features = get_available_features(df)
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    X = df[available_features].values
    y = df[TARGET_COLUMN].values
    
    # Remove any remaining NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    return X, y, available_features


def train_energy_model(use_gradient_boosting: bool = True) -> Dict[str, Any]:
    """
    Train energy prediction model.
    
    :param use_gradient_boosting: Use GradientBoosting (True) or RandomForest (False)
    :return: Dictionary with training results and metrics
    """
    print("=" * 60)
    print("Energy Model Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading training data...")
    try:
        df = load_training_data()
        print(f"   Loaded {len(df)} records")
    except FileNotFoundError:
        print("   Training data not found. Running data preparation...")
        from .energy_data_processor import prepare_energy_training_data
        df, _ = prepare_energy_training_data()
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y, feature_names = prepare_features_and_target(df)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Features used: {feature_names}")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (time-series aware: use later data for testing)
    print("\n4. Splitting data (80/20)...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Select and train model
    print("\n5. Training model...")
    if use_gradient_boosting:
        model_type = "GradientBoostingRegressor"
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    else:
        model_type = "RandomForestRegressor"
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    print(f"   Model type: {model_type}")
    model.fit(X_train, y_train)
    print("   Training complete!")
    
    # Evaluate model
    print("\n6. Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"   Train RMSE: {train_rmse:.2f} Wh")
    print(f"   Test RMSE:  {test_rmse:.2f} Wh")
    print(f"   Train R²:   {train_r2:.4f}")
    print(f"   Test R²:    {test_r2:.4f}")
    
    # Feature importance
    print("\n7. Feature importance:")
    importance = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5]:
        print(f"   {name}: {imp:.4f}")
    
    # Save model and scaler
    print("\n8. Saving model...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = MODELS_DIR / "energy_model.pkl"
    scaler_path = MODELS_DIR / "energy_scaler.pkl"
    metadata_path = MODELS_DIR / "energy_model_metadata.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   Model saved to: {model_path}")
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   Scaler saved to: {scaler_path}")
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "feature_names": feature_names,
        "training_date": datetime.now().isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "feature_importance": dict(zip(feature_names, importance.tolist()))
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return {
        "model_type": model_type,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "features_used": len(feature_names),
        "model_path": str(model_path)
    }


def load_energy_model() -> Tuple[Any, StandardScaler, Dict]:
    """
    Load trained energy model, scaler, and metadata.
    
    :return: Tuple of (model, scaler, metadata)
    """
    model_path = MODELS_DIR / "energy_model.pkl"
    scaler_path = MODELS_DIR / "energy_scaler.pkl"
    metadata_path = MODELS_DIR / "energy_model_metadata.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Energy model not found at {model_path}. Run training first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, metadata


if __name__ == "__main__":
    result = train_energy_model()
    print("\nTraining Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
