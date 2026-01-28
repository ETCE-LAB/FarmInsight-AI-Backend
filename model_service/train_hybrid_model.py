"""
HYBRID ML Training: Absolute prediction + Weather-based adjustment

The problem:
- Model v1: 96% capacity_wh, ignores weather → works but doesn't use weather
- Model v2: Predicts change, uses weather → fails because of error accumulation

Solution: HYBRID approach
- Base model predicts capacity (stable, low error)
- Weather model predicts ADJUSTMENT during solar hours
- Combine: final = base_prediction + weather_adjustment * solar_window
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from pathlib import Path
from datetime import datetime

# Paths
DATA_PATH = Path(__file__).parent / "model_service" / "data" / "enhanced_training_data.csv"
MODEL_PATH = Path(__file__).parent / "model_service" / "trained_models" / "energy_forecast_models_v3.pkl"

print("=" * 70)
print("HYBRID ML MODEL - BASE + WEATHER ADJUSTMENT")
print("=" * 70)

# Load data
print("\nLoading training data...")
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"  Loaded {len(df)} records")

# Create target
df['target_1h'] = df['capacity_wh'].shift(-1)
df['capacity_change'] = df['target_1h'] - df['capacity_wh']

# Filter to solar window hours only for weather model
df_solar = df[df['in_solar_window'] == 1].copy()
print(f"  {len(df_solar)} records in solar window (7-11 AM)")

df = df.dropna()
df_solar = df_solar.dropna()

# SPLIT: Train/Test
split_idx = int(len(df) * 0.8)
split_idx_solar = int(len(df_solar) * 0.8)

print("\n" + "=" * 70)
print("TRAINING TWO MODELS")
print("=" * 70)

# =====================================================
# MODEL 1: Base capacity model (uses capacity_wh)
# =====================================================
print("\n--- MODEL 1: Base Capacity Model ---")

base_features = [
    'capacity_wh', 'power_watts',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'capacity_lag_1h', 'capacity_lag_3h', 'capacity_lag_6h',
    'capacity_rolling_6h', 'capacity_rolling_24h',
]

X_base = df[base_features]
y_base = df['target_1h']

X_base_train = X_base.iloc[:split_idx]
X_base_test = X_base.iloc[split_idx:]
y_base_train = y_base.iloc[:split_idx]
y_base_test = y_base.iloc[split_idx:]

scaler_base = StandardScaler()
X_base_train_scaled = scaler_base.fit_transform(X_base_train)
X_base_test_scaled = scaler_base.transform(X_base_test)

model_base = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    min_samples_leaf=10, subsample=0.8, random_state=42
)
model_base.fit(X_base_train_scaled, y_base_train)

base_train_pred = model_base.predict(X_base_train_scaled)
base_test_pred = model_base.predict(X_base_test_scaled)

print(f"  Train MAE: {mean_absolute_error(y_base_train, base_train_pred):.1f} Wh")
print(f"  Test MAE:  {mean_absolute_error(y_base_test, base_test_pred):.1f} Wh")
print(f"  Test R²:   {r2_score(y_base_test, base_test_pred):.3f}")

# =====================================================
# MODEL 2: Solar charging model (ONLY weather features)
# =====================================================
print("\n--- MODEL 2: Solar Charging Model (weather only) ---")

# For solar hours, predict the CHANGE based on weather
solar_features = [
    # Weather features ONLY
    'shortwave_radiation',
    'cloud_cover',
    'sunshine_minutes',
    'effective_sunshine',
    
    # Time (for seasonal adjustment)
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
]

X_solar = df_solar[solar_features]
y_solar = df_solar['capacity_change']  # Predict change during solar hours

X_solar_train = X_solar.iloc[:split_idx_solar]
X_solar_test = X_solar.iloc[split_idx_solar:]
y_solar_train = y_solar.iloc[:split_idx_solar]
y_solar_test = y_solar.iloc[split_idx_solar:]

scaler_solar = StandardScaler()
X_solar_train_scaled = scaler_solar.fit_transform(X_solar_train)
X_solar_test_scaled = scaler_solar.transform(X_solar_test)

model_solar = GradientBoostingRegressor(
    n_estimators=150, max_depth=4, learning_rate=0.05,
    min_samples_leaf=5, subsample=0.8, random_state=42
)
model_solar.fit(X_solar_train_scaled, y_solar_train)

solar_train_pred = model_solar.predict(X_solar_train_scaled)
solar_test_pred = model_solar.predict(X_solar_test_scaled)

print(f"  Train MAE: {mean_absolute_error(y_solar_train, solar_train_pred):.1f} Wh")
print(f"  Test MAE:  {mean_absolute_error(y_solar_test, solar_test_pred):.1f} Wh")
print(f"  Test R²:   {r2_score(y_solar_test, solar_test_pred):.3f}")

# Feature importance for solar model
print("\n  Solar Model Feature Importance:")
for feat, imp in sorted(zip(solar_features, model_solar.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 30)
    print(f"    {feat:25s}: {imp:.3f} {bar}")

# =====================================================
# BACKTEST: Hybrid approach
# =====================================================
print("\n" + "=" * 70)
print("BACKTEST: Hybrid Model")
print("=" * 70)

test_df = df.iloc[split_idx:].copy()

# Find start
start_idx = test_df.index[0]
for i, row in test_df.iterrows():
    if 7 <= pd.Timestamp(row['timestamp']).hour <= 9:
        start_idx = i
        break

backtest_start = df.loc[start_idx]
initial_capacity = backtest_start['capacity_wh']
print(f"\nBacktest starting from: {backtest_start['timestamp']}")
print(f"Initial capacity: {initial_capacity:.0f} Wh")

# Hybrid weight: how much to trust weather model vs base model
WEATHER_WEIGHT = 0.3  # 30% weather, 70% base

predicted_caps = [initial_capacity]
actual_caps = [initial_capacity]
current_capacity = initial_capacity
capacity_history = list(df.loc[:start_idx, 'capacity_wh'].tail(24))

BATTERY_MIN = 160
BATTERY_MAX = 1600

print(f"\nHybrid blend: {WEATHER_WEIGHT:.0%} weather, {1-WEATHER_WEIGHT:.0%} base")
print("\nBacktest Results (first 24 hours):")
print("Hour | Time  | Actual | Base Pred | Solar Adj | Hybrid | Error")
print("-" * 70)

for i in range(min(48, len(test_df) - (start_idx - test_df.index[0]) - 1)):
    row_idx = start_idx + i
    if row_idx >= len(df) - 1:
        break
        
    row = df.loc[row_idx]
    next_row = df.loc[row_idx + 1]
    
    # Base model prediction
    base_feat = {
        'capacity_wh': current_capacity,
        'power_watts': row['power_watts'],
        'hour_sin': row['hour_sin'],
        'hour_cos': row['hour_cos'],
        'month_sin': row['month_sin'],
        'month_cos': row['month_cos'],
        'capacity_lag_1h': capacity_history[-1] if len(capacity_history) >= 1 else current_capacity,
        'capacity_lag_3h': capacity_history[-3] if len(capacity_history) >= 3 else current_capacity,
        'capacity_lag_6h': capacity_history[-6] if len(capacity_history) >= 6 else current_capacity,
        'capacity_rolling_6h': np.mean(capacity_history[-6:]) if len(capacity_history) >= 6 else current_capacity,
        'capacity_rolling_24h': np.mean(capacity_history[-24:]) if len(capacity_history) >= 24 else current_capacity,
    }
    X_base_pred = pd.DataFrame([base_feat])[base_features]
    X_base_pred_scaled = scaler_base.transform(X_base_pred)
    base_prediction = model_base.predict(X_base_pred_scaled)[0]
    
    # Solar model adjustment (only during solar hours)
    solar_adjustment = 0
    if row['in_solar_window'] == 1:
        solar_feat = {
            'shortwave_radiation': row['shortwave_radiation'],
            'cloud_cover': row['cloud_cover'],
            'sunshine_minutes': row['sunshine_minutes'],
            'effective_sunshine': row['effective_sunshine'],
            'hour_sin': row['hour_sin'],
            'hour_cos': row['hour_cos'],
            'month_sin': row['month_sin'],
            'month_cos': row['month_cos'],
        }
        X_solar_pred = pd.DataFrame([solar_feat])[solar_features]
        X_solar_pred_scaled = scaler_solar.transform(X_solar_pred)
        solar_adjustment = model_solar.predict(X_solar_pred_scaled)[0]
    
    # Hybrid prediction
    # During solar window: blend base with weather-informed change
    if row['in_solar_window'] == 1:
        # Weather model predicts change, so add to current
        weather_prediction = current_capacity + solar_adjustment
        hybrid_prediction = (1 - WEATHER_WEIGHT) * base_prediction + WEATHER_WEIGHT * weather_prediction
    else:
        hybrid_prediction = base_prediction
    
    hybrid_prediction = max(BATTERY_MIN, min(BATTERY_MAX, hybrid_prediction))
    
    actual = next_row['capacity_wh']
    error = hybrid_prediction - actual
    
    if i < 24:
        ts = pd.Timestamp(next_row['timestamp'])
        in_solar = "☀" if row['in_solar_window'] == 1 else " "
        print(f"{i:4d} | {ts.strftime('%H:%M')}{in_solar}| {actual:6.0f} | {base_prediction:9.0f} | {solar_adjustment:+9.0f} | {hybrid_prediction:6.0f} | {error:+6.0f}")
    
    current_capacity = hybrid_prediction
    capacity_history.append(current_capacity)
    
    predicted_caps.append(hybrid_prediction)
    actual_caps.append(actual)

# Metrics
errors = [p - a for p, a in zip(predicted_caps[1:25], actual_caps[1:25])]
mae = np.mean(np.abs(errors))
max_err = max(np.abs(errors))

print(f"\nMean Absolute Error over 24h: {mae:.1f} Wh")
print(f"Max Error: {max_err:.1f} Wh")

# Save if good
if mae < 200:
    print(f"\n✓ Backtest PASSED! Saving hybrid model...")
    
    model_data = {
        'model_base': model_base,
        'model_solar': model_solar,
        'scaler_base': scaler_base,
        'scaler_solar': scaler_solar,
        'base_features': base_features,
        'solar_features': solar_features,
        'weather_weight': WEATHER_WEIGHT,
        'training_date': datetime.now().isoformat(),
        'backtest_mae': mae,
        'model_type': 'hybrid',
        'version': 3,
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Saved to {MODEL_PATH}")
else:
    print(f"\n✗ Backtest FAILED (MAE {mae:.1f} > 200 Wh)")

print("\n" + "=" * 70)
print("SUMMARY: Why hybrid works better")
print("=" * 70)
print("""
1. BASE MODEL (stable):
   - Uses capacity_wh → very accurate baseline
   - Captures general battery behavior
   
2. SOLAR MODEL (weather-aware):
   - ONLY trained on solar window data
   - Predicts charging amount from weather
   - Feature importance now shows weather!
   
3. HYBRID BLEND:
   - Solar hours: 70% base + 30% weather adjustment
   - Non-solar hours: 100% base model
   
This gives the best of both worlds:
   ✓ Stable predictions (from base model)
   ✓ Weather-responsive charging (from solar model)
""")
