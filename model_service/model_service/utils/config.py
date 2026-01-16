"""
Configuration constants for water model inference.

Centralized configuration for beam search, physics, and uncertainty.
"""

# =============================================================================
# Beam Search Configuration
# =============================================================================

# Default beam width (number of candidates to keep per expansion)
BEAM_WIDTH_DEFAULT = 200

# Threshold for greedy fallback (beam_width=1) when forecast is too long
BEAM_GREEDY_FALLBACK_DAYS = 14

# Enable adaptive beam width (larger early, smaller late)
ADAPTIVE_BEAM_DEFAULT = True

# Threshold for enabling adaptive beam
ADAPTIVE_BEAM_THRESHOLD_DAYS = 10

# Number of diversity buckets for state preservation
DIVERSITY_BUCKETS = 10

# Enable lookahead/rollout for better pruning
LOOKAHEAD_ENABLED = True


# =============================================================================
# Physics Model Configuration
# =============================================================================

# Tank physics
TANK_EVAPORATION_LOSS_L_DAY = 0.0  # Daily evaporation loss in liters

# Soil physics coefficients
SOIL_K_IRRIG = 1.5   # How much 1 mm irrigation increases soil moisture
SOIL_K_RAIN = 0.3    # How much 1 mm rain affects soil moisture
SOIL_K_EVAP = 0.4    # Evaporation strength factor

# Soil moisture limits
SOIL_MIN_MOISTURE = 0.0
SOIL_MAX_MOISTURE = 100.0

# Maximum ML residual as fraction of physics value
ML_RESIDUAL_MAX_FRACTION = 0.2


# =============================================================================
# Uncertainty Configuration
# =============================================================================

# Feature flag: Use conformal prediction (True) or legacy multipliers (False)
USE_CONFORMAL_UNCERTAINTY = True

# Quantile levels for conformal prediction
UNCERTAINTY_QUANTILES = (0.1, 0.5, 0.9)

# Default tank residuals (for when no calibration data available)
DEFAULT_TANK_RESIDUALS = {
    0.1: -15.0,  # 10th percentile (pessimistic)
    0.5: 0.0,    # 50th percentile (median)
    0.9: 20.0,   # 90th percentile (optimistic)
}

# Default soil residuals
DEFAULT_SOIL_RESIDUALS = {
    0.1: -8.0,   # 10th percentile
    0.5: 0.0,    # 50th percentile
    0.9: 10.0,   # 90th percentile
}


# =============================================================================
# Irrigation Configuration
# =============================================================================

# Liters of water per pump activation level
LITERS_PER_PUMP_LEVEL = 1.5

# Maximum irrigation levels (0, 1, 2)
MAX_PUMP_LEVEL = 2


# =============================================================================
# Scoring Configuration
# =============================================================================

# Score weights for plan optimization
SCORE_ALPHA = 10.0   # Weight for moisture deficit penalty
SCORE_BETA = 0.1     # Weight for irrigation cost
SCORE_GAMMA = 0.3    # Weight for final tank level reward
SCORE_DELTA = 1.0    # Weight for overflow penalty
