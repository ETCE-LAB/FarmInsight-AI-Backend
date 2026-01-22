"""
Site Configuration for FarmInsight Energy Model

Calibration constants for the Clausthal-Zellerfeld installation site.
The site is heavily shaded by surrounding terrain/buildings, receiving
significantly less direct sunlight than theoretical weather data suggests.
"""

# ============================================================================
# Location Parameters
# ============================================================================
SITE_LATITUDE = 51.900994701794644
SITE_LONGITUDE = 10.43181359767914
SITE_ELEVATION = 289  # meters above sea level

# ============================================================================
# Solar/Shading Configuration
# ============================================================================

# Site shading factor - the fraction of theoretical sunshine that actually
# reaches the solar panels. Derived from comparing historical battery charging
# rates vs. reported sunshine hours from Open-Meteo.
# Value of 0.35 means only 35% of expected solar radiation is received.
# (Historical analysis shows avg ~73 Wh/hour charging vs theoretical ~200+ Wh)
SITE_SHADING_FACTOR = 0.35

# Effective solar hours - the time window when panels can receive sunlight
# at this specific location due to terrain shading. Outside this window,
# charging is minimal regardless of weather.
# IMPORTANT: Historical data shows charging 9:00-12:00 local time!
# Peak charging at 10:00-11:00 with +80 Wh/hour
EFFECTIVE_SOLAR_START_HOUR = 9   # 9:00 AM - charging starts
EFFECTIVE_SOLAR_END_HOUR = 12    # 12:00 PM - charging ends

# Maximum solar panel output in watts (under ideal conditions)
# Historical max observed: 400 Wh/hour charging rate
MAX_SOLAR_OUTPUT_WATTS = 600

# ============================================================================
# Battery Parameters (Anker Solar Bank)
# ============================================================================

# Maximum battery capacity in Watt-hours
BATTERY_MAX_WH = 1600

# Observed minimum battery level - the battery never drops below this value
# in practice (10% = 160 Wh). Using this for realistic worst-case scenarios.
BATTERY_MIN_WH = 160

# Typical/average power consumption in Watts
# Historical analysis shows nighttime drain of ~7.5 Wh/hour
# But daytime consumption is higher (~35-47W with equipment running)
# The actual battery drain is lower than equipment power draw
# because the system has some efficiency losses accounted for
# Using calibrated value based on historical battery drain rate
TYPICAL_CONSUMPTION_WATTS = 12.0  # Wh/hour battery drain

# ============================================================================
# Scenario Configuration
# ============================================================================

# Consumption factors for different scenarios
SCENARIO_CONSUMPTION_FACTORS = {
    "expected": 1.0,      # Normal consumption
    "optimistic": 0.75,   # 25% less consumption
    "pessimistic": 1.35,  # 35% higher consumption
}

# Solar efficiency factors for different scenarios
SCENARIO_SOLAR_FACTORS = {
    "expected": 1.0,      # Normal solar
    "optimistic": 1.3,    # 30% more solar (sunny days)
    "pessimistic": 0.3,   # 70% less solar (cloudy/winter)
}

# ============================================================================
# Seasonal Adjustments
# ============================================================================

# Monthly adjustment factors for solar production
# CALIBRATED FROM HISTORICAL DATA (July-November 2025):
# The site receives BEST sun in August-September (low angle morning sun 
# clears terrain better than high summer sun which is blocked)
# This is a unique characteristic of this heavily-shaded location.
SEASONAL_SOLAR_FACTORS = {
    1: 0.20,   # January  - Winter, minimal sun, estimated from November trend
    2: 0.25,   # February - Slightly improving
    3: 0.40,   # March    - Spring, sun angle improving
    4: 0.55,   # April    - Good conditions
    5: 0.65,   # May      
    6: 0.50,   # June     - Summer, but high sun may be blocked by terrain
    7: 0.45,   # July     - Measured: ~55 Wh/hour (relative to peak)
    8: 1.00,   # August   - Measured: ~130 Wh/hour - PEAK! Lower sun clears terrain
    9: 1.00,   # September - Measured: ~128 Wh/hour - Also peak performance
    10: 0.60,  # October   - Measured: ~75 Wh/hour
    11: 0.20,  # November  - Measured: ~24 Wh/hour
    12: 0.15,  # December  - Winter solstice, minimal
}


def get_seasonal_factor(month: int) -> float:
    """Get seasonal solar adjustment factor for a given month."""
    return SEASONAL_SOLAR_FACTORS.get(month, 0.5)


def get_effective_solar_fraction(hour: int) -> float:
    """
    Get the fraction of solar collection capability for a given hour.
    
    Returns 1.0 during peak solar window, 0.0 outside effective hours,
    with gradual ramp-up/down at the edges.
    """
    if hour < EFFECTIVE_SOLAR_START_HOUR - 1 or hour > EFFECTIVE_SOLAR_END_HOUR + 1:
        return 0.0
    elif EFFECTIVE_SOLAR_START_HOUR <= hour <= EFFECTIVE_SOLAR_END_HOUR:
        return 1.0
    elif hour == EFFECTIVE_SOLAR_START_HOUR - 1:
        return 0.3  # Ramp up
    elif hour == EFFECTIVE_SOLAR_END_HOUR + 1:
        return 0.3  # Ramp down
    return 0.0
