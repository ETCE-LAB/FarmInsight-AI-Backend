"""
Physics-based models for tank and soil calculations.

Ensures physical consistency that ML cannot override.
"""
from typing import Tuple


class TankPhysics:
    """
    Mass-balance based tank level calculation.
    
    Physics equation:
        tank_level_new = tank_level_prev + inflow - outflow - losses
        
    With proper clipping and overflow handling.
    """
    
    def __init__(self, evaporation_loss_per_day: float = 0.0):
        """
        Args:
            evaporation_loss_per_day: Daily evaporation loss in liters (default 0)
        """
        self.evaporation_loss = evaporation_loss_per_day
    
    def update(
        self,
        tank_level_prev: float,
        inflow_l: float,
        outflow_l: float,
        tank_capacity: float,
    ) -> Tuple[float, float]:
        """
        Calculate new tank level using mass balance.
        
        Args:
            tank_level_prev: Previous tank level in liters
            inflow_l: Water inflow in liters
            outflow_l: Water outflow (irrigation) in liters
            tank_capacity: Maximum tank capacity in liters
            
        Returns:
            (new_tank_level, overflow_l)
        """
        # Mass balance: delta = inflow - outflow - losses
        delta = inflow_l - outflow_l - self.evaporation_loss
        raw_level = tank_level_prev + delta
        
        # Clipping and overflow
        new_level = max(0.0, min(tank_capacity, raw_level))
        overflow = max(0.0, raw_level - tank_capacity)
        
        return new_level, overflow
    
    def compute_max_outflow(
        self,
        tank_level_prev: float,
        inflow_l: float,
    ) -> float:
        """
        Compute maximum possible outflow given current state.
        
        Ensures we don't extract more water than available.
        """
        available = tank_level_prev + inflow_l - self.evaporation_loss
        return max(0.0, available)


class SoilPhysics:
    """
    Bucket/ET0-proxy based soil moisture calculation.
    
    Base physics equation:
        soil_new = soil_prev + k_irrig * irrigation + k_rain * rain - k_evap * temp_factor
        
    ML model provides bounded residual corrections only.
    """
    
    # Default coefficients
    DEFAULT_K_IRRIG = 1.5  # How much 1 mm irrigation increases soil moisture
    DEFAULT_K_RAIN = 0.3   # How much 1 mm rain affects soil moisture  
    DEFAULT_K_EVAP = 0.4   # Evaporation strength factor
    DEFAULT_MAX_RESIDUAL_FRACTION = 0.2  # Max ML correction as fraction of physics value
    
    def __init__(
        self,
        k_irrig: float = DEFAULT_K_IRRIG,
        k_rain: float = DEFAULT_K_RAIN,
        k_evap: float = DEFAULT_K_EVAP,
        min_moisture: float = 0.0,
        max_moisture: float = 100.0,
    ):
        """
        Args:
            k_irrig: Irrigation effect coefficient
            k_rain: Rain effect coefficient
            k_evap: Evaporation coefficient
            min_moisture: Minimum soil moisture (%)
            max_moisture: Maximum soil moisture (%)
        """
        self.k_irrig = k_irrig
        self.k_rain = k_rain
        self.k_evap = k_evap
        self.min_moisture = min_moisture
        self.max_moisture = max_moisture
    
    def compute_base(
        self,
        soil_moisture_prev: float,
        irrigation_mm: float,
        rain_mm: float,
        temp_c: float,
    ) -> float:
        """
        Compute physics-based soil moisture (before ML correction).
        
        Args:
            soil_moisture_prev: Previous soil moisture (%)
            irrigation_mm: Irrigation amount in mm
            rain_mm: Rain amount in mm
            temp_c: Temperature in Celsius
            
        Returns:
            Physics-based soil moisture estimate
        """
        # Temperature-based evaporation factor
        temp_factor = max(0.0, temp_c / 30.0)
        
        # Physics model
        soil_new = (
            soil_moisture_prev
            + self.k_irrig * irrigation_mm
            + self.k_rain * rain_mm
            - self.k_evap * temp_factor
        )
        
        return soil_new
    
    def apply_ml_residual(
        self,
        physics_result: float,
        ml_residual: float,
        max_residual_fraction: float = DEFAULT_MAX_RESIDUAL_FRACTION,
    ) -> float:
        """
        Apply bounded ML residual correction to physics result.
        
        ML can only provide corrections up to max_residual_fraction of the
        physics value. This ensures physics cannot be completely overridden.
        
        Args:
            physics_result: Result from compute_base()
            ml_residual: Raw ML model residual prediction
            max_residual_fraction: Maximum allowed correction as fraction
            
        Returns:
            Final soil moisture with bounded ML correction
        """
        # Compute maximum allowed correction
        max_correction = max_residual_fraction * max(1.0, abs(physics_result))
        
        # Clamp residual
        clamped_residual = max(-max_correction, min(max_correction, ml_residual))
        
        # Apply and clip to valid range
        result = physics_result + clamped_residual
        return max(self.min_moisture, min(self.max_moisture, result))
    
    def update(
        self,
        soil_moisture_prev: float,
        irrigation_mm: float,
        rain_mm: float,
        temp_c: float,
        ml_residual: float = 0.0,
        max_residual_fraction: float = DEFAULT_MAX_RESIDUAL_FRACTION,
    ) -> float:
        """
        Full update: physics + bounded ML correction.
        
        Convenience method combining compute_base and apply_ml_residual.
        """
        physics_result = self.compute_base(
            soil_moisture_prev, irrigation_mm, rain_mm, temp_c
        )
        return self.apply_ml_residual(
            physics_result, ml_residual, max_residual_fraction
        )
