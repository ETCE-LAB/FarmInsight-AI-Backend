"""
Uncertainty estimation using Conformal Prediction.

Provides quantile-based uncertainty intervals for scenario generation
without requiring model re-training.
"""
from typing import Dict, Optional

import numpy as np


class ConformalUncertainty:
    """
    Conformal prediction for uncertainty intervals.
    
    Uses calibration residuals (y_true - y_pred) from validation data
    to compute prediction intervals at specified quantiles.
    
    Scenarios are derived from quantiles:
    - best-case: 90th percentile (optimistic)
    - average-case: 50th percentile (median)
    - worst-case: 10th percentile (pessimistic)
    """
    
    # Default residual values (estimated from typical water model behavior)
    # These can be updated by fitting on actual calibration data
    DEFAULT_TANK_RESIDUALS = {
        0.1: -15.0,  # Pessimistic: tank decreases more than expected
        0.5: 0.0,    # Median: close to prediction
        0.9: 20.0,   # Optimistic: tank increases more than expected
    }
    
    DEFAULT_SOIL_RESIDUALS = {
        0.1: -8.0,   # Pessimistic: soil dries faster
        0.5: 0.0,    # Median
        0.9: 10.0,   # Optimistic: soil retains more moisture
    }
    
    SCENARIO_QUANTILE_MAP = {
        "best-case": 0.9,
        "average-case": 0.5,
        "worst-case": 0.1,
    }
    
    def __init__(
        self,
        tank_residuals: Optional[Dict[float, float]] = None,
        soil_residuals: Optional[Dict[float, float]] = None,
    ):
        """
        Args:
            tank_residuals: {quantile: residual_value} for tank predictions
            soil_residuals: {quantile: residual_value} for soil predictions
        """
        self.tank_quantiles = tank_residuals or self.DEFAULT_TANK_RESIDUALS.copy()
        self.soil_quantiles = soil_residuals or self.DEFAULT_SOIL_RESIDUALS.copy()
    
    def fit_from_residuals(
        self,
        tank_residuals: np.ndarray,
        soil_residuals: np.ndarray,
        quantiles: tuple = (0.1, 0.5, 0.9),
    ) -> None:
        """
        Fit quantiles from calibration residuals.
        
        Args:
            tank_residuals: Array of (y_true - y_pred) for tank level
            soil_residuals: Array of (y_true - y_pred) for soil moisture
            quantiles: Quantile levels to compute
        """
        for q in quantiles:
            self.tank_quantiles[q] = float(np.percentile(tank_residuals, q * 100))
            self.soil_quantiles[q] = float(np.percentile(soil_residuals, q * 100))
    
    def get_tank_adjustment(self, scenario: str) -> float:
        """
        Get tank prediction adjustment for scenario.
        
        Args:
            scenario: One of "best-case", "average-case", "worst-case"
            
        Returns:
            Adjustment value to add to base prediction
        """
        quantile = self.SCENARIO_QUANTILE_MAP.get(scenario, 0.5)
        return self.tank_quantiles.get(quantile, 0.0)
    
    def get_soil_adjustment(self, scenario: str) -> float:
        """
        Get soil prediction adjustment for scenario.
        
        Args:
            scenario: One of "best-case", "average-case", "worst-case"
            
        Returns:
            Adjustment value to add to base prediction
        """
        quantile = self.SCENARIO_QUANTILE_MAP.get(scenario, 0.5)
        return self.soil_quantiles.get(quantile, 0.0)
    
    def apply_to_tank_prediction(
        self,
        base_prediction: float,
        scenario: str,
        tank_capacity: float,
    ) -> float:
        """
        Apply uncertainty adjustment to tank prediction.
        
        Args:
            base_prediction: Physics/ML combined prediction
            scenario: Scenario name
            tank_capacity: Maximum tank capacity
            
        Returns:
            Adjusted prediction, clipped to valid range
        """
        adjustment = self.get_tank_adjustment(scenario)
        result = base_prediction + adjustment
        return max(0.0, min(tank_capacity, result))
    
    def apply_to_soil_prediction(
        self,
        base_prediction: float,
        scenario: str,
        min_moisture: float = 0.0,
        max_moisture: float = 100.0,
    ) -> float:
        """
        Apply uncertainty adjustment to soil prediction.
        
        Args:
            base_prediction: Physics/ML combined prediction
            scenario: Scenario name
            min_moisture: Minimum valid moisture
            max_moisture: Maximum valid moisture
            
        Returns:
            Adjusted prediction, clipped to valid range
        """
        adjustment = self.get_soil_adjustment(scenario)
        result = base_prediction + adjustment
        return max(min_moisture, min(max_moisture, result))


class LegacyMultiplierUncertainty:
    """
    Backward-compatible uncertainty using the original multiplier approach.
    
    Used as fallback when conformal prediction is disabled.
    """
    
    MULTIPLIERS = {
        "best-case": {
            "temp": 0.9,
            "rain": 1.1,
            "inflow": 1.1,
            "state": 1.05,
        },
        "average-case": {
            "temp": 1.0,
            "rain": 1.0,
            "inflow": 1.0,
            "state": 1.0,
        },
        "worst-case": {
            "temp": 1.1,
            "rain": 0.9,
            "inflow": 0.9,
            "state": 0.95,
        },
    }
    
    def adjust_value(self, scenario: str, value: float, value_type: str) -> float:
        """
        Apply multiplier adjustment (legacy behavior).
        
        Args:
            scenario: Scenario name
            value: Value to adjust
            value_type: One of "temp", "rain", "inflow", "state"
            
        Returns:
            Adjusted value
        """
        multipliers = self.MULTIPLIERS.get(scenario, self.MULTIPLIERS["average-case"])
        multiplier = multipliers.get(value_type, 1.0)
        return value * multiplier
    
    # Provide same interface as ConformalUncertainty for duck typing
    def get_tank_adjustment(self, scenario: str) -> float:
        return 0.0  # Legacy applies to inputs, not outputs
    
    def get_soil_adjustment(self, scenario: str) -> float:
        return 0.0
    
    def apply_to_tank_prediction(
        self, base_prediction: float, scenario: str, tank_capacity: float
    ) -> float:
        return base_prediction
    
    def apply_to_soil_prediction(
        self, base_prediction: float, scenario: str, 
        min_moisture: float = 0.0, max_moisture: float = 100.0
    ) -> float:
        return base_prediction


# Feature flag for selecting uncertainty method
USE_CONFORMAL_UNCERTAINTY = True


def get_uncertainty_estimator(use_conformal: Optional[bool] = None) -> ConformalUncertainty:
    """
    Factory function to get uncertainty estimator.
    
    Args:
        use_conformal: Override feature flag. If None, uses USE_CONFORMAL_UNCERTAINTY.
        
    Returns:
        Uncertainty estimator instance
    """
    if use_conformal is None:
        use_conformal = USE_CONFORMAL_UNCERTAINTY
    
    if use_conformal:
        return ConformalUncertainty()
    else:
        return LegacyMultiplierUncertainty()


def case_value_multiplicator(scenario: str, value: float, value_name: str) -> float:
    """
    Legacy function for backward compatibility.
    
    Maps value_name to appropriate multiplier type and applies adjustment.
    This is kept for compatibility but new code should use ConformalUncertainty.
    """
    if scenario == "average-case":
        return value
    
    is_best = scenario == "best-case"
    
    if value_name in ("temp_max", "temp_today", "temp_tomorrow"):
        return value * (0.9 if is_best else 1.1)
    
    if value_name in ("rain", "rain_today", "rain_tomorrow"):
        return value * (1.1 if is_best else 0.9)
    
    if value_name in ("inflow_l", "calculated_total_l"):
        return value * (1.1 if is_best else 0.9)
    
    if value_name in ("soil_moisture_previous", "water_level_previous"):
        return value * (1.05 if is_best else 0.95)
    
    return value
