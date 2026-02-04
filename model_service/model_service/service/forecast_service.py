"""
Forecast service for water management optimization.

Refactored to use beam search instead of exponential plan enumeration.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from joblib import load

SCENARIO_TO_Q = {"worst_case": 0.1, "average_case": 0.5, "best_case": 0.9}

from ..utils.beam_search import BeamSearchOptimizer, BeamState
from ..utils.config import (
    BEAM_WIDTH_DEFAULT,
    BEAM_GREEDY_FALLBACK_DAYS,
    ADAPTIVE_BEAM_THRESHOLD_DAYS,
    DIVERSITY_BUCKETS,
    LOOKAHEAD_ENABLED,
    SCORE_ALPHA,
    SCORE_BETA,
    SCORE_GAMMA,
    SCORE_DELTA,
    LITERS_PER_PUMP_LEVEL,
)
from ..utils.inference_processor import prepare_data_for_prediction
from ..utils.simulation_loop import run_forecast_period, simulate_single_day

BASE_DIR = Path(__file__).resolve().parent
TRAINED_MODELS_DIR = BASE_DIR.parent / "trained_models"

FACE_AZIMUTH_DEG = 30.0
FACE_AREA_M2 = 6.668
SLOPE_DEG = 30.26

# Estimated plant area
A_SOIL_M2 = 8.0

MAX_IRRIGATION_L_DAY = 1.5


def score_plan(
        tank_results: List[Dict],
        soil_results: List[Dict],
        soil_threshold: float,
        scenario: str,
        alpha: float = SCORE_ALPHA,
        beta: float = SCORE_BETA,
        gamma: float = SCORE_GAMMA,
        delta: float = SCORE_DELTA,
) -> float:
    """
    Score a complete irrigation plan.
    
    Args:
        tank_results: List of daily tank results
        soil_results: List of daily soil results
        soil_threshold: Target soil moisture threshold
        scenario: Scenario name
        alpha: Weight for moisture deficit penalty
        beta: Weight for irrigation cost
        gamma: Weight for final tank level reward
        delta: Weight for overflow penalty
        
    Returns:
        Total score (higher is better)
    """
    moisture_deficit = 0.0
    total_irrigation = 0.0
    total_overflow = 0.0

    for t, s in zip(tank_results, soil_results):
        soil_mm = s["soil_mm"]
        irrigation_mm = s["irrigation_mm"]
        overflow_l = t["overflow_l"]

        moisture_deficit += max(0.0, soil_threshold - soil_mm)
        total_irrigation += irrigation_mm
        total_overflow += overflow_l

    final_tank = tank_results[-1]["tank_l"] if tank_results else 0.0

    return (
        - alpha * moisture_deficit
        - beta * total_irrigation
        - delta * total_overflow
        + gamma * final_tank
    )


def _create_day_scorer(soil_threshold: float) -> Callable:
    """Create per-day scoring function for beam search."""
    def score_day(tank_result: Dict, soil_result: Dict) -> float:
        soil_mm = soil_result["soil_mm"]
        irrigation_mm = soil_result["irrigation_mm"]
        overflow_l = tank_result["overflow_l"]
        tank_l = tank_result["tank_l"]
        
        moisture_deficit = max(0.0, soil_threshold - soil_mm)
        
        return (
            - SCORE_ALPHA * moisture_deficit
            - SCORE_BETA * irrigation_mm
            - SCORE_DELTA * overflow_l
            + SCORE_GAMMA * tank_l * 0.1  # Normalize tank contribution per day
        )
    
    return score_day


def _create_day_simulator(scenario: str, tank_capacity: float, plant_area: float):
    """Create day simulation function for beam search."""
    def simulate(
        tank_level_prev: float,
        moisture_prev: float,
        tank_level_pp: float,
        moisture_pp: float,
        action: int,
        day_data: Any,
        next_day_data: Any,
        ml_model: Any,
        **kwargs
    ) -> Tuple[Dict, Dict, float, float]:
        return simulate_single_day(
            tank_level_prev=tank_level_prev,
            moisture_prev=moisture_prev,
            tank_level_pp=tank_level_pp,
            moisture_pp=moisture_pp,
            action=action,
            day_data=day_data,
            next_day_data=next_day_data,
            ml_model=ml_model,
            tank_capacity=tank_capacity,
            plant_area=plant_area,
            scenario=scenario,
            irrigation_history=kwargs.get("irrigation_history", []),
        )
    
    return simulate

def _load_quantile_models():
    models = {}
    for scen, q in SCENARIO_TO_Q.items():
        models[scen] = load(TRAINED_MODELS_DIR / f"synthetic_quantile_q{int(q*100):02d}.pkl")
    return models

def model_forecast(
        latitude: float,
        longitude: float,
        forecast_days: int,
        initial_water_level: float,
        tank_capacity_liters: float,
        soil_threshold: float,
        scenarios: List[str],
        start_soil_moisture: float,
        beam_width: Optional[int] = None,
        use_beam_search: bool = True,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Generate water management forecast with optimal irrigation plans.
    
    Uses beam search optimization instead of exponential enumeration.
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        forecast_days: Number of days to forecast
        initial_water_level: Starting tank water level
        tank_capacity_liters: Tank capacity
        soil_threshold: Target soil moisture
        scenarios: List of scenario names
        start_soil_moisture: Starting soil moisture
        beam_width: Override default beam width (None = use config)
        use_beam_search: If False, falls back to legacy itertools.product
        
    Returns:
        (tank_results_by_scenario, soil_results_by_scenario)
    """
    quantile_models = _load_quantile_models()

    forecast, inflow_data, df_forecast = prepare_data_for_prediction(
        latitude=latitude,
        longitude=longitude,
        forecast_days=forecast_days
    )

    all_best_tank_results: Dict[str, List[Dict]] = {}
    all_best_soil_results: Dict[str, List[Dict]] = {}

    # Determine beam width
    if beam_width is None:
        beam_width = BEAM_WIDTH_DEFAULT
    
    # Use adaptive beam for longer horizons
    use_adaptive = forecast_days > ADAPTIVE_BEAM_THRESHOLD_DAYS
    
    # Check for greedy fallback
    use_greedy_fallback = forecast_days > BEAM_GREEDY_FALLBACK_DAYS

    for scenario in scenarios:
        ml_model = quantile_models[scenario]
        if use_beam_search:
            # Use beam search optimization
            best_plan, best_tank_results, best_soil_results = _optimize_with_beam_search(
                df_forecast=df_forecast,
                ml_model=ml_model,
                forecast_days=forecast_days,
                initial_water_level=initial_water_level,
                initial_moisture=start_soil_moisture,
                tank_capacity=tank_capacity_liters,
                plant_area=A_SOIL_M2,
                scenario=scenario,
                soil_threshold=soil_threshold,
                beam_width=1 if use_greedy_fallback else beam_width,
                adaptive_beam=use_adaptive and not use_greedy_fallback,
            )
        else:
            # Legacy fallback: exhaustive search (only for small forecast_days)
            best_plan, best_tank_results, best_soil_results = _optimize_exhaustive(
                df_forecast=df_forecast,
                ml_model=ml_model,
                forecast_days=forecast_days,
                initial_water_level=initial_water_level,
                initial_moisture=start_soil_moisture,
                tank_capacity=tank_capacity_liters,
                plant_area=A_SOIL_M2,
                scenario=scenario,
                soil_threshold=soil_threshold,
            )

        if best_tank_results is not None and best_soil_results is not None:
            all_best_tank_results[scenario] = best_tank_results
            all_best_soil_results[scenario] = best_soil_results

    return all_best_tank_results, all_best_soil_results


def _optimize_with_beam_search(
        df_forecast: Any,
        ml_model: Any,
        forecast_days: int,
        initial_water_level: float,
        initial_moisture: float,
        tank_capacity: float,
        plant_area: float,
        scenario: str,
        soil_threshold: float,
        beam_width: int,
        adaptive_beam: bool,
) -> Tuple[Tuple[int, ...], List[Dict], List[Dict]]:
    """
    Optimize irrigation plan using beam search.
    """
    optimizer = BeamSearchOptimizer(
        beam_width=beam_width,
        diversity_buckets=DIVERSITY_BUCKETS,
        lookahead_enabled=LOOKAHEAD_ENABLED,
        adaptive_beam=adaptive_beam,
        greedy_fallback_days=BEAM_GREEDY_FALLBACK_DAYS,
    )

    simulate_day = _create_day_simulator(scenario, tank_capacity, plant_area)
    score_day = _create_day_scorer(soil_threshold)

    best_plan, tank_results, soil_results = optimizer.optimize(
        df_forecast=df_forecast,
        simulate_day_fn=simulate_day,
        score_day_fn=score_day,
        initial_tank_level=initial_water_level,
        initial_moisture=initial_moisture,
        tank_capacity=tank_capacity,
        plant_area=plant_area,
        scenario=scenario,
        ml_model=ml_model,
        forecast_days=forecast_days
    )

    return best_plan, tank_results, soil_results


def _optimize_exhaustive(
        df_forecast: Any,
        ml_model: Any,
        forecast_days: int,
        initial_water_level: float,
        initial_moisture: float,
        tank_capacity: float,
        plant_area: float,
        scenario: str,
        soil_threshold: float,
) -> Tuple[Tuple[int, ...], List[Dict], List[Dict]]:
    """
    Legacy exhaustive search (exponential complexity).
    
    Only use for small forecast_days (<=5) or debugging.
    """
    import itertools
    
    plans = list(itertools.product([0, 1, 2], repeat=forecast_days))
    
    best_score: Optional[float] = None
    best_plan: Optional[Tuple[int, ...]] = None
    best_tank_results: Optional[List[Dict]] = None
    best_soil_results: Optional[List[Dict]] = None

    for plan in plans:
        tank_results, soil_results = run_forecast_period(
            df_forecast=df_forecast,
            ml_model=ml_model,
            forecast_days=forecast_days,
            initial_water_level=initial_water_level,
            initial_moisture=initial_moisture,
            plant_area=plant_area,
            scenario=scenario,
            tank_capacity=tank_capacity,
            plan=plan
        )

        score = score_plan(
            tank_results, soil_results, soil_threshold, scenario=scenario
        )
        
        if best_score is None or score > best_score:
            best_score = score
            best_plan = plan
            best_tank_results = tank_results
            best_soil_results = soil_results

    return best_plan, best_tank_results, best_soil_results
