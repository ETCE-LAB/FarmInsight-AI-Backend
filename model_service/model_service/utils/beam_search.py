"""
Beam Search Optimizer for Water Management Plan Optimization.

Replaces exponential O(3^n) itertools.product search with scalable beam search.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class BeamState:
    """State tracked during beam search."""
    partial_plan: Tuple[int, ...]  # Actions taken so far (0, 1, or 2 per day)
    tank_level: float
    soil_moisture: float
    cumulative_score: float
    tank_results: List[Dict] = field(default_factory=list)
    soil_results: List[Dict] = field(default_factory=list)
    
    def get_bucket_key(self, tank_bucket_size: float = 50.0, 
                       soil_bucket_size: float = 10.0) -> Tuple[int, int]:
        """Get state bucket for diversity preservation."""
        return (
            int(self.tank_level // tank_bucket_size),
            int(self.soil_moisture // soil_bucket_size)
        )


class BeamSearchOptimizer:
    """
    Beam search optimizer for irrigation planning.
    
    Features:
    - Partial scoring: cumulative costs/rewards per day
    - Lookahead: greedy rollout to estimate final score
    - Diversity: bucket-based preservation of varied states
    - Adaptive beam: optional larger width in early days
    """
    
    ACTIONS = (0, 1, 2)  # Pump usage levels
    
    def __init__(
        self,
        beam_width: int = 200,
        diversity_buckets: int = 10,
        lookahead_enabled: bool = True,
        adaptive_beam: bool = False,
        greedy_fallback_days: int = 14,
    ):
        self.beam_width = beam_width
        self.diversity_buckets = diversity_buckets
        self.lookahead_enabled = lookahead_enabled
        self.adaptive_beam = adaptive_beam
        self.greedy_fallback_days = greedy_fallback_days
    
    def get_beam_width_for_day(self, day: int, total_days: int) -> int:
        """Get beam width for specific day (adaptive or fixed)."""
        if not self.adaptive_beam:
            return self.beam_width
        
        # Larger beam early, smaller late
        # Day 0: 1.5x, Day n-1: 0.75x
        progress = day / max(1, total_days - 1)
        factor = 1.5 - 0.75 * progress
        return max(10, int(self.beam_width * factor))
    
    def optimize(
        self,
        df_forecast: pd.DataFrame,
        simulate_day_fn: Callable,
        score_day_fn: Callable,
        initial_tank_level: float,
        initial_moisture: float,
        tank_capacity: float,
        plant_area: float,
        scenario: str,
        ml_model: Any,
    ) -> Tuple[Tuple[int, ...], List[Dict], List[Dict]]:
        """
        Run beam search optimization.
        
        Args:
            df_forecast: Weather forecast DataFrame (sorted by date)
            simulate_day_fn: Function(state, action, day_data, ml_model, ...) -> (tank_result, soil_result, new_tank, new_soil)
            score_day_fn: Function(tank_result, soil_result, soil_threshold) -> float (day score)
            initial_tank_level: Starting tank level
            initial_moisture: Starting soil moisture
            tank_capacity: Tank capacity in liters
            plant_area: Plant area in m²
            scenario: Scenario name
            ml_model: Trained ML model
            
        Returns:
            (best_plan, tank_results, soil_results)
        """
        forecast_days = len(df_forecast)
        
        # Greedy fallback for very long horizons
        if forecast_days > self.greedy_fallback_days:
            effective_beam_width = 1
        else:
            effective_beam_width = self.beam_width
        
        df_forecast = df_forecast.sort_values("date").reset_index(drop=True)
        
        # Initialize beam with empty plan
        initial_state = BeamState(
            partial_plan=(),
            tank_level=initial_tank_level,
            soil_moisture=initial_moisture,
            cumulative_score=0.0,
            tank_results=[],
            soil_results=[],
        )
        beam: List[BeamState] = [initial_state]
        
        # Process each day
        for day_idx in range(forecast_days):
            day_data = df_forecast.iloc[day_idx]
            next_day_data = df_forecast.iloc[day_idx + 1] if day_idx + 1 < forecast_days else None
            
            current_beam_width = self.get_beam_width_for_day(day_idx, forecast_days)
            if forecast_days > self.greedy_fallback_days:
                current_beam_width = 1
            
            candidates: List[BeamState] = []
            
            # Expand each state with all actions
            for state in beam:
                for action in self.ACTIONS:
                    # Simulate this day
                    tank_result, soil_result, new_tank, new_soil = simulate_day_fn(
                        tank_level_prev=state.tank_level,
                        moisture_prev=state.soil_moisture,
                        action=action,
                        day_data=day_data,
                        next_day_data=next_day_data,
                        ml_model=ml_model,
                        tank_capacity=tank_capacity,
                        plant_area=plant_area,
                        scenario=scenario,
                        irrigation_history=self._get_irrigation_history(state, forecast_days),
                    )
                    
                    # Compute day score
                    day_score = score_day_fn(tank_result, soil_result)
                    
                    new_state = BeamState(
                        partial_plan=state.partial_plan + (action,),
                        tank_level=new_tank,
                        soil_moisture=new_soil,
                        cumulative_score=state.cumulative_score + day_score,
                        tank_results=state.tank_results + [tank_result],
                        soil_results=state.soil_results + [soil_result],
                    )
                    
                    # Add lookahead score if enabled and not last day
                    if self.lookahead_enabled and day_idx < forecast_days - 1:
                        lookahead_score = self._greedy_lookahead(
                            new_state, df_forecast, day_idx + 1, simulate_day_fn, 
                            score_day_fn, ml_model, tank_capacity, plant_area, scenario,
                            forecast_days
                        )
                        new_state.cumulative_score += lookahead_score * 0.5  # Discount lookahead
                    
                    candidates.append(new_state)
            
            # Prune beam with diversity preservation
            beam = self._prune_beam(candidates, current_beam_width)
        
        # Return best final state
        best_state = max(beam, key=lambda s: s.cumulative_score)
        return best_state.partial_plan, best_state.tank_results, best_state.soil_results
    
    def _get_irrigation_history(self, state: BeamState, forecast_days: int) -> List[float]:
        """Extract irrigation history from state for feature computation."""
        history = [0.0] * forecast_days
        for i, result in enumerate(state.soil_results):
            if i < len(history):
                history[i] = result.get("irrigation_mm", 0.0) * 8.0  # Convert back to liters approx
        return history
    
    def _greedy_lookahead(
        self,
        state: BeamState,
        df_forecast: pd.DataFrame,
        start_day: int,
        simulate_day_fn: Callable,
        score_day_fn: Callable,
        ml_model: Any,
        tank_capacity: float,
        plant_area: float,
        scenario: str,
        forecast_days: int,
    ) -> float:
        """
        Fast greedy rollout to estimate remaining score.
        Uses single best action per remaining day.
        """
        current_tank = state.tank_level
        current_soil = state.soil_moisture
        lookahead_score = 0.0
        temp_results = list(state.soil_results)
        
        for day_idx in range(start_day, len(df_forecast)):
            day_data = df_forecast.iloc[day_idx]
            next_day_data = df_forecast.iloc[day_idx + 1] if day_idx + 1 < len(df_forecast) else None
            
            best_action_score = float('-inf')
            best_result = None
            
            # Find greedy-best action
            for action in self.ACTIONS:
                tank_result, soil_result, new_tank, new_soil = simulate_day_fn(
                    tank_level_prev=current_tank,
                    moisture_prev=current_soil,
                    action=action,
                    day_data=day_data,
                    next_day_data=next_day_data,
                    ml_model=ml_model,
                    tank_capacity=tank_capacity,
                    plant_area=plant_area,
                    scenario=scenario,
                    irrigation_history=[r.get("irrigation_mm", 0.0) * 8.0 for r in temp_results] + [0.0] * (forecast_days - len(temp_results)),
                )
                
                action_score = score_day_fn(tank_result, soil_result)
                if action_score > best_action_score:
                    best_action_score = action_score
                    best_result = (tank_result, soil_result, new_tank, new_soil)
            
            if best_result:
                _, soil_result, current_tank, current_soil = best_result
                lookahead_score += best_action_score
                temp_results.append(soil_result)
        
        return lookahead_score
    
    def _prune_beam(self, candidates: List[BeamState], beam_width: int) -> List[BeamState]:
        """
        Prune candidates to beam_width, preserving diversity.
        
        Strategy:
        1. Keep top 80% by score
        2. Fill remaining 20% with bucket-diverse states
        """
        if len(candidates) <= beam_width:
            return candidates
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda s: s.cumulative_score, reverse=True)
        
        # Top candidates by score
        top_count = int(beam_width * 0.8)
        top_states = sorted_candidates[:top_count]
        remaining = sorted_candidates[top_count:]
        
        # Bucket remaining by state
        buckets: Dict[Tuple[int, int], List[BeamState]] = {}
        for state in remaining:
            key = state.get_bucket_key()
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(state)
        
        # Add diverse states from underrepresented buckets
        top_buckets = {s.get_bucket_key() for s in top_states}
        diverse_count = beam_width - top_count
        diverse_states: List[BeamState] = []
        
        # Prioritize buckets not in top states
        for bucket_key, bucket_states in buckets.items():
            if len(diverse_states) >= diverse_count:
                break
            if bucket_key not in top_buckets:
                diverse_states.append(bucket_states[0])  # Best from this bucket
        
        # Fill remaining slots with next best from any bucket
        all_remaining = [s for states in buckets.values() for s in states]
        all_remaining.sort(key=lambda s: s.cumulative_score, reverse=True)
        for state in all_remaining:
            if len(diverse_states) >= diverse_count:
                break
            if state not in diverse_states:
                diverse_states.append(state)
        
        return top_states + diverse_states


def create_day_simulator(scenario_adjuster: Callable):
    """
    Factory to create a day simulation function compatible with BeamSearchOptimizer.
    
    Args:
        scenario_adjuster: Function to adjust values based on scenario
    
    Returns:
        simulate_day_fn compatible with optimizer
    """
    from .model_trainer import X_feature_cols
    
    def simulate_day(
        tank_level_prev: float,
        moisture_prev: float,
        action: int,
        day_data: pd.Series,
        next_day_data: Optional[pd.Series],
        ml_model: Any,
        tank_capacity: float,
        plant_area: float,
        scenario: str,
        irrigation_history: List[float],
    ) -> Tuple[Dict, Dict, float, float]:
        """Simulate a single day and return results."""
        
        date = day_data["date"].strftime("%Y-%m-%d") if hasattr(day_data["date"], "strftime") else str(day_data["date"])
        temp_today = float(day_data["Tmax_c"])
        rain_today = float(day_data["rain_sum"])
        water_inflow = float(day_data["total_m3"] * 1000.0)
        
        if next_day_data is not None:
            temp_tomorrow = float(next_day_data["Tmax_c"])
            rain_tomorrow = float(next_day_data["rain_sum"])
        else:
            temp_tomorrow = temp_today
            rain_tomorrow = rain_today
        
        # Apply scenario adjustments
        temp_today = scenario_adjuster(scenario, temp_today, "temp_today")
        rain_today = scenario_adjuster(scenario, rain_today, "rain_today")
        temp_tomorrow = scenario_adjuster(scenario, temp_tomorrow, "temp_tomorrow")
        rain_tomorrow = scenario_adjuster(scenario, rain_tomorrow, "rain_tomorrow")
        water_inflow = scenario_adjuster(scenario, water_inflow, "inflow_l")
        
        day_of_year = day_data["date"].timetuple().tm_yday if hasattr(day_data["date"], "timetuple") else 1
        month = day_data["date"].month if hasattr(day_data["date"], "month") else 1
        
        pumps_today = action
        Qout_l = pumps_today * 1.5
        Qout_l_final = min(Qout_l, tank_level_prev + water_inflow)
        Qout_mm = Qout_l_final / plant_area
        
        irrigation_last_h_days = sum(irrigation_history[-7:]) if irrigation_history else 0.0
        
        feature_row = {
            "soil_moisture": moisture_prev,
            "water_level": tank_level_prev,
            "soil_moisture_prev": moisture_prev,
            "water_level_prev": tank_level_prev,
            "day_of_year": day_of_year,
            "month": month,
            "temp_today": temp_today,
            "rain_today": rain_today,
            "temp_tomorrow": temp_tomorrow,
            "rain_tomorrow": rain_tomorrow,
            "irrigation_last_h_days": irrigation_last_h_days,
            "pump_usage": pumps_today,
            "calculated_total_l": water_inflow,
            "irrigation_today": Qout_l_final,
        }
        
        X_infer = pd.DataFrame([feature_row])[X_feature_cols]
        pred = ml_model.predict(X_infer)
        
        # Physics-based tank calculation (mass balance)
        raw_tank = tank_level_prev + water_inflow - Qout_l_final
        new_tank = max(0.0, min(tank_capacity, raw_tank))
        overflow_l = max(0.0, raw_tank - tank_capacity)
        
        # Physics-based soil calculation with ML residual
        k_irrig = 1.5
        k_rain = 0.3
        k_evap = 0.4
        temp_factor = max(0.0, temp_today / 30.0)
        
        soil_physics = (
            moisture_prev
            + k_irrig * Qout_mm
            + k_rain * rain_today
            - k_evap * temp_factor
        )
        
        # ML provides bounded residual correction
        ml_residual = float(pred[0][1])
        max_correction = 0.2 * max(1.0, soil_physics)
        clamped_residual = max(-max_correction, min(max_correction, ml_residual))
        new_soil = max(0.0, min(100.0, soil_physics + clamped_residual))
        
        tank_result = {
            "date": date,
            "tank_l": new_tank,
            "qin_l": water_inflow,
            "qout_l": Qout_l_final,
            "overflow_l": overflow_l,
            "pump_usage": pumps_today,
        }
        
        soil_result = {
            "date": date,
            "soil_mm": new_soil,
            "irrigation_mm": Qout_mm,
        }
        
        return tank_result, soil_result, new_tank, new_soil
    
    return simulate_day


def create_day_scorer(soil_threshold: float, alpha: float = 10.0, beta: float = 0.1, 
                      gamma: float = 0.3, delta: float = 1.0):
    """
    Factory to create a day scoring function.
    
    Returns per-day score contribution for beam search.
    """
    def score_day(tank_result: Dict, soil_result: Dict) -> float:
        soil_mm = soil_result["soil_mm"]
        irrigation_mm = soil_result["irrigation_mm"]
        overflow_l = tank_result["overflow_l"]
        tank_l = tank_result["tank_l"]
        
        moisture_deficit = max(0.0, soil_threshold - soil_mm)
        
        return (
            - alpha * moisture_deficit
            - beta * irrigation_mm
            - delta * overflow_l
            + gamma * tank_l * 0.1  # Normalize tank contribution per day
        )
    
    return score_day
