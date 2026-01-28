"""
Energy Model Views for FarmInsight AI Backend

Provides REST endpoints for energy forecasting:
- GET /energy/params - Model configuration (forecasts, actions, scenarios)
- GET /energy/farm-insight - Generate predictions with proactive actions
"""

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from ..service.energy_forecast_service import energy_forecast


# Energy model parameter definitions
ENERGY_FORECASTS = [
    {"name": "battery_soc"},
    {"name": "solar_production"}
]

ENERGY_SCENARIOS = ["expected", "optimistic", "pessimistic"]

# Energy model actions - only connect_grid uses ActionMapping
# Consumer shutdown is handled by Dashboard Backend based on individual consumer thresholds
ENERGY_ACTIONS = [
    {"name": "connect_grid", "type": "boolean"}
]

# Parameter definitions - static types since Dashboard Backend injects live values
ENERGY_PARAMS = [
    {"name": "latitude", "type": "static", "input_type": "float", "default": "51.9", "description": "Location latitude for weather forecast"},
    {"name": "longitude", "type": "static", "input_type": "float", "default": "10.4", "description": "Location longitude for weather forecast"},
    {"name": "forecast_hours", "type": "static", "input_type": "int", "default": "336", "description": "Forecast period in hours (default 14 days)"},
    {"name": "max_solar_output_watts", "type": "static", "input_type": "float", "default": "600", "description": "Maximum solar panel output in watts"},
    {"name": "avg_consumption_watts", "type": "static", "input_type": "float", "default": "50", "description": "Average consumption (auto-injected from consumers)"},
    {"name": "initial_soc_wh", "type": "static", "input_type": "float", "default": "800", "description": "Current battery SoC (auto-injected from battery sensor)"},
    {"name": "battery_max_wh", "type": "static", "input_type": "float", "default": "1600", "description": "Battery maximum capacity in Wh"}
]


@api_view(['GET'])
@permission_classes([AllowAny])
def get_energy_params(request) -> Response:
    """
    Return energy model configuration.
    
    Response format matches what the Dashboard Frontend expects from /params endpoint.
    """
    return Response({
        "forecasts": ENERGY_FORECASTS,
        "scenarios": [{"name": s} for s in ENERGY_SCENARIOS],
        "actions": ENERGY_ACTIONS,
        "input_parameters": ENERGY_PARAMS
    }, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_energy_forecast(request) -> Response:
    """
    Generate energy forecast with proactive actions.
    
    Query parameters (all optional with defaults):
    - latitude: Location latitude (default: 51.9)
    - longitude: Location longitude (default: 10.4)
    - forecast_hours: Hours to forecast (default: 72)
    - max_solar_output_watts: Max solar panel output (default: 600)
    - avg_consumption_watts: Average power consumption (default: 50)
    - initial_soc_wh: Current battery state (default: 800)
    - battery_max_wh: Battery capacity (default: 1600)
    
    Response format matches what the Dashboard Backend expects from /farm-insight endpoint.
    """
    try:
        # Parse parameters with defaults
        latitude = float(request.query_params.get("latitude", 51.9))
        longitude = float(request.query_params.get("longitude", 10.4))
        forecast_hours = int(request.query_params.get("forecast_hours", 336))  # Default 14 days
        max_solar_output_watts = float(request.query_params.get("max_solar_output_watts", 600))
        avg_consumption_watts = float(request.query_params.get("avg_consumption_watts", 50))
        initial_soc_wh = float(request.query_params.get("initial_soc_wh", 800))
        battery_max_wh = float(request.query_params.get("battery_max_wh", 1600))
        
        # Generate forecast
        result = energy_forecast(
            latitude=latitude,
            longitude=longitude,
            forecast_hours=forecast_hours,
            max_solar_output_watts=max_solar_output_watts,
            avg_consumption_watts=avg_consumption_watts,
            initial_soc_wh=initial_soc_wh,
            battery_max_wh=battery_max_wh
        )
        
        return Response(result, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": f"Forecast generation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def train_energy_model(request) -> Response:
    """
    Train the energy prediction model.
    
    Prepares training data from JSON files and weather history,
    then trains a GradientBoosting model for battery SoC prediction.
    
    Returns training metrics and model info.
    """
    try:
        from ..service.train_energy_model_service import train_energy_model_service
        
        result = train_energy_model_service()
        
        return Response({
            "status": "success",
            "message": "Energy model trained successfully",
            "training_results": result
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": f"Training failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

