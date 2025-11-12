from typing import List

from rest_framework import views, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from ..serializers.tank_soil_forecast_serializer import ForecastWrappedResponseSerializer
from ..service.forecast_service import model_forecast
from ..utils.response_wrapper import response_wrapper


class ModelView(views.APIView):
    def get_permissions(self):
        if self.request.method == 'GET':
            return [AllowAny()]
        return super().get_permissions()


MODEL_FORECAST_PARAMS = [
    "latitude",
    "longitude",
    "forecast_days",
    "tank_capacity_liters",
    "starting_tank_volume",
    "soil_threshold",
    "scenario",
    "start_soil_moisture"
]

MODEL_CASES: List[str] = ["best-case", "average-case", "worst-case"]

MODEL_ACTIONS = [{"name": "watering", "type": "float"}]

MODEL_FORECASTS = [{"name": "tank-level"}, {"name": "soil-moisture"}]

MODEL_FORECAST_PARAM_DEFS = [
    {"name": "latitude", "type": "static", "input_type": "float", "required": True},
    {"name": "longitude", "type": "static","input_type": "float", "required": True},
    {"name": "forecast_days", "type": "static","input_type": "int", "required": True},
    {"name": "tank_capacity_liters", "type": "static","input_type": "int", "required": True},
    {"name": "starting_tank_volume", "type": "sensor","input_type": "int", "required": True},
    {"name": "soil_threshold", "type": "static", "input_type": "float", "required": True},
    {"name": "scenario", "type": "static", "input_type": "str", "required": True},
    {"name": "start_soil_moisture", "type": "sensor", "input_type": "float", "required": True},
]


@api_view(['GET'])
@permission_classes([AllowAny])
def get_model_forecast(request) -> Response:
    """
    Generate the MODEL_FORECAST and send it to the Dashboard
    :param latitude:
    :param longitude:
    :param forecast_days:
    :param tank_capacity_liters:
    :param starting_tank_volume:
    :param soil_threshold:
    """

    params = {}
    for name in MODEL_FORECAST_PARAMS:
        value = request.query_params.get(name)
        if value is None and not "scenario":
            return Response({"error": f"Missing parameter: {name}"}, status=status.HTTP_400_BAD_REQUEST)
        params[name] = value

    latitude = float(params.get("latitude"))
    longitude = float(params.get("longitude"))
    forecast_days = int(params.get("forecast_days"))

    tank_capacity_liters = int(params.get("tank_capacity_liters"))
    starting_tank_volume = int(float(params.get("starting_tank_volume")))
    soil_threshold = float(params.get("soil_threshold"))
    scenario = str(params.get("scenario"))
    start_soil_moisture = float(params.get("start_soil_moisture"))

    if scenario in MODEL_CASES:
        scenario = [scenario]
    else:
        scenario = MODEL_CASES

    try:
        tank_results, soil_results = model_forecast(
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            W0_l=starting_tank_volume,
            tank_capacity_liters=tank_capacity_liters,
            soil_threshold=soil_threshold,
            scenarios=scenario,
            start_soil_moisture=start_soil_moisture
        )
    except Exception as e:
        return Response({"error": f"Simulation failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    forecast_data = {
        "tank_forecast": tank_results,
        "soil_forecast": soil_results
    }

    response_data = response_wrapper(forecast_data)

    result_serializer = ForecastWrappedResponseSerializer(data=response_data)
    result_serializer.is_valid(raise_exception=True)

    return Response(result_serializer.data, status=status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_model_params_list(request) -> Response:
    return Response(
        {
            "input_parameters": MODEL_FORECAST_PARAM_DEFS,
            "scenarios": [{"name": s} for s in MODEL_CASES],
            "actions": MODEL_ACTIONS,
            "forecasts": MODEL_FORECASTS
        }, status=status.HTTP_200_OK)
