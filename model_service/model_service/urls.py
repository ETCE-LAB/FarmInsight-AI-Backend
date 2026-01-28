from django.urls import path
from .views.model_views import get_model_forecast, get_model_params_list, train_model
from .views.health_views import alive_check
from .views.energy_views import get_energy_params, get_energy_forecast, train_energy_model

urlpatterns = [
    # Water model endpoints
    path('water/farm-insight', get_model_forecast, name='get_model_forecast'),
    path('water/params', get_model_params_list, name='get_model_params_list'),
    path('water/train', train_model, name='train_model'),

    # Energy model endpoints
    path('energy/farm-insight', get_energy_forecast, name='get_energy_forecast'),
    path('energy/params', get_energy_params, name='get_energy_params'),
    path('energy/train', train_energy_model, name='train_energy_model'),

    path('alive', alive_check, name='alive_check'),
]
