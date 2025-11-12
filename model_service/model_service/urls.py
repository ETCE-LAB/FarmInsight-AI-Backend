from django.urls import path
from .views.model_views import get_model_forecast, get_model_params_list

urlpatterns = [
    path('get-farm-insight/water', get_model_forecast, name='get_model_forecast'),
    path('get-farm-insight/water/params', get_model_params_list, name='get_model_params_list'),
]
