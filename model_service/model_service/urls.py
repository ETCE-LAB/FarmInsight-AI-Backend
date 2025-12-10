from django.urls import path
from .views.model_views import get_model_forecast, get_model_params_list, train_model

urlpatterns = [
    path('water/farm-insight', get_model_forecast, name='get_model_forecast'),
    path('water/params', get_model_params_list, name='get_model_params_list'),
    path('water/train', train_model, name='train_model'),
]
