from rest_framework import serializers


class ForecastSummarySerializer(serializers.Serializer):
    plan = serializers.CharField()
    total_irrigation_l = serializers.FloatField()
    start_fill_l = serializers.FloatField()
    soil_threshold_mm = serializers.FloatField()
    days_below_threshold = serializers.IntegerField()


class ForecastSeriesItemSerializer(serializers.Serializer):
    date = serializers.DateTimeField()
    tank_l = serializers.FloatField()
    soil_mm = serializers.FloatField()
    irr_l = serializers.FloatField()
    irrigate = serializers.BooleanField()


class ForecastPlansSerializer(serializers.Serializer):
    summary = ForecastSummarySerializer()
    series = ForecastSeriesItemSerializer(many=True)


class ForecastResponseSerializer(serializers.Serializer):
    best_case = ForecastPlansSerializer()
    average_case = ForecastPlansSerializer()
    worst_case = ForecastPlansSerializer()


# for wrapped response:
from rest_framework import serializers


class TimeValueSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    value = serializers.FloatField()


class CaseSeriesSerializer(serializers.Serializer):
    name = serializers.CharField()  # cases
    value = TimeValueSerializer(many=True)


class ForecastMetricSerializer(serializers.Serializer):
    name = serializers.CharField()  # tank-level or soil-moisture
    values = CaseSeriesSerializer(many=True)


class ActionPointSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    value = serializers.FloatField()  # irr_l
    action = serializers.ChoiceField(choices=["watering"], required=False)


class ActionsByCaseSerializer(serializers.Serializer):
    name = serializers.CharField()
    value = ActionPointSerializer(many=True)


class ForecastWrappedResponseSerializer(serializers.Serializer):
    forecasts = ForecastMetricSerializer(many=True)
    actions = ActionsByCaseSerializer(many=True)


class QuantileResultSerializer(serializers.Serializer):
    model_path = serializers.CharField()
    pinball_loss_water_level = serializers.FloatField()
    pinball_loss_soil_moisture = serializers.FloatField()
    best_params_water_level = serializers.DictField()
    best_params_soil_moisture = serializers.DictField()


class TrainingSerializer(serializers.Serializer):
    len_x = serializers.IntegerField(required=False)
    len_y = serializers.IntegerField(required=False)
    model_type = serializers.CharField(required=False)
    best_params_water_level = serializers.DictField(required=False)
    best_params_soil_moisture = serializers.DictField(required=False)
    median_pinball_loss = serializers.FloatField(required=False)

    rows = serializers.IntegerField(required=False)
    train_rows = serializers.IntegerField(required=False)
    test_rows = serializers.IntegerField(required=False)
    quantiles = serializers.DictField(child=QuantileResultSerializer(), required=False)
