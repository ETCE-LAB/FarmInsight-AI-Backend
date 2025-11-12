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
    action = serializers.ChoiceField(choices=["watering", "none"])

class ActionsByCaseSerializer(serializers.Serializer):
    name = serializers.CharField()
    value = ActionPointSerializer(many=True)

class ForecastWrappedResponseSerializer(serializers.Serializer):
    forecasts = ForecastMetricSerializer(many=True)
    actions = ActionsByCaseSerializer(many=True)
