from ..utils.data_processor import prepare_data_for_training
from ..utils.synthetic_historical_data import prepare_synthetic_training_data
from ..utils.model_trainer import train_quantile_models

USE_SYNTHETIC_DATA = True


def train_water_model():
    prepare_synthetic_training_data() if USE_SYNTHETIC_DATA else prepare_data_for_training()
    return train_quantile_models(USE_SYNTHETIC_DATA)
