"""
Energy Model Training Service for FarmInsight

Orchestrates the training of the energy prediction model.
"""

from ..utils.energy_data_processor import prepare_energy_training_data
from ..utils.energy_model_trainer import train_energy_model


def train_energy_model_service() -> dict:
    """
    Train the energy model by preparing data and running training.
    
    :return: Dictionary with training results
    """
    # Prepare training data (loads JSON, fetches weather, creates features)
    print("Step 1: Preparing training data...")
    df, data_path = prepare_energy_training_data()
    print(f"Training data ready: {len(df)} samples at {data_path}")
    
    # Train the model
    print("\nStep 2: Training model...")
    result = train_energy_model(use_gradient_boosting=True)
    
    return result
