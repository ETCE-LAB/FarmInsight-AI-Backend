<img src="https://github.com/user-attachments/assets/bb514772-084e-439f-997a-badfe089be76" width="300">

# FarmInsight-AI-Backend

A Django-based AI service that provides machine learning predictions for FarmInsight Food Production Facilities.

## Table of Contents

- [The FarmInsight Project](#the-farminsight-project)
- [Overview](#overview)
  - [Built with](#built-with)
- [Features](#features)
- [Development Setup](#development-setup)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## The FarmInsight Project

Welcome to the FarmInsight Project by ETCE!

The FarmInsight platform brings together advanced monitoring of "Food Production Facilities" (FPF), enabling users to
document, track, and optimize every stage of food production seamlessly.

All FarmInsight Repositories:
- <a href="https://github.com/ETCE-LAB/FarmInsight-Dashboard-Frontend">Dashboard-Frontend</a>
- <a href="https://github.com/ETCE-LAB/FarmInsight-Dashboard-Backend">Dashboard-Backend</a>
- <a href="https://github.com/ETCE-LAB/FarmInsight-FPF-Backend">FPF-Backend</a>
- <a href="https://github.com/ETCE-LAB/FarmInsight-AI-Backend">AI-Backend</a>
Link to our productive System:<a href="https://farminsight.etce.isse.tu-clausthal.de"> FarmInsight.etce.isse.tu-clausthal.de</a>

## Overview

The AI-Backend provides predictive models for FarmInsight. It runs as an independent service that the Dashboard-Backend queries periodically for forecasts. Currently supported model types:

- **Water Model**: Predicts water levels and irrigation needs
- **Energy Model**: Forecasts battery state-of-charge (SoC) and energy consumption

The service exposes a standardized REST API that allows easy integration of new model types.

### Built with

[![Python][Python-img]][Python-url] <br>
[![Django][Django-img]][Django-url] <br>
[![Scikit-Learn][Sklearn-img]][Sklearn-url]

## Features

- **Water Level Forecasting**: Predict future water levels including best/average/worst-case scenarios
- **Energy Forecasting**: Battery SoC predictions with multi-scenario support
- **Proactive Actions**: Models can suggest automated actions (e.g., refill triggers)
- **Weather Integration**: Energy models incorporate weather forecast data from Open-Meteo
- **REST API**: Standardized endpoints for easy Dashboard-Backend integration
- **Model Training**: Endpoints to trigger retraining with new data

## Development Setup

### Prerequisites

- Python 3.11 or higher
- `pip` (Python package manager)
- `virtualenv` (recommended for isolated environments)

### Step-by-Step Guide

1. Navigate to the model_service directory:

```bash
cd model_service
```

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the development server:

```bash
python manage.py runserver 8002
```

The default port is `8002` to avoid conflicts with other FarmInsight services.

## API Endpoints

### Water Model

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/water/params` | Returns model configuration and input parameters |
| GET | `/water/farm-insight` | Returns water level forecasts |
| POST | `/water/train` | Triggers model retraining |

### Energy Model

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/energy/params` | Returns model configuration and input parameters |
| GET | `/energy/farm-insight` | Returns battery SoC forecasts with scenarios |
| POST | `/energy/train` | Triggers model retraining |

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/alive` | Returns 200 if service is running |

## Model Training

Trained models are stored in `model_service/model_service/trained_models/`.

To retrain a model with new data, use the respective `/train` endpoint or run the training scripts manually:

```bash
python train_hybrid_model.py  # For water model
```

For detailed documentation on creating custom energy models, see `/docs/ENERGY_MODEL_DOCUMENTATION.md`.

## 🔄 Contribute to FarmInsight

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add new feature'`
4. Push the branch: `git push origin feature/your-feature`
5. Create a pull request.

## License

This project is licensed under the [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) license.

<!-- MARKDOWN LINKS & IMAGES -->
[Python-img]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[Django-img]: https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white
[Django-url]: https://www.djangoproject.com/
[Sklearn-img]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Sklearn-url]: https://scikit-learn.org/

---
For more information or questions, please contact the ETCE-Lab team.
