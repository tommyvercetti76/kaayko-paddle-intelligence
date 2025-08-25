# 🔧 API Reference

## Core Classes

### PaddlePredictor

The main prediction interface for Kaayko Paddle Intelligence.

```python
from kaayko.predictor import PaddlePredictor

predictor = PaddlePredictor()
```

#### Methods

- `predict_safety(weather_data: dict) -> float`: Predict paddle safety score (1-5)
- `predict_skill_level(weather_data: dict) -> str`: Recommend skill level
- `load_model(model_path: str)`: Load a trained model
- `train_model(training_data: str)`: Train a new model

### WeatherDataProcessor

Processes and validates weather data inputs.

```python
from kaayko.data import WeatherDataProcessor

processor = WeatherDataProcessor()
```

#### Methods

- `validate_data(data: dict) -> bool`: Validate weather data format
- `normalize_features(data: dict) -> dict`: Normalize weather features
- `extract_features(data: dict) -> list`: Extract ML features

## Data Format

Weather data should be provided as a dictionary with the following keys:

```python
{
    'temperature': float,       # °C
    'wind_speed': float,       # km/h  
    'humidity': int,           # %
    'pressure': float,         # hPa
    'visibility': float,       # km (optional)
    'cloud_cover': int        # % (optional)
}
```

## Example Usage

```python
from kaayko.predictor import PaddlePredictor

# Initialize
predictor = PaddlePredictor()

# Weather data
weather = {
    'temperature': 25.0,
    'wind_speed': 15.5,
    'humidity': 70,
    'pressure': 1015.2
}

# Get predictions
safety = predictor.predict_safety(weather)
skill = predictor.predict_skill_level(weather)

print(f"Safety Score: {safety}")
print(f"Recommended Skill Level: {skill}")
```
