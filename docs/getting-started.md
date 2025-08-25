# 🚀 Getting Started

## Quick Installation

```bash
git clone https://github.com/tommyvercetti76/kaayko-paddle-intelligence.git
cd kaayko-paddle-intelligence
pip install -r requirements.txt
```

## Basic Usage

```python
from kaayko.predictor import PaddlePredictor

# Initialize predictor
predictor = PaddlePredictor()

# Load a model (or train a new one)
predictor.load_model('path/to/model')

# Make a prediction
weather_data = {
    'temperature': 22.5,
    'wind_speed': 10.2,
    'humidity': 65,
    'pressure': 1013.2
}

safety_score = predictor.predict_safety(weather_data)
print(f"Paddle Safety Score: {safety_score}/5")
```

## Running Examples

```bash
# Run the sample prediction
python examples/predict_paddle_conditions.py

# Train a new model
python training/train_model.py
```

For more details, see the [API Reference](api-reference.md).
