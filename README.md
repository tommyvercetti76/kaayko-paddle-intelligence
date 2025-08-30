# Kaayko Paddle Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.28%25-brightgreen.svg)](https://github.com/tommyvercetti76/kaayko-paddle-intelligence)

> **Enterprise-grade paddle safety prediction system powered by 260+ million data points**
> Professional machine learning for water activity safety assessment

## System Overview

Kaayko is a sophisticated AI system that analyzes weather conditions and predicts paddle safety scores for water activities. Built on **260+ million global weather data points** with advanced hierarchical machine learning:

- **Target Accuracy:** 99.28% (Production RandomForest)
- **Skill Level Intelligence:** Beginner to Expert recommendations  
- **Global Coverage:** 4,905+ lakes across 6 continents
- **Hierarchical Routing:** Global -> Continental -> National -> Lake-specific models
- **Advanced Ensemble:** 6-algorithm system (HistGradientBoosting, GradientBoosting, RandomForest, ExtraTrees, Ridge, ElasticNet)

## Performance Metrics

| **Component** | **Accuracy** | **Algorithm** | **Features** |
|---------------|-------------|---------------|--------------|
| **Production Model** | **99.28%** | RandomForest | 15 optimized |
| **Ensemble System** | **97.4% R²** | HistGradientBoosting | Advanced |

## Core Architecture

### Production Prediction Engine
- **[kaayko/predictor.py](kaayko/predictor.py)** - Main prediction orchestrator
- **[kaayko/models.py](kaayko/models.py)** - Professional data validation schemas  
- **[kaayko/kaayko_inference_system.py](kaayko/kaayko_inference_system.py)** - Advanced hierarchical inference
- **[models/kaayko_paddle_model.pkl](models/)** - 99.28% accurate production model (49.3MB)

### Data Collection Infrastructure
- **[data-collection/README.md](data-collection/README.md)** - Complete collection system documentation
- **[data-collection/scripts/kaaykokollect.py](data-collection/scripts/kaaykokollect.py)** - Professional weather collector
- **[data-collection/config/collection_config.py](data-collection/config/collection_config.py)** - Enterprise configuration
- **Production Endpoint:** **https://us-central1-kaaykostore.cloudfunctions.net/api/paddlingOut** (reference data source)

### Advanced Training Pipeline
- **[kaayko_training_suite/](kaayko_training_suite/)** - Professional ML training framework
- **[training/advanced/kaayko_production_training_suite.py](training/advanced/kaayko_production_training_suite.py)** - 6-algorithm ensemble trainer
- **[kaayko/04_inference_router.py](kaayko/04_inference_router.py)** - Intelligent geographic routing

## Quick Start

### Installation
```bash
git clone https://github.com/tommyvercetti76/kaayko-paddle-intelligence.git
cd kaayko-paddle-intelligence
pip install -r requirements.txt
```

### Basic Usage
```python
from kaayko import PaddlePredictor

# Initialize production model (99.28% accurate)
predictor = PaddlePredictor()

# Get paddle safety prediction
score = predictor.predict_conditions(
    temperature=22.5, 
    wind_speed=15, 
    humidity=65,
    lake_name="Lake Tahoe"
)

print(f"Safety Score: {score}")
```

## Advanced Features

### Data Collection System
Professional weather data collection infrastructure:
```bash
# Setup data collection
export KAAYKO_WEATHER_API_KEY="your_weatherapi_key"
cd data-collection/scripts

# Generate global lakes database
python generate_global_lakes.py

# Collect weather data (260M+ data points capability)
python kaaykokollect.py --start-date 2024-01-01 --end-date 2024-12-31
```

### Advanced Training
Enterprise ML training with 6-algorithm ensemble:
```bash
# Install advanced dependencies
pip install -r requirements-advanced.txt

# Run professional training pipeline
python training/advanced/kaayko_production_training_suite.py
```

## Global Data Foundation

### Dataset Infrastructure
- **Scale:** 260+ million weather data points (global historical archive)
- **Coverage:** 4,905+ lakes across all continents  
- **Reference Source:** Kaayko Production API production endpoints
- **Collection Method:** Professional WeatherAPI.com integration with rate limiting

### Geographic Coverage
```
Global Lake Distribution:
├── North America: USA, Canada, Mexico (1,200+ lakes)
├── Europe: UK, Germany, France, Italy, etc. (800+ lakes)
├── Asia: China, Japan, India, Southeast Asia (1,500+ lakes)
├── South America: Brazil, Argentina, Chile (600+ lakes)
├── Africa: Major regions and countries (400+ lakes)
└── Oceania: Australia, New Zealand (400+ lakes)
```

## Documentation

- **[Getting Started](docs/getting-started.md)** - Setup and basic usage guide
- **[Model Training](docs/model-training.md)** - Professional training pipeline documentation
- **[Data Collection](data-collection/README.md)** - Enterprise data collection system
- **[Model Architecture](docs/MODEL_CLARIFICATION.md)** - Detailed model specifications
- **[API Reference](docs/api-reference.md)** - Complete function documentation
- **[Integration Summary](docs/ADVANCED_INTEGRATION_SUMMARY.md)** - System capabilities overview

## Testing & Quality

### Test Suite
- **[tests/test_predictor.py](tests/test_predictor.py)** - Core prediction engine tests
- **[tests/test_model_real_paddlingout.py](tests/test_model_real_paddlingout.py)** - Production validation tests
- **[kaayko_training_suite/data_integrity.py](kaayko_training_suite/data_integrity.py)** - Data validation pipeline

### Production Validation
```bash
# Run test suite
python -m pytest tests/

# Validate against Kaayko Production API production data
python tests/test_model_real_paddlingout.py
```

## Contributing

Professional contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for enterprise development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Kaayko Paddle Intelligence** - Enterprise-grade water safety prediction through advanced machine learning.
