# Kaayko Paddle Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-97.40%25%20R²-brightgreen.svg)](https://github.com/tommyvercetti76/kaayko-paddle-intelligence)

> **Professional machine learning system for paddle safety prediction**  
> **This is the ML training engine - web APIs are in separate kaayko-api project**

## 🎯 What This Does

**Simple:** ML training system for paddle safety prediction models.

**Technical:** HistGradientBoosting training engine achieving 97.40% R² accuracy. Generates models for production use in other systems.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/tommyvercetti76/kaayko-paddle-intelligence.git
cd kaayko-paddle-intelligence
pip install -r requirements.txt  # ONE file installs everything
```

### Basic Usage
```python
# Train new ML models
python src/kaayko_trainer_superior_v1.py

# That's it! Pure ML training system.
```

## 📁 What's In This Repo

### Core System (`src/`)
- **`kaayko_trainer_superior_v1.py`** - ML training system (97.40% R² champion)

### Models (`models/`)
- **`README.md`** - Model generation and management guide
- **`model_metadata.json`** - Model performance metrics
- **`scaler.pkl`** - Feature scaling parameters
- *Production models locally generated (49MB - too big for GitHub)*

### Data Collection (`data-collection/`)
- **`README.md`** - Complete data collection system guide
- **`scripts/`** - Weather data collection tools
- **Professional HydroLAKES integration for 1.4M+ lakes**

### Documentation (`docs/`)
- **`CURRENT_WORKING_SYSTEM.md`** - System overview
- **Component-specific guides and references**

## 🏆 Performance

**Current Champion: HistGradientBoosting**
- **Accuracy:** 97.40% R²
- **Error:** 3.57 RMSE  
- **Training Data:** 1.93M records, 37GB dataset
- **Global Coverage:** 4,905+ lakes across continents

**Algorithm Comparison:**
```
HistGradientBoosting  ✅ 97.40% R² (Champion)
RandomForest         ✅ 96.97% R² (Excellent)
ExtraTrees           ✅ 96.45% R² (Strong)
GradientBoosting     ✅ 96.13% R² (Good)
Ridge/ElasticNet     ❌ Failed on complex patterns
```

## 🔧 Advanced Features

### ML Training
```bash
# Train new models with latest data
python src/kaayko_trainer_superior_v1.py

# Handles algorithm comparison, safety logic, fast interrupts
# Automatically saves best performing models to models/ directory
```

### Safety Features
- **Temperature constraints:** Prevents unrealistic scores in freezing conditions
- **Wind safety limits:** Adjusts for dangerous wind speeds
- **Seasonal adjustments:** Accounts for seasonal safety variations
- **Fast interrupts:** Ctrl+C handling during long training sessions

## 📊 Training Data

**Scale:** 260+ million weather data points  
**Sources:** WeatherAPI.com professional integration  
**Coverage:** Global lakes across all continents  
**Format:** Clean CSV with comprehensive feature engineering

## 🛠️ Development

### Requirements
- **Python 3.8+**
- **All dependencies in one file:** `requirements.txt`
- **Sections:** Core production, ML training, data collection, development tools

### Code Quality
- **Black** code formatting
- **pytest** testing framework  
- **Professional error handling**
- **Comprehensive logging**

### Testing
```bash
# Run test suite
python -m pytest

# Validate model performance
python tests/test_model_validation.py
```

## 📚 Documentation

- **[System Overview](docs/CURRENT_WORKING_SYSTEM.md)** - Architecture and components
- **[Data Collection Guide](data-collection/README.md)** - Professional data collection
- **[Model Management](models/README.md)** - Model generation and deployment
- **[Contributing Guide](CONTRIBUTING.md)** - Development workflow

## 🤝 Contributing

Professional contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

**Setup:**
```bash
git clone <your-fork>
cd kaayko-paddle-intelligence
pip install -r requirements.txt  # Includes all dev tools
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Kaayko Paddle Intelligence** - Professional machine learning for water activity safety.

*Built with ❤️ for paddle safety worldwide*

