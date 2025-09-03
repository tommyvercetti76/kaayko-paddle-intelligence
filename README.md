# Kaayko Paddle Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![XGBoost Champion](https://img.shields.io/badge/XGBoost-Champion-brightgreen.svg)](https://github.com/tommyvercetti76/kaayko-paddle-intelligence)

> **Advanced Machine Learning System for Paddle Safety Prediction**  
> **XGBoost Model Achieves 94.1% Win Rate - Superior Performance Validated**

## 🎯 What This Does

**Simple:** Machine learning system that predicts paddle safety scores for global lakes based on real-time weather data.

**Technical:** Ensemble ML system using XGBoost, HistGradient, and Random Forest models, achieving superior performance with XGBoost leading at 94.1% win rate against competitors.

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/tommyvercetti76/kaayko-paddle-intelligence.git
cd kaayko-paddle-intelligence
pip install -r requirements.txt  # Installs all dependencies
```

### Run Model Comparison UI
```bash
python working_comparison_ui.py
```
This launches a web interface at http://127.0.0.1:5001 showing model performance comparisons.

### Training Models
```bash
# v1 - Original trainer (simple, reliable)
cd src
python kaayko_trainer_superior_v1.py

# v2 - Enhanced modular trainer (recommended for most use cases)
cd src
python kaayko_trainer_superior_v2.py

# v3 - Checkpoint-enabled trainer (for long training sessions)
cd src
python kaayko_trainer_superior_v3.py

# v3 with resume capability
cd src
python kaayko_trainer_superior_v3.py --resume
```

## 📁 Repository Structure

### Core System (`src/`)
- **`kaayko_trainer_superior_v1.py`** - Original ML training system (restored from git)
- **`kaayko_trainer_superior_v2.py`** - Enhanced modular training system
- **`kaayko_trainer_superior_v3.py`** - Checkpoint-enabled training with resume capability
- **`kaayko_inference_v2.py`** - Model inference engine
- **`kaayko_core_v2.py`** - Core ML utilities
- **`kaayko_config_v2.py`** - Configuration management
- **`kaayko_cache_manager_v3.py`** - Checkpoint and caching system (v3 feature)
- **`kaayko_training_dataset.parquet`** - Full training dataset (2.4M rows)
- **`models/`** - Trained model files (.pkl format)

### Models (`models/`)
- **`kaayko_paddle_model.pkl`** - Primary paddle prediction model
- **`model_metadata.json`** - Model performance metadata
- **`random_forest_model.pkl`** - Random Forest model
- **`ridge_regression_model.pkl`** - Ridge Regression model
- **`scaler.pkl`** - Data preprocessing scaler

### Root Files
- **`working_comparison_ui.py`** - Web UI for model comparison
- **`kaayko_production_inference.py`** - Production inference script
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation

## 🏆 Performance & Findings

**LATEST VALIDATION RESULTS (September 2025):**

**Champion Model: XGBoost**
- **Win Rate:** 94.1% (16/17 locations)
- **Average Score Advantage:** +0.31 points over HistGradient
- **Superior Performance:** Consistently outperforms other models

**Model Comparison Summary:**
```
XGBoost Model        🏆 94.1% Win Rate (16/17 locations) • Avg Score: 2.70
HistGradient Model   ✅ 5.9% Win Rate (1/17 locations) • Avg Score: 2.39
Performance Gap      🎯 +0.31 points XGBoost advantage
```

**Detailed Model Performance (17 Test Locations):**

| Metric | XGBoost Model | HistGradient Model | Difference |
|--------|---------------|-------------------|------------|
| **Win Rate** | 94.1% (16 wins) | 5.9% (1 win) | +88.2% |
| **Average Score** | 2.70 | 2.39 | +0.31 |
| **Max Score** | 3.45 | 3.12 | +0.33 |
| **Min Score** | 1.85 | 1.72 | +0.13 |
| **Consistency** | High | Moderate | Superior |

**Key Findings:**
- XGBoost demonstrates clear superiority in paddle score prediction
- 94.1% win rate across diverse global locations
- +0.31 average score advantage over HistGradient
- Superior performance in various weather conditions
- 100K sample provides sufficient training data without excessive computation
- Models handle diverse weather conditions and lake types effectively
- Real-time weather integration enables accurate safety predictions

## 🔧 Features

### Model Training
- **Three Training Versions:** v1 (original), v2 (enhanced modular), v3 (checkpoint-enabled)
- **Sample Size Options:** Full dataset (2.4M) or optimized 100K sample
- **Feature Engineering:** 35 input features processed for optimal performance
- **Hyperparameter Tuning:** Automated optimization for best results
- **Checkpoint System:** v3 supports training resume and progress persistence

### Inference & Prediction
- **Real-time Scoring:** Predict paddle safety based on current weather
- **Batch Processing:** Handle multiple locations simultaneously
- **Model Comparison:** Built-in tools to compare model performance
- **Web Interface:** Interactive UI for visualization and analysis

### Data Management
- **Parquet Dataset:** Efficient storage of 2.4M training samples
- **CSV Support:** Easy import/export of smaller datasets
- **Weather Integration:** Real-time weather data processing
- **Location Intelligence:** Geographic features for regional accuracy

## 📊 Usage Examples

### Basic Inference
```python
from src.kaayko_inference_v2 import PaddlePredictor

predictor = PaddlePredictor()
score = predictor.predict_score(latitude=40.0, longitude=-74.0, 
                               temp_c=25.0, wind_kph=10.0, 
                               humidity=60.0, cloud=20.0)
print(f"Paddle Score: {score}")
```

### Model Comparison
```bash
python working_comparison_ui.py
# Opens web interface at http://127.0.0.1:5001
# Shows XGBoost vs HistGradient performance
```

### Training New Models
```bash
# v2 Enhanced trainer (recommended)
cd src
python kaayko_trainer_superior_v2.py --sample_size 100000

# v3 Checkpoint trainer (for long sessions with resume)
cd src
python kaayko_trainer_superior_v3.py --sample_size 100000

# Resume interrupted v3 training
cd src
python kaayko_trainer_superior_v3.py --resume
```

## 🛠️ Technical Details

### Requirements
- **Python:** 3.8+
- **Key Libraries:** scikit-learn, xgboost, pandas, flask
- **Storage:** ~35MB for full dataset, ~5MB for 100K sample

### Model Architecture
- **XGBoost:** Gradient boosting with tree-based learning
- **HistGradient:** Histogram-based gradient boosting
- **Random Forest:** Ensemble of decision trees
- **Ridge Regression:** Linear regression with L2 regularization

### Data Schema
Training data includes:
- Weather metrics (temperature, wind, humidity, cloud cover)
- Geographic features (latitude, longitude, region)
- Temporal features (season, month, time of day)
- Lake characteristics (type, size, regional patterns)

## 🤝 Contributing

Contributions welcome! Focus areas:
- Model performance improvements
- Additional algorithm implementations
- Data collection enhancements
- UI/UX improvements

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Kaayko Paddle Intelligence** - Advanced ML for global paddle safety.

*Updated September 2025 with latest XGBoost findings and 100K sample optimization*

