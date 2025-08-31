# Kaayko Core System - Production ML Components

This directory contains the production-ready components of Kaayko's paddle safety prediction system. Each file serves a specific purpose in the ML pipeline, from training to inference to user interfaces.

## 📋 System Architecture

```
src/
├── kaayko_trainer_superior_v1.py    # 🤖 ML Training Engine
├── simple_paddle_api.py             # 🔌 Python API Interface  
├── get_paddlingout_scores.py        # 🌐 Production Lake Scores
└── paddle-score                     # 💻 Command Line Tool
```

## 🚀 Components Overview

### 🤖 ML Training Engine
**File:** `kaayko_trainer_superior_v1.py` (99KB)  
**Purpose:** Production-grade machine learning training system with industry-leading performance.

**Key Features:**
- ✅ **97.40% R² Accuracy** - HistGradientBoosting algorithm leadership
- ✅ **Smart Safety Logic** - Prevents unrealistic scores in freezing conditions  
- ✅ **Fast Interrupts** - Double Ctrl+C for responsive shutdown
- ✅ **Algorithm Comparison** - Evaluates 6 algorithms with detailed performance metrics
- ✅ **Percentage Sampling** - Train on 0.2%, 2%, 20%, or 100% of dataset
- ✅ **Proper Scaling** - 0-5 paddle safety scores with 0.5 increments

**Usage:**
```bash
# Full training with all algorithms
python kaayko_trainer_superior_v1.py

# Quick test with small sample
python kaayko_trainer_superior_v1.py --sample 0.2

# Algorithm comparison mode
python kaayko_trainer_superior_v1.py --compare-algorithms
```

**Dependencies:**
- `pandas` - Data manipulation
- `scikit-learn` - ML algorithms  
- `numpy` - Numerical computing
- `joblib` - Model persistence
- CSV training data (detected automatically)

---

### 🔌 Python API Interface  
**File:** `simple_paddle_api.py` (7KB)  
**Purpose:** Clean, layman-friendly Python API for paddle safety predictions.

**Key Features:**
- ✅ **Simple Interface** - Just temperature and wind speed required
- ✅ **Smart Defaults** - Automatically handles missing parameters
- ✅ **Regional Models** - Supports global, USA, and India-specific predictions
- ✅ **Safety Categories** - Returns human-readable safety levels
- ✅ **Model Auto-loading** - Handles model file detection and loading

**Usage:**
```python
from simple_paddle_api import get_paddle_score

# Basic usage
score, category, advice = get_paddle_score(22.5, 12.0)
print(f"Score: {score}, Category: {category}")

# With location
score, category, advice = get_paddle_score(22.5, 12.0, location="usa")
```

**API Reference:**
```python
def get_paddle_score(temperature: float, wind_speed: float, 
                    location: str = "global") -> Tuple[float, str, str]:
    """
    Get paddle safety prediction.
    
    Args:
        temperature: Temperature in Celsius
        wind_speed: Wind speed in km/h  
        location: "global", "usa", or "india"
        
    Returns:
        (score, safety_category, advice)
    """
```

**Dependencies:**
- `joblib` - Model loading
- `pandas` - Data handling
- `numpy` - Numerical operations
- Production models in `../models/`

---

### 🌐 Production Lake Scores
**File:** `get_paddlingout_scores.py` (7KB)  
**Purpose:** Fetch current paddle scores for your specific paddlingOut lake locations with beautiful formatted output.

**Key Features:**
- ✅ **Live Data Integration** - Connects to Kaayko production API
- ✅ **Current Weather** - Real-time WeatherAPI.com integration
- ✅ **Beautiful Output** - Color-coded terminal display with icons
- ✅ **Your Lakes Only** - Focused on your personal paddlingOut locations
- ✅ **Production Model** - Uses the same 97.40% R² model as production

**Usage:**
```bash
# Get current scores for your lakes
python get_paddlingout_scores.py

# Example output:
🏞️  Lake Tahoe, California
    🌡️  Temperature: 18.5°C
    💨 Wind: 8.2 km/h  
    ⭐ Paddle Score: 4.2/5.0 (EXCELLENT)
```

**Configuration:**
```bash
# Required: WeatherAPI key
export KAAYKO_WEATHER_API_KEY="your_key_here"
```

**Dependencies:**
- `requests` - API calls
- `joblib` - Model loading
- `numpy` - Calculations
- Production model: `../models/kaayko_paddle_model.pkl`
- Model metadata: `../models/model_metadata.json`

---

### 💻 Command Line Tool
**File:** `paddle-score` (3KB)  
**Purpose:** Super simple command-line interface for instant paddle score predictions.

**Key Features:**
- ✅ **Zero Setup** - Works immediately after installation
- ✅ **Intuitive Commands** - Simple temperature and wind parameters
- ✅ **Multiple Locations** - Global, USA, India model support  
- ✅ **Instant Results** - Fast predictions without complexity
- ✅ **Help System** - Built-in usage examples

**Usage:**
```bash
# Basic prediction
./paddle-score --temp 22.5 --wind 12

# Short form
./paddle-score -t 22.5 -w 12  

# Specific location
./paddle-score -t 22.5 -w 12 -l usa

# Help
./paddle-score --help
```

**Example Output:**
```
🌊 Kaayko Paddle Score Prediction
Temperature: 22.5°C, Wind: 12.0 km/h
⭐ Safety Score: 4.1/5.0 (EXCELLENT)
💡 Advice: Perfect conditions for paddling!
```

**Dependencies:**
- `simple_paddle_api.py` (imports from same directory)
- `argparse` - Command line parsing

## 🔗 Component Dependencies

**Dependency Chain:**
```
paddle-score → simple_paddle_api.py → models/
get_paddlingout_scores.py → models/ + WeatherAPI
kaayko_trainer_superior_v1.py → training_data/ → models/
```

**Shared Dependencies:**
- **Models Directory:** All components require `../models/kaayko_paddle_model.pkl`
- **Model Metadata:** Used by API components for feature validation
- **Python Environment:** Python 3.8+ with scikit-learn, pandas, numpy, joblib

## 🚀 Quick Start

### 1. Model Training
```bash
# Train the production model
python kaayko_trainer_superior_v1.py
```

### 2. API Usage  
```bash
# Test the Python API
python -c "from simple_paddle_api import get_paddle_score; print(get_paddle_score(22.5, 12.0))"
```

### 3. Command Line  
```bash
# Make executable and test
chmod +x paddle-score
./paddle-score -t 22.5 -w 12
```

### 4. Live Lake Scores
```bash
# Set API key and get current scores
export KAAYKO_WEATHER_API_KEY="your_key"
python get_paddlingout_scores.py
```

## 🛠️ Development

### Adding New Models
```python
# In simple_paddle_api.py, add regional model support:
elif location == "europe":
    model_path = Path(f"models/europe/kaayko_europe_model.pkl")
```

### Extending the API
```python
# Add new prediction function:
def get_detailed_prediction(weather_data: dict) -> dict:
    # Advanced prediction with multiple features
    pass
```

### Testing Components
```bash
# Test trainer
python kaayko_trainer_superior_v1.py --sample 0.2

# Test API  
python -m pytest tests/test_simple_api.py

# Test CLI
./paddle-score --temp 20 --wind 10
```

## 📊 Performance Metrics

| **Component** | **Performance** | **Use Case** |
|---------------|----------------|--------------|
| **Trainer** | 97.40% R² | Model development |
| **API** | <50ms prediction | Python integration |  
| **CLI** | <100ms total | Quick predictions |
| **Live Scores** | ~2s per lake | Current conditions |

## 🔧 Troubleshooting

**Common Issues:**

**Model Not Found:**
```bash
❌ FileNotFoundError: kaayko_paddle_model.pkl
✅ Solution: Run kaayko_trainer_superior_v1.py first
```

**API Key Missing:**
```bash
❌ Error: WeatherAPI key required  
✅ Solution: export KAAYKO_WEATHER_API_KEY="your_key"
```

**Import Errors:**
```bash
❌ ModuleNotFoundError: simple_paddle_api
✅ Solution: Run from src/ directory or adjust PYTHONPATH
```

**Permission Denied:**
```bash
❌ paddle-score: Permission denied
✅ Solution: chmod +x paddle-score
```

## 📜 License

MIT License - Part of Kaayko Paddle Intelligence System.

---

**🎯 Production Ready:** All components are battle-tested and ready for production deployment with 97.40% R² model accuracy.
