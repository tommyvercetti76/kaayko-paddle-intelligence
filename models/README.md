# Kaayko Production Models

## Model Storage Policy

Due to GitHub's file size limitations (100MB max, 50MB warning), large trained models are **excluded from version control** and must be generated locally.

## Available Models

### 📋 Model Metadata (Tracked)
- **`model_metadata.json`** (653B) - Production model specifications and performance metrics
- **`scaler.pkl`** (2.7KB) - Feature scaling parameters

### 🚫 Large Models (Not Tracked - Generate Locally)
- **`kaayko_paddle_model.pkl`** (49MB) - Main production model **[GENERATE LOCALLY]**
- **`random_forest_model.pkl`** (115KB) - Alternative RandomForest model **[GENERATE LOCALLY]**  
- **`ridge_regression_model.pkl`** (1.9KB) - Baseline regression model **[GENERATE LOCALLY]**

## 🚀 Model Generation

### Generate All Models
```bash
cd src
python kaayko_trainer_superior_v1.py
```

This will create:
- `models/kaayko_paddle_model.pkl` - Primary production model (97.40% R²)
- `models/random_forest_model.pkl` - RandomForest backup  
- `models/ridge_regression_model.pkl` - Baseline model
- Updated performance metrics in `model_metadata.json`

### Quick Validation
```bash
# Verify model was created
ls -lh models/kaayko_paddle_model.pkl

# Test model loading
python -c "import joblib; model = joblib.load('models/kaayko_paddle_model.pkl'); print('✅ Model loaded successfully')"
```

## 📊 Production Model Specifications

**Primary Model:** `kaayko_paddle_model.pkl`
- **Algorithm:** RandomForestRegressor (production-optimized)
- **Accuracy:** 97.40% R² coefficient  
- **Features:** 15 engineered weather features
- **Size:** ~49MB (too large for GitHub)
- **Training Data:** 1.93M+ records, 37GB dataset
- **Geographic Coverage:** Global with continental routing

**Performance Metrics:**
```json
{
  "model_name": "RandomForest",
  "score": 0.9928,
  "rmse": 3.57,
  "feature_count": 15,
  "training_samples": "1.93M+",
  "file_size": "49MB"
}
```

## 🔄 Model Usage in Components

### Production API (`src/simple_paddle_api.py`)
```python
model = joblib.load("models/kaayko_paddle_model.pkl")  # Main model
scaler = joblib.load("models/scaler.pkl")              # Feature scaling
```

### Live Scoring (`src/get_paddlingout_scores.py`)  
```python
model = joblib.load("models/kaayko_paddle_model.pkl")
with open("models/model_metadata.json") as f:
    metadata = json.load(f)
```

### Training Pipeline (`src/kaayko_trainer_superior_v1.py`)
```python
# Saves trained model to:
joblib.dump(model, "models/kaayko_paddle_model.pkl")
```

## 🛠️ Development Workflow

### 1. Fresh Clone Setup
```bash
git clone <repository>
cd kaayko-paddle-intelligence
cd src
python kaayko_trainer_superior_v1.py  # Generate models locally
```

### 2. Model Updates
```bash
# Retrain with new data
python kaayko_trainer_superior_v1.py --full-training

# Commit only metadata updates (not the large model files)
git add models/model_metadata.json models/scaler.pkl  
git commit -m "Update model metadata and scaler"
```

### 3. Production Deployment
```bash
# Models must be generated on production servers
python kaayko_trainer_superior_v1.py --production-mode
```

## ⚠️ Important Notes

**❌ Never Commit Large Models:**
- `kaayko_paddle_model.pkl` (49MB) - Exceeds GitHub limits
- `random_forest_model.pkl` (115KB) - Still too large for frequent updates

**✅ Always Commit Small Files:**
- `model_metadata.json` - Performance metrics and specs
- `scaler.pkl` - Essential for feature preprocessing  

**🔧 Model Regeneration Required:**
- After fresh clone
- On production servers  
- When training data changes
- For performance validation

## 📜 License

MIT License - Part of Kaayko Paddle Intelligence System.
