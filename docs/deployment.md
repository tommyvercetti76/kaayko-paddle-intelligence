# 🐳 Deployment

## Overview

Kaayko Paddle Intelligence can be deployed in multiple environments for production use.

## Local Deployment

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/tommyvercetti76/kaayko-paddle-intelligence.git
cd kaayko-paddle-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run predictions
python examples/predict_paddle_conditions.py
```

## Docker Deployment

### Build Image

```bash
# Build Docker image
docker build -t kaayko-paddle-intelligence .

# Run container
docker run -p 8000:8000 kaayko-paddle-intelligence
```

### Docker Compose

```yaml
version: '3.8'
services:
  kaayko:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WEATHER_API_KEY=${WEATHER_API_KEY}
    volumes:
      - ./data:/app/data
```

## Cloud Deployment

### AWS Lambda

```bash
# Package for Lambda
pip install -r requirements.txt -t ./lambda-package
cp -r kaayko ./lambda-package/
cd lambda-package && zip -r ../kaayko-lambda.zip .
```

### Google Cloud Functions

```bash
# Deploy to GCP
gcloud functions deploy kaayko-predict \
  --runtime python38 \
  --trigger-http \
  --entry-point predict_safety
```

### Azure Functions

```bash
# Deploy to Azure
func azure functionapp publish kaayko-functions
```

## API Deployment

### FastAPI Server

```python
from fastapi import FastAPI
from kaayko.predictor import PaddlePredictor

app = FastAPI(title="Kaayko Paddle Intelligence API")
predictor = PaddlePredictor()

@app.post("/predict")
async def predict_safety(weather_data: dict):
    safety_score = predictor.predict_safety(weather_data)
    skill_level = predictor.predict_skill_level(weather_data)
    
    return {
        "safety_score": safety_score,
        "skill_level": skill_level
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

## Environment Variables

Required environment variables:

```bash
export WEATHER_API_KEY="your-weather-api-key"
export MODEL_PATH="path/to/trained/model"
export LOG_LEVEL="INFO"
```

## Performance Tuning

### Production Optimizations

- **Model Loading**: Cache models in memory
- **Batch Processing**: Process multiple predictions together
- **Caching**: Redis for frequently accessed data
- **Load Balancing**: Multiple API instances

### Resource Requirements

- **Memory**: 512MB minimum, 1GB recommended
- **CPU**: 1 core minimum, 2+ cores for high traffic
- **Storage**: 100MB for models and cache
- **Network**: API access for weather data

## Monitoring

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Security

- **API Keys**: Store in environment variables
- **HTTPS**: Use SSL certificates in production
- **Rate Limiting**: Implement request throttling
- **Input Validation**: Sanitize all inputs
