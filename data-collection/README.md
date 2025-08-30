# 🌊 Kaayko Data Collection System

## **Enterprise-Grade Weather Data Collection Infrastructure**

This directory contains the professional data collection system used to build Kaayko's **260+ million data point** weather dataset - the foundation of our 99.28% accurate paddle safety prediction models.

## ��️ **Architecture Overview**

```
data-collection/
├── scripts/           # Professional collection tools
├── config/           # Secure configuration management  
├── examples/         # Sample data and usage examples
└── README.md         # This documentation
```

## 📊 **Collection Capabilities**

### **Scale & Performance**
- **260+ million data points** collected globally
- **5,000+ lakes** across all continents
- **100 requests/minute** with intelligent rate limiting
- **12-thread optimization** for M1 Max performance
- **Token-bucket algorithm** with adaptive backoff

### **Data Sources**
- **WeatherAPI.com** professional-grade historical data
- **PaddlingOut locations** as core reference points
- **Global lake expansion** using geographic algorithms
- **Multi-year coverage** (2019-2025) for robust training

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Install dependencies
pip install requests pandas pathlib

# Set your WeatherAPI key (never commit to git!)
export KAAYKO_WEATHER_API_KEY="your_api_key_here"
export KAAYKO_DATA_DIR="./data/raw_monthly"
```

### **2. Configure Collection**
```python
from data_collection.config.collection_config import validate_config

# Validate your setup
valid, errors = validate_config()
if valid:
    print("✅ Configuration ready!")
else:
    print(f"❌ Errors: {errors}")
```

### **3. Generate Lake Database**
```bash
cd data-collection/scripts
python generate_global_lakes.py
```

### **4. Collect Weather Data**
```bash
# Enhanced collector (recommended)
python kaaykokollect.py --start-date 2024-01-01 --end-date 2024-12-31

# Rate-limited collector (legacy)
python rate_limited_collector.py
```

## 🔧 **Key Components**

### **Core Scripts**

| **File** | **Purpose** | **Features** |
|----------|-------------|--------------|
| `kaaykokollect.py` | Main data collector | Adaptive rate limiting, resume capability |
| `rate_limited_collector.py` | Legacy collector | Token-bucket algorithm, threading |
| `generate_global_lakes.py` | Lake database generator | Global coverage, distance optimization |

### **Configuration**
- **`collection_config.py`** - Professional configuration management
- **Environment variables** - Secure credential handling
- **Validation system** - Pre-flight configuration checks

## 📈 **Data Collection Strategy**

### **Geographic Coverage**
```
Global Distribution:
├── North America     # USA, Canada, Mexico
├── Europe           # UK, Germany, France, Italy, etc.
├── Asia             # China, Japan, India, Southeast Asia
├── South America    # Brazil, Argentina, Chile
├── Africa           # Major regions and countries
└── Oceania          # Australia, New Zealand
```

### **Collection Parameters**
- **Rate Limiting**: 100 RPM with token-bucket algorithm
- **Threading**: 12 concurrent threads (M1 Max optimized)
- **Retry Logic**: Exponential backoff on failures
- **Data Validation**: Automated quality checks
- **Resume Capability**: Skip already-collected data

## 🛡️ **Security & Best Practices**

### **API Key Management**
```bash
# ✅ Good - Use environment variables
export KAAYKO_WEATHER_API_KEY="your_key_here"

# ❌ Bad - Never hardcode keys in scripts
api_key = "hardcoded_key"  # DON'T DO THIS
```

### **Rate Limiting**
- **Respect API limits** - Configure based on your plan
- **Adaptive backoff** - Automatic adjustment on 429 errors
- **Request optimization** - Minimize payload size with `aqi=no&alerts=no`

### **Data Organization**
- **Per-lake-per-month files** - Efficient I/O and resumption
- **Structured logging** - Comprehensive collection tracking
- **Progress tracking** - JSON-based state management

## 🎯 **Advanced Usage**

### **Custom Lake Lists**
```python
# Create your own lake database
custom_lakes = [
    {"name": "Custom Lake", "lat": 45.0, "lng": -120.0, "region": "Custom"},
    # Add more lakes...
]
```

### **Parallel Collection**
```python
# Adjust threading for your system
export KAAYKO_THREADS=8  # Reduce for lower-end systems
export KAAYKO_THREADS=16 # Increase for high-end systems
```

### **Data Processing Pipeline**
```bash
# 1. Generate lakes
python generate_global_lakes.py

# 2. Collect weather data  
python kaaykokollect.py

# 3. Process for training
# (Connect to your ML training pipeline)
```

## 📚 **Data Schema**

### **Lake Database Format**
```csv
name,lat,lng,type,region,base_lake
Lake Tahoe,39.0968,-120.0324,Lake,USA_California,Lake Tahoe
```

### **Weather Data Format**
```csv
date,temperature_c,wind_speed_kph,humidity,pressure_hpa,visibility_km,...
2024-01-01,15.2,12.5,65,1013.25,10.0,...
```

## 🤝 **Contributing**

Professional contributions to improve the data collection system are welcome:

1. **Performance optimizations** for different hardware
2. **Additional data sources** integration
3. **Enhanced error handling** and recovery
4. **Geographic expansion** algorithms

## 📄 **License**

This data collection system is part of Kaayko Paddle Intelligence and is licensed under MIT License.

---

**Enterprise-grade data collection powering the world's most accurate paddle safety prediction system.**
