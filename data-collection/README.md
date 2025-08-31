# Kaayko Weather Data Collection System

Professional-grade weather data collection infrastructure for machine learning training datasets. Built for large-scale historical weather data acquisition from global lake locations.

## 📋 Overview

This system collects historical weather data from **1.4+ million lake polygons** worldwide using the HydroLAKES database. The collected data powers Kaayko's paddle safety prediction models with comprehensive weather patterns, seasonal variations, and geographic coverage.

**Key Capabilities:**
- ✅ **1.4M+ Lakes** - Complete HydroLAKES polygon database support
- ✅ **Global Coverage** - All continents with intelligent geographic filtering
- ✅ **Production Scale** - 100+ requests/minute with adaptive rate limiting
- ✅ **Resume Support** - Automatic detection and skipping of existing data
- ✅ **Smart Filtering** - Lake size, importance, and regional prioritization
- ✅ **Professional Output** - Clean CSV format ready for ML training

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt  # Main repo requirements include all data collection deps

# All required packages included: requests, pandas, geopy, rich, colorama
```

### 2. WeatherAPI Setup

1. Sign up at [WeatherAPI.com](https://www.weatherapi.com/) (Free tier: 1M calls/month)
2. Get your API key from the dashboard
3. Set environment variable:

```bash
export KAAYKO_WEATHER_API_KEY="your_api_key_here"
```

### 3. Data Source Setup

Download the HydroLAKES polygon database:

```bash
# Option 1: Direct download (recommended)
wget https://wp.geomar.de/geobon/files/2016/11/HydroLAKES_polys_v10.gdb.zip
unzip HydroLAKES_polys_v10.gdb.zip

# Option 2: Manual download
# Visit: https://www.hydrosheds.org/products/hydrolakes
# Download: HydroLAKES_polys_v10.gdb (Polygon format)
```

## 🎯 Usage

### Interactive Collection (Recommended)

```bash
cd scripts
python interactive_collector.py
```

**Features:**
- 🎨 Beautiful menu-driven interface
- 📊 Real-time progress tracking with colors
- ⚡ Smart collection strategies (10K, 50K, 200K, or all lakes)
- 🛡️ Built-in error handling and resume capability

### Direct Collection

```bash
# Collect from HydroLAKES database
python hydrolakes_collector.py --source /path/to/HydroLAKES_polys_v10.gdb \
                              --start-date 2023-01-01 \
                              --end-date 2023-12-31 \
                              --collect-weather

# Limit collection for testing
python hydrolakes_collector.py --source /path/to/HydroLAKES_polys_v10.gdb \
                              --limit 1000 \
                              --collect-weather
```

### CSV Input Support

```bash
# Use existing lake coordinates CSV
python hydrolakes_collector.py --source lakes.csv \
                              --format csv \
                              --collect-weather
```

## 📁 Data Sources

### HydroLAKES Database

**Source:** [HydroSHEDS - HydroLAKES](https://www.hydrosheds.org/products/hydrolakes)  
**Format:** Geodatabase (.gdb) or Shapefile  
**Size:** ~2.4GB (polygon data)  
**Lakes:** 1.4+ million global water bodies  

**Required Fields:**
- `Hylak_id` - Unique lake identifier
- `Lake_name` - Lake name (if available)  
- `Country` - ISO country code
- `Continent` - Continent identifier
- `Lake_area` - Lake surface area (km²)
- `geometry` - Polygon coordinates

### WeatherAPI.com

**Endpoint:** `https://api.weatherapi.com/v1/history.json`  
**Features Used:**
- Historical weather data (up to 2 years)
- Daily weather observations
- No AQI/alerts (faster responses)

**Required Parameters:**
```
key: Your API key
q: lat,lng coordinates  
dt: Date (YYYY-MM-DD)
end_dt: End date (for ranges)
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export KAAYKO_WEATHER_API_KEY="your_weatherapi_key"

# Optional customization
export KAAYKO_RPM_LIMIT="100"          # API requests per minute
export KAAYKO_THREADS="12"             # Concurrent processing threads
export KAAYKO_OUTPUT_DIR="./output"    # Data output directory
```

### Collection Parameters

```python
# Automatic filtering by lake importance
MIN_LAKE_AREA = 0.5      # km² - Skip very small lakes
MAX_THREADS = 12         # Concurrent processing threads  
API_RPM_LIMIT = 100      # WeatherAPI requests per minute
REQUEST_TIMEOUT = 12     # HTTP request timeout (seconds)
```

## 📊 Output Structure

```
output/
├── collected_data/
│   ├── hydrolakes_weather/
│   │   ├── Lake_Tahoe/
│   │   │   ├── 2023-01.csv
│   │   │   ├── 2023-02.csv
│   │   │   └── ...
│   │   └── Superior/
│   │       ├── 2023-01.csv
│   │       └── ...
├── coordinates/
│   ├── extracted_coordinates.csv
│   └── lakes_metadata.json
└── logs/
    ├── collection.log
    └── errors.log
```

### Weather Data Schema

Each CSV file contains daily weather observations:

```csv
date,temperature_c,wind_speed_kph,humidity,pressure_hpa,visibility_km,cloud_cover,precip_mm,uv,dew_point_c,feelslike_c,gust_kph
2023-01-01,15.2,12.5,65,1013.25,10.0,25,0.0,3,8.7,14.8,18.2
2023-01-02,16.1,8.3,72,1015.80,9.5,40,2.1,2,9.2,15.9,12.1
```

### Coordinates Schema

Lake coordinate extractions:

```csv
lake_name,lat,lng,hylak_id,country,continent,area_km2
Lake Tahoe,39.0968,-120.0324,109951,US,North America,495.0
Superior,47.7511,-87.9987,109876,US,North America,82414.0
```

## 🔄 Resume & Progress Tracking

### Automatic Resume

The system automatically detects existing data and skips completed collections:

```python
# Checks for existing monthly CSV files
# Resumes from the next uncollected month
# Logs skipped files for transparency
```

### Progress Monitoring

```bash
# Real-time progress tracking
✅ Completed: Lake_Tahoe (12/12 months) 
⏳ Processing: Superior (3/12 months)
📊 Progress: 1,547/10,000 lakes (15.47%)
```

### Manual Resume

```bash
# Resume specific lake collection
python hydrolakes_collector.py --source hydrolakes.gdb \
                              --start-date 2023-06-01 \
                              --collect-weather

# Skip to specific lakes (by ID range)
python hydrolakes_collector.py --source hydrolakes.gdb \
                              --offset 5000 \
                              --limit 1000
```

## 🛡️ Error Handling & Rate Limiting

### Intelligent Rate Limiting

```python
# Token bucket algorithm with adaptive backoff
# Automatic 429 (rate limit) detection and retry
# Dynamic RPM adjustment based on API response times
# Graceful degradation during network issues
```

### Network Resilience

```python
# Connection timeout handling
# Automatic retry with exponential backoff  
# Network connectivity validation
# Graceful shutdown on Ctrl+C (double-tap for force exit)
```

### Data Integrity

```python
# CSV validation before writing
# Atomic file operations (temp → final)
# Duplicate detection and prevention
# Corrupted file recovery
```

## 📈 Performance & Scaling

### Throughput

- **100 RPM** - Standard WeatherAPI free tier
- **1000 RPM** - Professional tier (upgrade recommended)
- **~8,640 lakes/day** - At 100 RPM with 1 year of data per lake

### Resource Usage

- **Memory:** ~100MB for coordinate processing
- **CPU:** Multi-threaded (12 cores recommended)
- **Storage:** ~50KB per lake-year of weather data
- **Network:** ~2-5KB per API request

### Optimization Tips

```bash
# 1. Use SSD storage for faster I/O
# 2. Increase threads on multi-core systems
# 3. Filter by minimum lake size to focus on important lakes
# 4. Use date ranges to collect specific seasons
# 5. Monitor API quota usage
```

## 🔧 Troubleshooting

### Common Issues

**API Key Invalid:**
```bash
❌ Error: Invalid API key
✅ Solution: Check KAAYKO_WEATHER_API_KEY environment variable
```

**Rate Limit Exceeded:**
```bash
❌ Error: 429 Too Many Requests  
✅ Solution: System auto-retries, or reduce KAAYKO_RPM_LIMIT
```

**HydroLAKES File Not Found:**
```bash
❌ Error: Cannot access geodatabase
✅ Solution: Download HydroLAKES_polys_v10.gdb and update path
```

**Network Connectivity:**
```bash
❌ Error: Connection timeout
✅ Solution: Check internet connection, system retries automatically  
```

### Debug Mode

```bash
# Enable verbose logging
python hydrolakes_collector.py --source hydrolakes.gdb --debug

# Check collection logs
tail -f output/logs/collection.log
```

## 📜 License

MIT License - Part of Kaayko Paddle Intelligence System.

---

**Professional Support:** For enterprise deployment, custom filtering, or high-volume collection needs, contact the Kaayko development team.
