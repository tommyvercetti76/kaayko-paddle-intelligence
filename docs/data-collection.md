# 🌍 Data Collection

## Overview

Kaayko collects weather data from global lakes to train paddle safety prediction models.

## Data Sources

- **Primary**: WeatherAPI.com (Current data)
- **Historical**: OpenWeatherMap (Backup source)
- **Coverage**: 4,900+ lakes across 7 continents
- **Update Frequency**: Hourly data collection

## Global Lake Network

### Current Sample Lakes (8 locations)

| Lake | Country | Climate | Data Points |
|------|---------|---------|-------------|
| Lake Mead | USA | Desert | 2,160 |
| Lake Michigan | USA | Continental | 2,160 |
| Lake Minnewanka | Canada | Mountain | 2,160 |
| Lake Murray | USA | Subtropical | 2,160 |
| Windermere | UK | Maritime | 2,160 |
| Yellowstone Lake | USA | Alpine | 2,160 |
| Washoe Lake | USA | High Desert | 2,160 |
| West Lake | China | Humid Subtropical | 2,160 |

**Total**: 17,280 data points across diverse climates

## Data Format

Each weather record contains:

```json
{
  "timestamp": "2025-08-24T14:30:00Z",
  "location": "Lake_Mead",
  "temperature_c": 35.2,
  "wind_speed_kmh": 12.5,
  "humidity_percent": 25,
  "pressure_hpa": 1013.2,
  "visibility_km": 10.0,
  "condition": "Sunny",
  "safety_score": 4.2,
  "skill_level": "Intermediate"
}
```

## Collection Pipeline

```bash
# Run data collection
python scripts/collect_weather_data.py

# Validate collected data  
python scripts/validate_data.py

# Process for training
python scripts/process_training_data.py
```

## Quality Assurance

- **Validation**: Automated data quality checks
- **Redundancy**: Multiple API sources for reliability
- **Historical**: 3+ months per location minimum
- **Geographic**: Diverse climate representation

## Expanding Coverage

To add new lakes:

1. Update `lake_coordinates.json` with new locations
2. Run collection script: `python scripts/collect_weather_data.py --lake NEW_LAKE`  
3. Validate data quality: `python scripts/validate_data.py`
4. Retrain models: `python training/train_model.py`

## API Rate Limits

- **WeatherAPI.com**: 1M calls/month
- **Collection Rate**: 1 call per lake per hour
- **Estimated Runtime**: 2 days for full collection
- **Storage**: ~4.3MB for 8 lakes × 3 months
