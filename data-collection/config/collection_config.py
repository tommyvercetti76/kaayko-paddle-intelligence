#!/usr/bin/env python3
"""
Data Collection Configuration for Kaayko Paddle Intelligence
============================================================

This configuration file defines the settings for large-scale weather data collection
used to build the 260+ million data point dataset powering Kaayko's ML models.

SECURITY NOTE: Never commit API keys to version control.
Use environment variables or separate credential files.
"""
import os
from pathlib import Path

# ============================================================================
# API CONFIGURATION
# ============================================================================

# WeatherAPI.com configuration (Professional Plan recommended)
WEATHER_API_KEY = os.getenv("KAAYKO_WEATHER_API_KEY", "YOUR_WEATHERAPI_KEY_HERE")
WEATHER_BASE_URL = "https://api.weatherapi.com/v1/history.json"

# Rate limiting (adjust based on your API plan)
API_RPM_LIMIT = int(os.getenv("KAAYKO_RPM_LIMIT", "100"))  # Requests per minute
MAX_THREADS = int(os.getenv("KAAYKO_THREADS", "12"))       # Concurrent threads
REQUEST_TIMEOUT = 12                                        # Request timeout (seconds)

# ============================================================================
# DATA COLLECTION PATHS
# ============================================================================

# Base directory for raw monthly data collection
RAW_DATA_DIR = Path(os.getenv("KAAYKO_DATA_DIR", "./data/raw_monthly"))

# Processed data output
PROCESSED_DATA_DIR = Path("./data/processed")
PACKAGED_DATA_DIR = Path("./data/packaged")

# Model and log directories  
MODELS_DIR = Path("./models")
LOGS_DIR = Path("./logs")

# ============================================================================
# COLLECTION PARAMETERS
# ============================================================================

# Threading and performance
THREAD_JITTER_RANGE = (0.05, 0.25)  # Random delay between requests
MAX_RETRIES = 3                      # Retry failed requests
BACKOFF_SECONDS = 2                  # Exponential backoff base

# API parameters (optimize payload size)
API_ADDITIONAL_PARAMS = {
    "aqi": "no",      # Skip air quality data
    "alerts": "no"    # Skip weather alerts
}

# ============================================================================
# LAKE DATABASE CONFIGURATION
# ============================================================================

# Minimum distance between lakes (km) to ensure diversity
MIN_LAKE_DISTANCE_KM = 5.0

# Date range for historical data collection
DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = "2025-08-30"

# Lake data sources
LAKE_SOURCES = {
    "paddlingout_base": 17,      # Core PaddlingOut locations
    "global_expansion": 4900,    # Additional global lakes
    "total_coverage": "5000+"    # Target total coverage
}

def validate_config():
    """Validate configuration before starting data collection."""
    errors = []
    
    if WEATHER_API_KEY == "YOUR_WEATHERAPI_KEY_HERE":
        errors.append("Weather API key not configured")
    
    if API_RPM_LIMIT < 1 or API_RPM_LIMIT > 1000:
        errors.append("API rate limit should be between 1-1000 requests/minute")
    
    return len(errors) == 0, errors
