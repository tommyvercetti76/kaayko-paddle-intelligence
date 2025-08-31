#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAAYKO – HydroLAKES Polygon Data Collection System (BULLETPROOF VERSION)
------------------------------------------------------------------------
• Reads HydroLAKES polygon geodatabase (.gdb) OR CSV files
• Extracts centroids from polygons for weather data collection
• Supports filtering by lake size, importance, and region
• Uses same rate-limited collection engine as kaaykokollect.py
• Optimized for massive scale (1.4M+ lakes capability)
• BULLETPROOF error handling and keyboard interrupt management

Usage:
    python hydrolakes_collector.py --source /path/to/HydroLAKES_polys_v10.gdb
    python hydrolakes_collector.py --source /path/to/lakes.csv --format csv
    python hydrolakes_collector.py --source /path/to/HydroLAKES_polys_v10.gdb --limit 10000

ENV OVERRIDES:
    KAAYKO_API_KEY, KAAYKO_RPM, KAAYKO_THREADS, KAAYKO_START, KAAYKO_END,
    KAAYKO_OUTPUT_DIR

Author: Kaayko
"""

import os
import sys
import csv
import math
import json
import time
import signal
import logging
import argparse
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Geospatial libraries for polygon processing
try:
    import geopandas as gpd
    import pandas as pd
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    print("⚠️  GeoPandas not available. Install with: pip install geopandas")
    GEOSPATIAL_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────────
#  ANSI Colors / Glyphs
# ────────────────────────────────────────────────────────────────────────────────
class C:
    R = "\033[0m"     # Reset
    B = "\033[1m"     # Bold
    RED = "\033[31m"
    GRN = "\033[32m"
    YEL = "\033[33m"
    BLU = "\033[34m"
    MAG = "\033[35m"
    CYA = "\033[36m"
    GRY = "\033[38;5;240m"

# Icons
OK = "✓"
WARN = "⚠"
FAIL = "✗"
INFO = "ℹ"

# ────────────────────────────────────────────────────────────────────────────────
#  Signal Handling & Cleanup
# ────────────────────────────────────────────────────────────────────────────────

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals gracefully."""
    signal_names = {
        signal.SIGINT: 'SIGINT (Ctrl+C)',
        signal.SIGTERM: 'SIGTERM',
    }
    
    if hasattr(signal, 'SIGHUP'):
        signal_names[signal.SIGHUP] = 'SIGHUP'
    
    signal_name = signal_names.get(signum, f'Signal {signum}')
    
    print(f"\n\n{C.YEL}🛑 {signal_name} received{C.R}")
    print(f"{C.BLU}{INFO} Gracefully shutting down...{C.R}")
    print(f"{C.GRY}Cleaning up resources and saving progress...{C.R}")
    
    # Save any in-progress work
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except:
        pass
    
    print(f"{C.GRN}✓ Clean exit completed{C.R}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, signal_handler)

# ────────────────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────────────────
API_KEY         = os.getenv("KAAYKO_API_KEY", "")
BASE_RPM        = int(os.getenv("KAAYKO_RPM", "100"))
MAX_THREADS     = int(os.getenv("KAAYKO_THREADS", "12"))
OUTPUT_DIR      = os.getenv("KAAYKO_OUTPUT_DIR", "./data/hydrolakes_weather")

# Date range
DEFAULT_START_OLD = "2019-01-01"
DEFAULT_END_OLD = "2024-12-31"

# Calculate date range from 2022 to current date (August 30, 2025)
from datetime import date
today = date(2025, 8, 30)
start_2022 = date(2022, 1, 1)
DEFAULT_START = start_2022.strftime("%Y-%m-%d")  # 2022-01-01
DEFAULT_END = today.strftime("%Y-%m-%d")  # 2025-08-30

START_DATE_STR = os.getenv("KAAYKO_START", DEFAULT_START)
END_DATE_STR = os.getenv("KAAYKO_END", DEFAULT_END)

# HydroLAKES Configuration
HYDROLAKES_PATH = "/Users/Rohan/Downloads/HydroLAKES_polys_v10.gdb/HydroLAKES_polys_v10.gdb"
USA_COUNTRIES = {"United States of America"}
INDIA_COUNTRIES = {"India"}
MIN_LAKE_AREA = 1.0  # km²

# ────────────────────────────────────────────────────────────────────────────────
#  API Key Management
# ────────────────────────────────────────────────────────────────────────────────

def get_api_key():
    """Get API key from environment or user input with bulletproof handling."""
    api_key = os.getenv("KAAYKO_API_KEY", "").strip()
    
    if not api_key:
        print(f"\n{C.YEL}{INFO} WeatherAPI Key Required{C.R}")
        print(f"{C.GRY}╭─────────────────────────────────────────────────╮{C.R}")
        print(f"{C.GRY}│  Get your FREE API key (100K requests/month):  │{C.R}")
        print(f"{C.GRY}│  https://www.weatherapi.com/signup.aspx         │{C.R}")
        print(f"{C.GRY}╰─────────────────────────────────────────────────╯{C.R}")
        
        max_attempts = 3
        attempt = 0
        
        while not api_key and attempt < max_attempts:
            attempt += 1
            try:
                if attempt > 1:
                    print(f"\n{C.YEL}Attempt {attempt}/{max_attempts}{C.R}")
                
                # Handle both regular input and potential piped input
                try:
                    api_key = input(f"\n{C.CYA}Enter your WeatherAPI key: {C.R}").strip()
                except EOFError:
                    print(f"\n{C.RED}{FAIL} EOF detected - cannot read from stdin{C.R}")
                    print(f"{C.YEL}Please set environment variable: export KAAYKO_API_KEY='your_key'{C.R}")
                    sys.exit(1)
                
                # Handle empty input
                if not api_key:
                    if attempt < max_attempts:
                        print(f"{C.YEL}{WARN} API key cannot be empty. Please try again.{C.R}")
                    continue
                
                # Handle whitespace-only input
                if api_key.isspace():
                    if attempt < max_attempts:
                        print(f"{C.YEL}{WARN} API key cannot be whitespace only. Please try again.{C.R}")
                    api_key = ""
                    continue
                
                # Basic validation - WeatherAPI keys are typically 32 characters
                if len(api_key) < 20:
                    if attempt < max_attempts:
                        print(f"{C.YEL}{WARN} API key seems too short (expected ~32 chars, got {len(api_key)}). Please verify.{C.R}")
                        print(f"{C.GRY}Example format: 1234567890abcdef1234567890abcdef{C.R}")
                        
                        # Ask for confirmation on short keys
                        try:
                            confirm = input(f"{C.CYA}Continue with this key anyway? (y/N): {C.R}").strip().lower()
                            if confirm not in ['y', 'yes']:
                                api_key = ""
                                continue
                        except (EOFError, KeyboardInterrupt):
                            raise KeyboardInterrupt
                    else:
                        api_key = ""
                        continue
                
                # Check for obviously invalid patterns
                invalid_patterns = [
                    'your_key_here', 'api_key_here', 'replace_me', 'example',
                    'test', '12345', 'abcd', 'xxxx', 'yyyy', 'sample'
                ]
                
                if any(pattern in api_key.lower() for pattern in invalid_patterns):
                    if attempt < max_attempts:
                        print(f"{C.YEL}{WARN} API key appears to be a placeholder. Please enter your actual key.{C.R}")
                        api_key = ""
                        continue
                    else:
                        api_key = ""
                        continue
                
                # If we get here, key seems reasonable
                print(f"{C.GRY}Key length: {len(api_key)} characters - looks good!{C.R}")
                
            except KeyboardInterrupt:
                print(f"\n{C.YEL}^C{C.R}")
                print(f"\n{C.BLU}{INFO} Collection interrupted by user.{C.R}")
                print(f"{C.GRY}You can also set the key as an environment variable:{C.R}")
                print(f"{C.CYA}export KAAYKO_API_KEY='your_key_here'{C.R}")
                sys.exit(0)
            except Exception as e:
                print(f"\n{C.RED}{FAIL} Unexpected error during input: {e}{C.R}")
                if attempt >= max_attempts:
                    print(f"{C.RED}Maximum attempts reached. Exiting.{C.R}")
                    sys.exit(1)
        
        # Final validation after all attempts
        if not api_key:
            print(f"\n{C.RED}{FAIL} Could not obtain valid API key after {max_attempts} attempts{C.R}")
            print(f"\n{C.YEL}Alternative setup methods:{C.R}")
            print(f"{C.GRY}1. Environment variable: export KAAYKO_API_KEY='your_key'{C.R}")
            print(f"{C.GRY}2. Shell config: echo 'export KAAYKO_API_KEY=your_key' >> ~/.zshrc{C.R}")
            print(f"{C.GRY}3. Get key from: https://www.weatherapi.com/signup.aspx{C.R}")
            sys.exit(1)
    
    # Final sanitization
    api_key = api_key.strip().replace('"', '').replace("'", '').replace(' ', '')
    
    if not api_key:
        print(f"{C.RED}{FAIL} API key became empty after sanitization{C.R}")
        sys.exit(1)
    
    return api_key

# ────────────────────────────────────────────────────────────────────────────────
#  Lake Data Loading Functions
# ────────────────────────────────────────────────────────────────────────────────

def load_lakes_from_gdb(gdb_path: str, limit: int = None, per_country: int = None, usa_lakes: int = None, india_lakes: int = None) -> list:
    """Load lakes from HydroLAKES geodatabase with bulletproof error handling."""
    if not GEOSPATIAL_AVAILABLE:
        raise ImportError(f"{C.RED}GeoPandas required. Install with: pip install geopandas{C.R}")
    
    # Validate input path
    if not gdb_path:
        raise ValueError(f"{C.RED}GDB path cannot be empty{C.R}")
    
    gdb_path = Path(gdb_path)
    if not gdb_path.exists():
        raise FileNotFoundError(f"{C.RED}HydroLAKES file not found: {gdb_path}{C.R}")
    
    if not gdb_path.is_dir():
        raise NotADirectoryError(f"{C.RED}HydroLAKES path is not a directory: {gdb_path}{C.R}")
    
    print(f"{C.CYA}{INFO} Loading HydroLAKES from: {gdb_path}{C.R}")
    
    try:
        # Attempt to read the geodatabase
        print(f"{C.GRY}Reading geodatabase (this may take 30-60 seconds for large datasets)...{C.R}")
        
        try:
            gdf = gpd.read_file(str(gdb_path))
        except Exception as e:
            # Try different layer names if default fails
            layer_candidates = ["HydroLAKES_polys_v10", "HydroLAKES_polys", "lakes", "polygons"]
            
            gdf = None
            for layer_name in layer_candidates:
                try:
                    print(f"{C.YEL}Trying layer: {layer_name}{C.R}")
                    gdf = gpd.read_file(str(gdb_path), layer=layer_name)
                    print(f"{C.GRN}✓ Successfully loaded layer: {layer_name}{C.R}")
                    break
                except:
                    continue
            
            if gdf is None:
                raise RuntimeError(f"{C.RED}Could not load geodatabase. Original error: {str(e)}{C.R}")
        
        if gdf is None or len(gdf) == 0:
            raise ValueError(f"{C.RED}Geodatabase loaded but contains no data{C.R}")
        
        print(f"{C.GRN}{OK} Loaded {len(gdf):,} lake polygons{C.R}")
        
        # Validate essential columns
        required_columns = ['geometry']
        missing_required = [col for col in required_columns if col not in gdf.columns]
        if missing_required:
            raise ValueError(f"{C.RED}Missing required columns: {missing_required}. Available: {list(gdf.columns)}{C.R}")
        
        # Filter by size if available
        if 'Lake_area' in gdf.columns:
            try:
                original_count = len(gdf)
                gdf = gdf[gdf['Lake_area'] >= 0.5]  # Keep lakes ≥ 0.5 km²
                filtered_count = len(gdf)
                print(f"{C.YEL}{INFO} Filtered by size: {original_count:,} → {filtered_count:,} lakes (≥0.5 km²){C.R}")
                
                if filtered_count == 0:
                    print(f"{C.YEL}{WARN} No lakes remaining after size filter. Using all lakes.{C.R}")
                    gdf = gpd.read_file(str(gdb_path))
            except Exception as e:
                print(f"{C.YEL}{WARN} Size filtering failed: {e}. Using all lakes.{C.R}")
        
        # Process regional selection
        if usa_lakes or india_lakes or per_country:
            gdf = process_regional_selection(gdf, per_country, usa_lakes, india_lakes)
        
        # Apply final limit
        if limit and limit < len(gdf):
            gdf = gdf.head(limit)
            print(f"{C.YEL}{INFO} Limited to {limit:,} lakes{C.R}")
        
        # Extract centroids
        return extract_centroids(gdf)
        
    except KeyboardInterrupt:
        print(f"\n{C.YEL}Loading interrupted by user{C.R}")
        sys.exit(0)
    except MemoryError:
        print(f"{C.RED}{FAIL} Not enough memory to load geodatabase{C.R}")
        print(f"{C.YEL}Try using --limit parameter to reduce memory usage{C.R}")
        sys.exit(1)
    except Exception as e:
        print(f"{C.RED}{FAIL} Error loading geodatabase: {str(e)}{C.R}")
        raise

def process_regional_selection(gdf, per_country, usa_lakes, india_lakes):
    """Process regional lake selection with error handling."""
    try:
        print(f"{C.CYA}╭─────────────────────────────────────────╮{C.R}")
        print(f"{C.CYA}│           REGIONAL SELECTION            │{C.R}")
        print(f"{C.CYA}╰─────────────────────────────────────────╯{C.R}")
        
        if 'Country' in gdf.columns and 'Lake_area' in gdf.columns:
            country_groups = gdf.groupby('Country')
            selected_lakes = []
            
            for country, group in country_groups:
                # Determine how many lakes to select for this country
                if country.upper() in ['UNITED STATES', 'USA', 'US'] and usa_lakes:
                    n_lakes = usa_lakes
                    priority_marker = f"{C.B}★{C.R}"
                elif country.upper() in ['INDIA', 'IND'] and india_lakes:
                    n_lakes = india_lakes
                    priority_marker = f"{C.B}★{C.R}"
                elif per_country:
                    n_lakes = per_country
                    priority_marker = " "
                else:
                    continue
                
                # Sort by area and take top N
                top_lakes = group.sort_values('Lake_area', ascending=False).head(n_lakes)
                selected_lakes.append(top_lakes)
                
                # Pretty console output
                flag = "🇺🇸" if country.upper() in ['UNITED STATES', 'USA', 'US'] else "🇮🇳" if country.upper() in ['INDIA', 'IND'] else "🌍"
                print(f"{C.GRY}{priority_marker} {flag} {country}: {len(top_lakes):,} lakes{C.R}")
            
            # Combine all selected lakes
            if selected_lakes:
                gdf = gpd.GeoDataFrame(pd.concat(selected_lakes, ignore_index=True))
                print(f"{C.GRN}{OK} Regional selection: {len(gdf):,} lakes total{C.R}")
            
        elif 'Lake_area' in gdf.columns:
            # No country data, just sort by size
            gdf = gdf.sort_values('Lake_area', ascending=False)
            print(f"{C.YEL}{INFO} Sorted by lake area (largest first){C.R}")
        
        return gdf
        
    except Exception as e:
        print(f"{C.RED}{FAIL} Error in regional processing: {e}{C.R}")
        raise

def extract_centroids(gdf):
    """Extract centroids from geodataframe with error handling."""
    try:
        print(f"\n{C.CYA}{INFO} Calculating lake centroids...{C.R}")
        print(f"{C.CYA}╭─────────────────────────────────────────╮{C.R}")
        print(f"{C.CYA}│           CENTROID EXTRACTION           │{C.R}")
        print(f"{C.CYA}╰─────────────────────────────────────────╯{C.R}")
        
        lakes = []
        total_lakes = len(gdf)
        
        for idx, row in gdf.iterrows():
            try:
                if idx % 1000 == 0:
                    progress = (idx / total_lakes) * 100
                    print(f"{C.GRY}Processing: {idx:,}/{total_lakes:,} ({progress:.1f}%){C.R}")
                
                # Extract centroid
                if hasattr(row.geometry, 'centroid'):
                    centroid = row.geometry.centroid
                    lat, lon = centroid.y, centroid.x
                else:
                    # Fallback for complex geometries
                    bounds = row.geometry.bounds
                    lat = (bounds[1] + bounds[3]) / 2
                    lon = (bounds[0] + bounds[2]) / 2
                
                # Validate coordinates
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    print(f"{C.YEL}{WARN} Invalid coordinates skipped: {lat}, {lon}{C.R}")
                    continue
                
                # Build lake record
                lake_data = {
                    'latitude': lat,
                    'longitude': lon,
                    'lake_name': str(row.get('Lake_name', f'Lake_{idx}')),
                    'country': str(row.get('Country', 'Unknown')),
                    'lake_area': float(row.get('Lake_area', 0.0)),
                    'hylak_id': str(row.get('Hylak_id', f'ID_{idx}'))
                }
                
                lakes.append(lake_data)
                
            except Exception as e:
                print(f"{C.YEL}{WARN} Skipping lake {idx}: {e}{C.R}")
                continue
        
        print(f"{C.GRN}{OK} Successfully processed {len(lakes):,} lake centroids{C.R}")
        return lakes
        
    except KeyboardInterrupt:
        print(f"\n{C.YEL}Centroid extraction interrupted by user{C.R}")
        sys.exit(0)
    except Exception as e:
        print(f"{C.RED}{FAIL} Error extracting centroids: {e}{C.R}")
        raise

# ────────────────────────────────────────────────────────────────────────────────
#  Weather Collection Integration
# ────────────────────────────────────────────────────────────────────────────────

# Weather collection globals
SHOULD_STOP = threading.Event()
MAX_RETRIES = 3
TIMEOUT_SEC = 20
HEADERS = {"User-Agent": "kaayko-weather-collector/1.0"}

class AdaptiveLimiter:
    """Adaptive rate limiter with token bucket for weather API calls."""
    def __init__(self, start_rpm=40, min_rpm=10, max_rpm=120, burst=10):
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm
        self._rpm = max(min(start_rpm, max_rpm), min_rpm)
        self.rate_per_sec = self._rpm / 60.0
        self.capacity = burst
        self.tokens = burst
        self.last = time.time()
        self.lock = threading.Lock()
        self.success_counter = 0

    @property
    def rpm(self):
        with self.lock:
            return int(self._rpm)

    def _refill(self):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate_per_sec)
        self.last = now

    def take(self, n=1):
        while not SHOULD_STOP.is_set():
            with self.lock:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return
            time.sleep(0.03)

    def on_success(self):
        with self.lock:
            self.success_counter += 1
            if self.success_counter % 300 == 0 and self._rpm < self.max_rpm:
                self._rpm = min(self.max_rpm, int(self._rpm * 1.10))
                self.rate_per_sec = self._rpm / 60.0

    def on_throttle(self):
        with self.lock:
            self._rpm = max(self.min_rpm, int(self._rpm * 0.80))
            self.rate_per_sec = self._rpm / 60.0
            self.success_counter = 0

    def snapshot(self):
        with self.lock:
            return {"rpm": int(self._rpm), "tokens": f"{self.tokens:.1f}"}

# Global rate limiter
limiter = AdaptiveLimiter(start_rpm=BASE_RPM)

def slugify(name: str) -> str:
    """Convert lake name to filesystem-safe slug."""
    import re
    return re.sub(r'[^\w\-_.]', '_', name.strip().replace(' ', '_'))[:50]

def month_ranges(start_date: str, end_date: str):
    """Generate monthly date ranges for weather collection."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start.replace(day=1)
    while current <= end:
        # Calculate month end
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        
        month_end = next_month - timedelta(days=1)
        if month_end > end:
            month_end = end
            
        yield (
            current.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
            current.year,
            current.month
        )
        
        current = next_month
        if current > end:
            break

def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def estimate_water_temperature(air_temp_c, latitude, month):
    """Estimate water temperature based on air temperature and seasonal factors."""
    try:
        # Seasonal adjustment based on month
        seasonal_factor = {
            1: -0.3, 2: -0.2, 3: -0.1, 4: 0.0, 5: 0.1, 6: 0.2,
            7: 0.3, 8: 0.2, 9: 0.1, 10: 0.0, 11: -0.1, 12: -0.2
        }
        
        # Latitude adjustment (colder at higher latitudes)
        lat_adjustment = -0.01 * abs(latitude)
        
        # Water temperature estimation (water is slower to change than air)
        water_temp = air_temp_c * 0.8 + seasonal_factor.get(month, 0) * 5 + lat_adjustment
        
        return max(0, water_temp)  # Water can't be below 0°C for liquid
    except:
        return air_temp_c * 0.8  # Fallback

def calculate_paddle_score(row, seasonal_info):
    """Calculate paddling conditions score (0-5 with 0.5 increments)."""
    try:
        temp_c = float(row.get('temp_c', 0))
        humidity = float(row.get('humidity', 50))
        wind_kph = float(row.get('wind_kph', 0))
        precip_mm = float(row.get('precip_mm', 0))
        
        # Temperature score (optimal 18-25°C) - scale 0-5
        if 18 <= temp_c <= 25:
            temp_score = 5.0
        elif 10 <= temp_c < 18:
            temp_score = 3.5 + (temp_c - 10) * 0.1875  # Linear scale from 3.5 to 5
        elif 25 < temp_c <= 35:
            temp_score = 5.0 - (temp_c - 25) * 0.15  # Decrease from 5 to 3.5
        else:
            temp_score = max(0, 2.0 - abs(temp_c - 20) * 0.1)
        
        # Wind score (optimal 0-15 kph) - scale 0-5
        if wind_kph <= 15:
            wind_score = 5.0 - wind_kph * 0.1  # 5.0 at 0 kph, 3.5 at 15 kph
        else:
            wind_score = max(0, 3.5 - (wind_kph - 15) * 0.15)
        
        # Precipitation score - scale 0-5
        if precip_mm == 0:
            precip_score = 5.0
        elif precip_mm <= 2:
            precip_score = 4.0
        elif precip_mm <= 10:
            precip_score = 3.0
        else:
            precip_score = max(1.0, 3.0 - precip_mm * 0.1)
        
        # Humidity score (optimal 40-70%) - scale 0-5
        if 40 <= humidity <= 70:
            humidity_score = 5.0
        else:
            humidity_score = max(3.0, 5.0 - abs(humidity - 55) * 0.075)
        
        # Weighted average (same weights as original)
        paddle_score = (temp_score * 0.4 + wind_score * 0.3 + precip_score * 0.2 + humidity_score * 0.1)
        
        # Ensure range 0-5 and round to nearest 0.5 increment
        paddle_score = max(0, min(5, paddle_score))
        return round(paddle_score * 2) / 2
    except:
        return 2.5  # Neutral score on error

def check_network_connectivity():
    """Check if network is available before starting collection."""
    try:
        response = requests.get("https://api.weatherapi.com", timeout=10)
        return True
    except:
        return False

def fetch_weather_data(lat: float, lon: float, dt_iso: str, end_dt_iso: str, api_key: str):
    """Fetch historical weather data from WeatherAPI."""
    params = {
        "key": api_key,
        "q": f"{lat},{lon}",
        "dt": dt_iso,
        "end_dt": end_dt_iso,
        "aqi": "no",
        "alerts": "no",
    }
    url = "https://api.weatherapi.com/v1/history.json"

    for attempt in range(1, MAX_RETRIES + 1):
        if SHOULD_STOP.is_set():
            return None
        
        limiter.take()
        
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_SEC)
            
            if response.status_code == 429:
                print(f"{C.YEL}{WARN} Rate limited, backing off...{C.R}")
                limiter.on_throttle()
                time.sleep(min(8, 2 ** attempt))
                continue
                
            if response.status_code >= 500:
                print(f"{C.YEL}{WARN} Server error {response.status_code}, retrying...{C.R}")
                time.sleep(min(8, 2 ** attempt))
                continue
                
            response.raise_for_status()
            limiter.on_success()
            return response.json()
            
        except KeyboardInterrupt:
            print(f"\n{C.YEL}Weather collection interrupted by user{C.R}")
            SHOULD_STOP.set()
            return None
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"{C.YEL}{WARN} Network error (attempt {attempt}): Connection timeout/failed{C.R}")
            if attempt >= MAX_RETRIES:
                print(f"{C.RED}{FAIL} Max network retries exceeded - skipping this request{C.R}")
                return None
            # Exponential backoff for network issues
            wait_time = min(30, 5 * (2 ** attempt))
            print(f"{C.GRY}Waiting {wait_time}s before retry...{C.R}")
            time.sleep(wait_time)
        except Exception as e:
            print(f"{C.YEL}{WARN} API error (attempt {attempt}): {e}{C.R}")
            if attempt >= MAX_RETRIES:
                return None
            time.sleep(2 ** attempt)
    
    return None

def process_weather_data(lake, lat, lon, weather_data, region):
    """Process raw weather data into structured format."""
    if not weather_data or 'forecast' not in weather_data:
        return []
    
    rows = []
    location_data = weather_data.get('location', {})
    
    for forecast_day in weather_data['forecast']['forecastday']:
        day_data = forecast_day['day']
        date = forecast_day['date']
        dt = datetime.strptime(date, "%Y-%m-%d")
        
        # Estimate water temperature
        air_temp = day_data.get('avgtemp_c', 0)
        water_temp = estimate_water_temperature(air_temp, lat, dt.month)
        
        # Calculate paddle score
        temp_row = {
            'temp_c': day_data.get('avgtemp_c', 0),
            'humidity': day_data.get('avghumidity', 50),
            'wind_kph': day_data.get('maxwind_kph', 0),
            'precip_mm': day_data.get('totalprecip_mm', 0)
        }
        paddle_score = calculate_paddle_score(temp_row, {'month': dt.month})
        
        row = {
            'lake_name': lake['lake_name'],
            'latitude': lat,
            'longitude': lon,
            'date': date,
            'region': region,
            'country': lake.get('country', 'Unknown'),
            'lake_area_km2': lake.get('lake_area', 0),
            'temp_c': day_data.get('avgtemp_c'),
            'temp_min_c': day_data.get('mintemp_c'),
            'temp_max_c': day_data.get('maxtemp_c'),
            'humidity': day_data.get('avghumidity'),
            'wind_kph': day_data.get('maxwind_kph'),
            'wind_dir': day_data.get('condition', {}).get('text', ''),
            'precip_mm': day_data.get('totalprecip_mm'),
            'condition': day_data.get('condition', {}).get('text', ''),
            'water_temp_c': round(water_temp, 1),
            'paddle_score': paddle_score,
            'year': dt.year,
            'month': dt.month,
            'day': dt.day
        }
        rows.append(row)
    
    return rows

def write_weather_file(lake_slug: str, year: int, month: int, rows: list):
    """Write weather data to monthly CSV file."""
    if not rows:
        return
    
    # Create output directory structure
    lake_dir = Path(OUTPUT_DIR) / lake_slug
    ensure_dir(str(lake_dir))
    
    filename = f"{year}_{month:02d}.csv"
    filepath = lake_dir / filename
    
    fieldnames = [
        'lake_name', 'latitude', 'longitude', 'date', 'region', 'country', 'lake_area_km2',
        'temp_c', 'temp_min_c', 'temp_max_c', 'humidity', 'wind_kph', 'wind_dir',
        'precip_mm', 'condition', 'water_temp_c', 'paddle_score', 'year', 'month', 'day'
    ]
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def weather_file_exists(lake_slug: str, year: int, month: int) -> bool:
    """Check if weather data file already exists."""
    filepath = Path(OUTPUT_DIR) / lake_slug / f"{year}_{month:02d}.csv"
    return filepath.exists()

def collect_weather_for_lake(lake, start_date: str, end_date: str, api_key: str):
    """Collect weather data for a single lake."""
    if SHOULD_STOP.is_set():
        return 0, 0
    
    lat = lake['latitude']
    lon = lake['longitude']
    name = lake['lake_name']
    region = f"HydroLAKES_{lake.get('country', 'Unknown')}"
    
    lake_slug = slugify(name)
    total_rows, months_done = 0, 0

    print(f"\n{C.CYA}{INFO} {C.B}{name}{C.R} {C.GRY}({lat:.4f},{lon:.4f}) • {lake.get('country', 'Unknown')}{C.R}")

    for dt_iso, end_iso, year, month in month_ranges(start_date, end_date):
        if SHOULD_STOP.is_set():
            break

        # Skip if already exists
        if weather_file_exists(lake_slug, year, month):
            print(f"   {C.GRN}{OK} Skip {year}-{month:02d} (exists){C.R}")
            months_done += 1
            continue

        print(f"   {C.BLU}→{C.R} Fetching {dt_iso} → {end_iso} {C.GRY}[rpm:{limiter.rpm}]{C.R}")

        weather_data = fetch_weather_data(lat, lon, dt_iso, end_iso, api_key)
        if weather_data is None:
            print(f"   {C.RED}{FAIL} Failed {year}-{month:02d}{C.R}")
            continue

        rows = process_weather_data(lake, lat, lon, weather_data, region)
        if rows:
            write_weather_file(lake_slug, year, month, rows)
            total_rows += len(rows)
            months_done += 1
            print(f"   {C.GRN}{OK} Saved {len(rows)} days to {year}-{month:02d}.csv{C.R}")
        else:
            print(f"   {C.YEL}{WARN} No data for {year}-{month:02d}{C.R}")

    return total_rows, months_done

# ────────────────────────────────────────────────────────────────────────────────
#  Main Processing (Enhanced)
# ────────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HydroLAKES Data Collection with Weather Integration")
    parser.add_argument("--limit", type=int, help="Limit number of lakes to process")
    parser.add_argument("--per-country", type=int, default=0, help="Number of largest lakes per country")
    parser.add_argument("--usa-lakes", type=int, help="Number of largest lakes from USA")
    parser.add_argument("--india-lakes", type=int, help="Number of largest lakes from India")
    parser.add_argument("--save-lakes", help="Save processed lake coordinates to CSV file")
    parser.add_argument("--collect-weather", action="store_true", help="Collect historical weather data after extracting coordinates")
    parser.add_argument("--start-date", default=DEFAULT_START, help="Weather start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=DEFAULT_END, help="Weather end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    print(f"{C.B}KAAYKO HydroLAKES Integrated Collection{C.R}")
    print(f"{C.CYA}╭═══════════════════════════════════════════════════════════════╮{C.R}")
    print(f"{C.CYA}│                   INTEGRATED COLLECTION                      │{C.R}")
    print(f"{C.CYA}╰═══════════════════════════════════════════════════════════════╯{C.R}")
    print(f"{C.GRY}• HydroLAKES:{C.R} {HYDROLAKES_PATH}")
    print(f"{C.GRY}• Min Area:{C.R} {MIN_LAKE_AREA} km²")
    
    if args.limit:
        print(f"{C.GRY}• Total Limit:{C.R} {args.limit:,} lakes")
    if args.per_country:
        print(f"{C.GRY}• Per Country:{C.R} {args.per_country} largest lakes")
    if args.usa_lakes:
        print(f"{C.B}• 🇺🇸 USA Focus:{C.R} {args.usa_lakes} largest lakes")
    if args.india_lakes:
        print(f"{C.B}• 🇮🇳 India Focus:{C.R} {args.india_lakes} largest lakes")
    
    if args.collect_weather:
        print(f"{C.GRY}• Weather Output:{C.R} {OUTPUT_DIR}")
        print(f"{C.GRY}• Date Range:{C.R} {args.start_date} to {args.end_date}")
    
    try:
        # Step 1: Load and process HydroLAKES data
        print(f"\n{C.BLU}{INFO} Loading HydroLAKES polygons...{C.R}")
        lakes = load_hydrolakes_data(
            limit=args.limit,
            per_country=args.per_country,
            usa_lakes=args.usa_lakes,
            india_lakes=args.india_lakes
        )
        
        print(f"{C.GRN}{OK} Loaded {len(lakes):,} lakes for processing{C.R}")
        
        # Step 2: Save lake coordinates if requested
        if args.save_lakes:
            save_lakes_to_csv(lakes, args.save_lakes)
            print(f"{C.GRN}{OK} Lake coordinates saved to: {args.save_lakes}{C.R}")
        
        # Step 3: Weather collection if requested
        if args.collect_weather:
            # Check network connectivity first
            print(f"\n{C.BLU}{INFO} Checking network connectivity...{C.R}")
            if not check_network_connectivity():
                print(f"{C.RED}{FAIL} Network connectivity issue - cannot reach WeatherAPI{C.R}")
                print(f"{C.YEL}Please check your internet connection and try again{C.R}")
                sys.exit(1)
            print(f"{C.GRN}{OK} Network connectivity confirmed{C.R}")
            
            # Get API key for weather collection
            api_key = get_weather_api_key()
            
            print(f"\n{C.BLU}{INFO} Starting weather data collection...{C.R}")
            print(f"{C.GRY}Processing {len(lakes):,} lakes from {args.start_date} to {args.end_date}{C.R}")
            
            total_rows = 0
            total_months = 0
            processed = 0
            failed_lakes = 0
            
            for i, lake in enumerate(lakes, 1):
                if SHOULD_STOP.is_set():
                    print(f"\n{C.YEL}Weather collection interrupted{C.R}")
                    break
                
                print(f"\n{C.CYA}[{i:,}/{len(lakes):,}]{C.R}", end=" ")
                
                rows, months = collect_weather_for_lake(lake, args.start_date, args.end_date, api_key)
                if rows == 0 and months == 0:
                    failed_lakes += 1
                
                total_rows += rows
                total_months += months
                processed += 1
                
                if processed % 10 == 0:
                    print(f"\n{C.GRY}Progress: {processed}/{len(lakes)} lakes • {total_rows:,} weather records • {total_months} months • {failed_lakes} failed{C.R}")
            
            print(f"\n{C.GRN}🌤️  WEATHER COLLECTION COMPLETE{C.R}")
            print(f"{C.GRY}• Processed: {processed:,} lakes{C.R}")
            print(f"{C.GRY}• Failed: {failed_lakes:,} lakes{C.R}")
            print(f"{C.GRY}• Weather records: {total_rows:,}{C.R}")
            print(f"{C.GRY}• Months collected: {total_months:,}{C.R}")
            print(f"{C.GRY}• Output directory: {OUTPUT_DIR}{C.R}")
        else:
            print(f"\n{C.GRN}📍 COORDINATE EXTRACTION COMPLETE{C.R}")
            print(f"{C.GRY}• Processed: {len(lakes):,} lakes{C.R}")
            print(f"{C.YEL}• Use --collect-weather to fetch historical weather data{C.R}")
        
    except KeyboardInterrupt:
        print(f"\n{C.YEL}Collection interrupted by user{C.R}")
        SHOULD_STOP.set()
        sys.exit(0)
    except Exception as e:
        print(f"\n{C.RED}{FAIL} Collection failed: {e}{C.R}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def get_weather_api_key():
    """Get WeatherAPI key from environment or user input."""
    api_key = os.getenv("WEATHERAPI_KEY", "").strip()
    
    if not api_key:
        print(f"\n{C.YEL}{WARN} WeatherAPI key not found in environment{C.R}")
        print(f"{C.GRY}Please get your free API key from: https://www.weatherapi.com/signup.aspx{C.R}")
        
        while True:
            try:
                api_key = input(f"{C.CYA}Enter WeatherAPI key: {C.R}").strip()
                if api_key:
                    break
                print(f"{C.YEL}API key cannot be empty{C.R}")
            except KeyboardInterrupt:
                print(f"\n{C.YEL}Operation cancelled{C.R}")
                sys.exit(0)
    
    print(f"{C.GRN}{OK} WeatherAPI key configured{C.R}")
    return api_key

def save_lakes_to_csv(lakes: list, filename: str):
    """Save lake data to CSV file."""
    fieldnames = ['latitude', 'longitude', 'lake_name', 'country', 'lake_area', 'hylak_id']
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lakes)

def load_hydrolakes_data(limit=None, per_country=0, usa_lakes=None, india_lakes=None):
    """Load and process HydroLAKES data with filtering."""
    try:
        print(f"{C.GRY}Reading HydroLAKES geodatabase...{C.R}")
        # Specify the layer name for the geodatabase
        gdf = gpd.read_file(HYDROLAKES_PATH, layer='HydroLAKES_polys_v10')
        print(f"{C.GRN}{OK} Loaded {len(gdf):,} lake polygons{C.R}")
        
        # Filter by minimum area
        gdf = gdf[gdf['Lake_area'] >= MIN_LAKE_AREA]
        print(f"{C.GRN}{OK} Filtered to {len(gdf):,} lakes >= {MIN_LAKE_AREA} km²{C.R}")
        
        # Extract centroids and country data
        print(f"{C.GRY}Computing centroids and processing...{C.R}")
        
        # Get centroids
        centroids = gdf.geometry.centroid
        
        # Prepare lake data
        lakes = []
        for idx, row in gdf.iterrows():
            if SHOULD_STOP.is_set():
                break
                
            centroid = centroids.iloc[idx]
            lake_data = {
                'latitude': centroid.y,
                'longitude': centroid.x,
                'lake_name': str(row.get('Lake_name', f'Lake_{row.get("Hylak_id", idx)}')).strip(),
                'country': str(row.get('Country', 'Unknown')).strip(),
                'lake_area': float(row.get('Lake_area', 0)),
                'hylak_id': row.get('Hylak_id', idx)
            }
            lakes.append(lake_data)
        
        # Sort by area (largest first)
        lakes.sort(key=lambda x: x['lake_area'], reverse=True)
        
        # Apply country-specific filters for ALL lakes in USA and India
        filtered_lakes = []
        
        if usa_lakes or usa_lakes is None:  # Get ALL USA lakes if not specified
            usa_candidates = [l for l in lakes if l['country'] in USA_COUNTRIES]
            filtered_lakes.extend(usa_candidates)
            print(f"{C.GRN}{OK} Selected ALL {len(usa_candidates):,} USA lakes{C.R}")
        
        if india_lakes or india_lakes is None:  # Get ALL India lakes if not specified  
            india_candidates = [l for l in lakes if l['country'] in INDIA_COUNTRIES]
            filtered_lakes.extend(india_candidates)
            print(f"{C.GRN}{OK} Selected ALL {len(india_candidates):,} India lakes{C.R}")
        
        # If no specific country filtering, apply per_country logic
        if not filtered_lakes and per_country > 0:
            countries = {}
            for lake in lakes:
                country = lake['country']
                if country not in countries:
                    countries[country] = []
                if len(countries[country]) < per_country:
                    countries[country].append(lake)
            
            country_lakes = []
            for country_list in countries.values():
                country_lakes.extend(country_list)
            filtered_lakes.extend(country_lakes)
            print(f"{C.GRN}{OK} Selected {len(country_lakes)} lakes ({per_country} per country){C.R}")
        
        # Use filtered lakes if we have them, otherwise use all lakes
        if filtered_lakes:
            lakes = filtered_lakes
            # Remove duplicates that might occur from overlapping filters
            seen = set()
            unique_lakes = []
            for lake in lakes:
                lake_id = (lake['latitude'], lake['longitude'], lake['hylak_id'])
                if lake_id not in seen:
                    seen.add(lake_id)
                    unique_lakes.append(lake)
            lakes = unique_lakes
            print(f"{C.GRN}{OK} Final selection: {len(lakes):,} unique lakes{C.R}")
        
        # Apply global limit if specified
        if limit:
            lakes = lakes[:limit]
            print(f"{C.YEL}Applied limit: {len(lakes):,} lakes{C.R}")
        
        return lakes
        
    except Exception as e:
        print(f"{C.RED}{FAIL} Error loading HydroLAKES data: {e}{C.R}")
        raise

if __name__ == "__main__":
    main()
