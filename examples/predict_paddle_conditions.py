#!/usr/bin/env python3
"""
Example: Basic paddle condition prediction

This example shows how to use Kaayko to predict paddle safety conditions
for a specific location and weather scenario.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add kaayko package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kaayko import PaddlePredictor
from kaayko.models import WeatherInput


def main():
    """Demonstrate basic paddle prediction"""
    
    print("🌊 Kaayko Paddle Intelligence - Basic Example")
    print("=" * 50)
    
    # Sample weather conditions for Lake Tahoe, CA
    weather = WeatherInput(
        latitude=39.0968,
        longitude=-120.0324,
        datetime=datetime(2025, 8, 25, 10, 0, 0),
        temp_c=22.5,
        wind_kph=12.0,
        wind_dir=270,
        humidity=65,
        cloud=25,
        uv=7.2,
        precip_mm=0.0,
        pressure_mb=1013.2,
        dew_point_c=15.8,
        feelslike_c=24.1,
        gust_kph=18.5,
        vis_km=16.0,
        is_day=1,
        will_it_rain=0,
        will_it_snow=0
    )
    
    try:
        # Initialize predictor
        print("🔄 Loading Kaayko models...")
        predictor = PaddlePredictor()
        
        # Make prediction
        print("🎯 Predicting paddle conditions for Lake Tahoe...")
        result = predictor.predict(
            latitude=weather.latitude,
            longitude=weather.longitude,
            datetime=weather.datetime,
            weather_data=weather
        )
        
        # Display results
        print("\n📊 PREDICTION RESULTS")
        print("-" * 30)
        print(f"🏄‍♂️ Paddle Score: {result.paddle_score:.1f}/5.0")
        print(f"🎯 Skill Level: {result.skill_level.value}")
        print(f"📈 Safety Rating: {result.safety_rating}")
        print(f"🤖 Model Used: {result.model_used}")
        print(f"📍 Location: {result.location[0]:.4f}, {result.location[1]:.4f}")
        
        print("\n🌡️ WEATHER ANALYSIS")
        print("-" * 30)
        print(f"🌡️ Air Temperature: {result.temperature_c}°C")
        print(f"💧 Water Temperature: {result.estimated_water_temp_c}°C")
        print(f"💨 Wind Speed: {result.wind_kph} km/h")
        print(f"🌊 Wave Height: {result.estimated_wave_height_m:.1f}m")
        print(f"☔ Precipitation: {result.precipitation_mm}mm")
        print(f"☀️ UV Index: {result.uv_index}")
        
        print(f"\n💡 {result.description}")
        print(f"🔮 Confidence: {result.confidence*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Trained models in the models/ directory")
        print("   2. Installed all dependencies: pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
