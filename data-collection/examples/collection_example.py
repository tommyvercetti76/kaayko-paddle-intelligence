#!/usr/bin/env python3
"""
Kaayko Data Collection - Example Usage
======================================

This example demonstrates how to use the Kaayko data collection system
to gather weather data for paddle safety model training.

BEFORE RUNNING:
1. Set your WeatherAPI key: export KAAYKO_WEATHER_API_KEY="your_key"
2. Install requirements: pip install -r requirements.txt
3. Configure your data directory: export KAAYKO_DATA_DIR="./data"
"""

import os
import sys
from pathlib import Path

# Add the config directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "config"))

from collection_config import validate_config, get_collection_config

def main():
    """Example data collection workflow"""
    
    print("🌊 Kaayko Data Collection Example")
    print("=" * 50)
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    try:
        valid, errors = validate_config()
        if not valid:
            print(f"❌ Configuration errors: {errors}")
            print("Please set your KAAYKO_WEATHER_API_KEY environment variable")
            return
        print("✅ Configuration valid!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Step 2: Show current configuration
    print("\n2. Current configuration:")
    config = get_collection_config()
    for key, value in config.items():
        if "key" in key.lower() and value != "YOUR_WEATHERAPI_KEY_HERE":
            value = "***CONFIGURED***"
        print(f"   {key}: {value}")
    
    # Step 3: Example data collection workflow
    print("\n3. Example collection workflow:")
    print("   📊 Generate lake database:")
    print("      cd data-collection/scripts")
    print("      python generate_global_lakes.py")
    print()
    print("   🌡️ Collect weather data:")
    print("      python kaaykokollect.py --start-date 2024-01-01 --end-date 2024-03-31")
    print()
    print("   🔄 Process for training:")
    print("      # Connect to your ML training pipeline")
    
    # Step 4: Check sample data
    sample_lakes = Path(__file__).parent / "global_lakes_sample.csv"
    if sample_lakes.exists():
        print(f"\n4. Sample lake database available:")
        print(f"   📁 {sample_lakes}")
        
        # Show first few lakes
        with open(sample_lakes, 'r') as f:
            lines = f.readlines()[:6]  # Header + 5 lakes
            print("   Sample lakes:")
            for line in lines:
                print(f"      {line.strip()}")
            print(f"   ... and {len(open(sample_lakes).readlines())-6} more lakes")
    
    print("\n🚀 Ready to start collecting data!")
    print("   Set your API key and run the collection scripts above.")

if __name__ == "__main__":
    main()
