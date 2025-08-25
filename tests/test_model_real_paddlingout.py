#!/usr/bin/env python3
"""
Test the trained Kaayko model against REAL paddlingOut API lakes
Fetches actual lakes from your paddlingOut API for validation
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import requests
import time
from datetime import datetime

# Your paddlingOut API endpoints
PADDLINGOUT_API_URL = "https://us-central1-kaaykostore.cloudfunctions.net/api/paddlingOut"
API_KEY = "a0ede903980f45c4a27183708252308"

def load_trained_model():
    """Load the trained Kaayko model"""
    model_path = "models/kaayko_paddle_model.pkl"
    metadata_path = "models/model_metadata.json"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print("🤖 Loading trained Kaayko model...")
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"✅ Model loaded: {metadata['model_name']}")
    print(f"📊 Training R²: {metadata['score']:.4f}")
    print(f"🎯 Features: {len(metadata['feature_names'])}")
    print(f"⏰ Trained: {metadata['training_timestamp']}")
    
    return model, metadata

def fetch_real_paddlingout_lakes():
    """Fetch REAL lakes from your paddlingOut API"""
    print("🌊 Fetching REAL lakes from paddlingOut API...")
    
    try:
        response = requests.get(PADDLINGOUT_API_URL, timeout=15)
        response.raise_for_status()
        lakes_data = response.json()
        
        print(f"✅ Successfully fetched data from paddlingOut API")
        
        # Parse the lake data
        lakes = []
        for lake in lakes_data:
            if 'location' in lake and 'latitude' in lake['location'] and 'longitude' in lake['location']:
                lakes.append({
                    'name': lake.get('title', lake.get('lakeName', 'Unknown')),
                    'id': lake.get('id', ''),
                    'lat': float(lake['location']['latitude']),
                    'lon': float(lake['location']['longitude']),
                    'subtitle': lake.get('subtitle', ''),
                    'text': lake.get('text', '')
                })
        
        print(f"✅ Parsed {len(lakes)} REAL paddlingOut lakes:")
        for i, lake in enumerate(lakes, 1):
            print(f"   {i:2d}. {lake['name']} ({lake['subtitle']})")
        
        return lakes
        
    except Exception as e:
        print(f"❌ Error fetching from paddlingOut API: {e}")
        return None

def fetch_current_weather(lat, lon, lake_name):
    """Fetch current weather data for testing"""
    url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}&aqi=no"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data['current']
        
        # Extract weather features matching training data
        weather_data = {
            'temperature_c': current['temp_c'],
            'wind_speed_kph': current['wind_kph'],
            'humidity': current['humidity'],
            'pressure_hpa': current['pressure_mb'],
            'visibility_km': current['vis_km'],
            'cloud_cover': current['cloud'],
            'precip_mm': current['precip_mm'],
            'uv': current['uv'],
            'dew_point_c': current.get('dewpoint_c', current['temp_c'] - 5),
            'feelslike_c': current['feelslike_c'],
            'gust_kph': current['gust_kph'],
            'condition': current['condition']['text'],
            'location': lake_name
        }
        
        return weather_data
        
    except Exception as e:
        print(f"❌ Error fetching weather for {lake_name}: {e}")
        return None

def engineer_test_features(weather_data, lake_id=0):
    """Engineer features for a single test sample"""
    # Base weather features
    features = {
        'temperature_c': weather_data['temperature_c'],
        'wind_speed_kph': weather_data['wind_speed_kph'],
        'humidity': weather_data['humidity'],
        'pressure_hpa': weather_data['pressure_hpa'],
        'visibility_km': weather_data['visibility_km'],
        'cloud_cover': weather_data['cloud_cover'],
        'precip_mm': weather_data['precip_mm'],
        'uv': weather_data['uv'],
        'dew_point_c': weather_data['dew_point_c'],
        'feelslike_c': weather_data['feelslike_c'],
        'gust_kph': weather_data['gust_kph']
    }
    
    # Derived features (matching training)
    features['temp_comfort'] = 1 if 15 <= features['temperature_c'] <= 25 else 0
    
    # Wind category (0=calm, 1=moderate, 2=strong, 3=dangerous)
    wind_kph = features['wind_speed_kph']
    if wind_kph <= 15:
        features['wind_category'] = 0
    elif wind_kph <= 30:
        features['wind_category'] = 1
    elif wind_kph <= 50:
        features['wind_category'] = 2
    else:
        features['wind_category'] = 3
    
    features['visibility_good'] = 1 if features['visibility_km'] > 5 else 0
    features['lake_encoded'] = lake_id  # Encode lake ID
    
    return features

def calculate_reference_score(weather_data):
    """Calculate reference paddle score using rule-based approach"""
    score = 3.0
    
    temp = weather_data['temperature_c']
    if 18 <= temp <= 26:
        score += 1.0
    elif temp < 5:
        score -= 2.0
    elif temp > 35:
        score -= 1.5
    
    wind = weather_data['wind_speed_kph']
    if wind > 30:
        score -= 2.0
    elif wind > 50:
        score -= 1.5
    elif wind > 20:
        score -= 0.5
    
    precip = weather_data['precip_mm']
    if precip > 10:
        score -= 1.5
    elif precip > 5:
        score -= 0.5
    
    return max(1.0, min(5.0, score))

def test_real_paddlingout_lakes():
    """Test model against REAL paddlingOut API lakes"""
    print("🏄‍♂️ TESTING KAAYKO MODEL AGAINST REAL PADDLINGOUT API LAKES")
    print("=" * 80)
    
    # Fetch REAL lakes from your API
    lakes = fetch_real_paddlingout_lakes()
    
    if not lakes:
        print("❌ Could not fetch lakes from paddlingOut API")
        return []
    
    # Load model
    model, metadata = load_trained_model()
    feature_names = metadata['feature_names']
    
    print(f"\n🎯 Testing ALL {len(lakes)} REAL paddlingOut lakes...")
    print(f"📊 Expected features: {feature_names}")
    
    results = []
    
    for i, lake in enumerate(lakes, 1):
        print(f"\n🏞️  {i:2d}/{len(lakes)} - Testing {lake['name']} ({lake['subtitle']})...")
        
        # Fetch current weather
        weather = fetch_current_weather(lake['lat'], lake['lon'], lake['name'])
        
        if not weather:
            print(f"   ❌ Skipped due to weather API error")
            continue
        
        # Engineer features
        features = engineer_test_features(weather, i-1)  # lake_encoded
        
        # Create feature vector matching training format
        feature_vector = [features[fname] for fname in feature_names]
        X_test = np.array(feature_vector).reshape(1, -1)
        
        # Model prediction
        ml_prediction = float(model.predict(X_test)[0])
        
        # Reference calculation
        rule_prediction = calculate_reference_score(weather)
        
        # Store results
        result = {
            'lake': lake['name'],
            'lake_id': lake.get('id', ''),
            'subtitle': lake.get('subtitle', ''),
            'lat': lake['lat'],
            'lon': lake['lon'],
            'ml_prediction': ml_prediction,
            'rule_prediction': rule_prediction,
            'difference': abs(ml_prediction - rule_prediction),
            'weather': {
                'temp_c': weather['temperature_c'],
                'wind_kph': weather['wind_speed_kph'],
                'condition': weather['condition'],
                'humidity': weather['humidity'],
                'visibility_km': weather['visibility_km'],
                'precip_mm': weather['precip_mm'],
                'uv': weather['uv']
            }
        }
        
        results.append(result)
        
        # Display result
        print(f"   🤖 ML Prediction: {ml_prediction:.2f}/5.0")
        print(f"   📐 Rule-based:    {rule_prediction:.2f}/5.0")
        print(f"   📊 Difference:    {result['difference']:.3f}")
        print(f"   🌡️  Weather: {weather['temperature_c']}°C, {weather['wind_speed_kph']}kph, {weather['condition']}")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary analysis
    if results:
        print(f"\n📈 REAL PADDLINGOUT API VALIDATION RESULTS:")
        print("=" * 70)
        
        ml_scores = [r['ml_prediction'] for r in results]
        rule_scores = [r['rule_prediction'] for r in results]
        differences = [r['difference'] for r in results]
        
        print(f"🎯 REAL paddlingOut lakes tested: {len(results)}")
        print(f"�� ML Predictions - Mean: {np.mean(ml_scores):.2f}, Std: {np.std(ml_scores):.3f}")
        print(f"📐 Rule Predictions - Mean: {np.mean(rule_scores):.2f}, Std: {np.std(rule_scores):.3f}")
        print(f"📊 Mean Difference: {np.mean(differences):.3f}")
        print(f"📊 Max Difference: {np.max(differences):.3f}")
        
        # Correlation
        if len(ml_scores) > 1:
            correlation = np.corrcoef(ml_scores, rule_scores)[0,1]
            print(f"🔗 ML vs Rule Correlation: {correlation:.3f}")
        
        # Best and worst predictions
        if len(results) > 1:
            best_match = min(results, key=lambda x: x['difference'])
            worst_match = max(results, key=lambda x: x['difference'])
            
            print(f"\n🏆 Best Match: {best_match['lake']} (diff: {best_match['difference']:.3f})")
            print(f"⚠️  Largest Diff: {worst_match['lake']} (diff: {worst_match['difference']:.3f})")
        
        # Detailed results table
        print(f"\n�� DETAILED RESULTS FOR ALL {len(results)} REAL LAKES:")
        print("-" * 90)
        print(f"{'#':<2} {'Lake Name':<20} {'Location':<25} {'ML':<4} {'Rule':<4} {'Diff':<5} {'Weather'}")
        print("-" * 90)
        
        for i, r in enumerate(results, 1):
            location = f"{r['subtitle'][:24]}"
            weather_summary = f"{r['weather']['temp_c']:.0f}°C, {r['weather']['wind_kph']:.0f}kph"
            print(f"{i:<2} {r['lake'][:19]:<20} {location:<25} {r['ml_prediction']:>4.1f} {r['rule_prediction']:>4.1f} {r['difference']:>5.3f} {weather_summary}")
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        results_df.to_csv('real_paddlingout_api_validation.csv', index=False)
        print(f"\n💾 Detailed results saved to: real_paddlingout_api_validation.csv")
        
        # Model validation status
        mean_diff = np.mean(differences)
        if mean_diff < 0.5:
            print(f"\n✅ REAL API VALIDATION: EXCELLENT (mean diff {mean_diff:.3f} < 0.5)")
        elif mean_diff < 1.0:
            print(f"\n✅ REAL API VALIDATION: GOOD (mean diff {mean_diff:.3f} < 1.0)")
        else:
            print(f"\n⚠️  REAL API VALIDATION: NEEDS REVIEW (mean diff {mean_diff:.3f} > 1.0)")
    
    return results

if __name__ == "__main__":
    try:
        results = test_real_paddlingout_lakes()
        print(f"\n🎉 REAL paddlingOut API validation complete! Tested {len(results)} actual lakes.")
        
    except Exception as e:
        print(f"💥 Testing failed: {e}")
        import traceback
        traceback.print_exc()
