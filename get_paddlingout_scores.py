#!/usr/bin/env python3
"""
Get current paddle scores for YOUR paddlingOut locations ONLY - BEAUTIFUL OUTPUT
"""
import joblib
import numpy as np
import requests
import json
from datetime import datetime

# Load model
model = joblib.load("models/kaayko_paddle_model.pkl")
with open("models/model_metadata.json") as f:
    metadata = json.load(f)

def get_paddlingout_lakes():
    """Get YOUR paddlingOut lakes"""
    response = requests.get("https://us-central1-kaaykostore.cloudfunctions.net/api/paddlingOut")
    lakes = response.json()
    
    return [(lake['title'], lake['subtitle'], lake['location']['latitude'], lake['location']['longitude']) 
            for lake in lakes if 'location' in lake]

def get_weather(lat, lon):
    """Get current weather for coordinates"""
    url = f"https://api.weatherapi.com/v1/current.json?key=a0ede903980f45c4a27183708252308&q={lat},{lon}&aqi=no"
    response = requests.get(url).json()
    data = response['current']
    
    return {
        'temperature_c': data['temp_c'],
        'wind_speed_kph': data['wind_kph'],
        'humidity': data['humidity'],
        'pressure_hpa': data['pressure_mb'],
        'visibility_km': data['vis_km'],
        'cloud_cover': data['cloud'],
        'precip_mm': data['precip_mm'],
        'uv': data['uv'],
        'dew_point_c': data.get('dewpoint_c', data['temp_c'] - 5),
        'feelslike_c': data['feelslike_c'],
        'gust_kph': data['gust_kph'],
        'condition': data['condition']['text']
    }

def get_paddle_score(weather, lake_id=0):
    """Weather → Model → Score"""
    features = {
        'temperature_c': weather['temperature_c'],
        'wind_speed_kph': weather['wind_speed_kph'],
        'humidity': weather['humidity'],
        'pressure_hpa': weather['pressure_hpa'],
        'visibility_km': weather['visibility_km'],
        'cloud_cover': weather['cloud_cover'],
        'precip_mm': weather['precip_mm'],
        'uv': weather['uv'],
        'dew_point_c': weather['dew_point_c'],
        'feelslike_c': weather['feelslike_c'],
        'gust_kph': weather['gust_kph'],
        'temp_comfort': 1 if 15 <= weather['temperature_c'] <= 25 else 0,
        'wind_category': 0 if weather['wind_speed_kph'] <= 15 else 1 if weather['wind_speed_kph'] <= 30 else 2 if weather['wind_speed_kph'] <= 50 else 3,
        'visibility_good': 1 if weather['visibility_km'] > 5 else 0,
        'lake_encoded': lake_id
    }
    
    feature_vector = [features[fname] for fname in metadata['feature_names']]
    X = np.array(feature_vector).reshape(1, -1)
    return float(model.predict(X)[0])

def get_score_emoji(score):
    """Get emoji for score"""
    if score >= 4.5:
        return "🟢"
    elif score >= 3.5:
        return "🟡"
    elif score >= 2.5:
        return "🟠"
    else:
        return "🔴"

def get_recommendation(score):
    """Get recommendation text"""
    if score >= 4.5:
        return "EXCELLENT"
    elif score >= 3.5:
        return "GOOD"
    elif score >= 2.5:
        return "MODERATE"
    else:
        return "CHALLENGING"

def main():
    """Get scores for ALL paddlingOut locations - BEAUTIFUL OUTPUT"""
    print("🏄‍♂️ KAAYKO PADDLE INTELLIGENCE - LIVE SCORES")
    print("=" * 80)
    print(f"📅 Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    print("�� Current conditions for all paddlingOut locations\n")
    
    lakes = get_paddlingout_lakes()
    results = []
    
    print("⏳ Fetching live weather data...")
    print()
    
    for i, (name, location, lat, lon) in enumerate(lakes):
        try:
            weather = get_weather(lat, lon)
            score = get_paddle_score(weather, i)
            
            results.append({
                'name': name,
                'location': location,
                'score': score,
                'weather': weather
            })
            
        except Exception as e:
            results.append({
                'name': name,
                'location': location,
                'score': 0,
                'weather': None,
                'error': str(e)
            })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Display beautiful results
    print("🏆 PADDLE SCORES RANKED (Best to Challenging)")
    print("=" * 80)
    print(f"{'Rank':<4} {'Lake':<22} {'Location':<25} {'Score':<6} {'Conditions'}")
    print("-" * 80)
    
    for rank, result in enumerate(results, 1):
        name = result['name'][:21]
        location = result['location'][:24]
        
        if result['weather']:
            score = result['score']
            emoji = get_score_emoji(score)
            rec = get_recommendation(score)
            temp = result['weather']['temperature_c']
            wind = result['weather']['wind_speed_kph']
            condition = result['weather']['condition'][:15]
            
            conditions = f"{temp:.0f}°C, {wind:.0f}kph, {condition}"
            
            print(f"{rank:<4} {name:<22} {location:<25} {emoji} {score:.1f} {conditions}")
        else:
            print(f"{rank:<4} {name:<22} {location:<25} ❌ ERR  Weather unavailable")
    
    print("\n" + "=" * 80)
    
    # Summary stats
    valid_scores = [r['score'] for r in results if r['weather']]
    if valid_scores:
        excellent = len([s for s in valid_scores if s >= 4.5])
        good = len([s for s in valid_scores if 3.5 <= s < 4.5])
        moderate = len([s for s in valid_scores if 2.5 <= s < 3.5])
        challenging = len([s for s in valid_scores if s < 2.5])
        
        print(f"📊 SUMMARY: {len(valid_scores)} locations analyzed")
        print(f"   �� Excellent (4.5+): {excellent} lakes")
        print(f"   🟡 Good (3.5-4.4):   {good} lakes") 
        print(f"   🟠 Moderate (2.5-3.4): {moderate} lakes")
        print(f"   🔴 Challenging (<2.5): {challenging} lakes")
        
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"   📈 Average Score: {avg_score:.1f}/5.0")
    
    # Top recommendations
    print(f"\n🌟 TOP RECOMMENDATIONS RIGHT NOW:")
    top_3 = [r for r in results if r['weather']][:3]
    for i, result in enumerate(top_3, 1):
        emoji = get_score_emoji(result['score'])
        rec = get_recommendation(result['score'])
        print(f"   {i}. {emoji} {result['name']} - {result['score']:.1f}/5.0 ({rec})")
    
    print(f"\n💡 Powered by Kaayko AI • 14M+ data points • 99.28% accuracy")
    print("=" * 80)

if __name__ == "__main__":
    main()
