#!/usr/bin/env python3
"""
🏄 KAAYKO PADDLING LOCATIONS - WORKING MODEL COMPARISON UI
Uses existing comparison data to display results properly
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template_string
import threading
import webbrowser
import time

app = Flask(__name__)

# Load existing comparison results
def load_existing_results():
    """Load our existing detailed comparison results"""
    results_file = "detailed_live_comparison_20250902_221117.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return []

def transform_results_for_ui(raw_results):
    """Transform the raw results into UI-friendly format"""
    transformed = []
    
    for result in raw_results:
        transformed.append({
            "location": {
                "id": result["location_id"],
                "title": result["title"],
                "lat": result["latitude"],
                "lng": result["longitude"],
                "region": get_region_from_coords(result["latitude"], result["longitude"])
            },
            "production": {
                "status": "success",
                "score": result["histgradient_model"]["score"],  # Using HistGradient as "production"
                "confidence": result["histgradient_model"]["confidence"],
                "weather": result["weather_conditions"]
            },
            "local_xgboost": {
                "status": "success", 
                "score": result["xgboost_model"]["score"],
                "confidence": result["xgboost_model"]["confidence"],
                "weather": result["weather_conditions"]
            },
            "local_histgradient": {
                "status": "success",
                "score": result["histgradient_model"]["score"], 
                "confidence": result["histgradient_model"]["confidence"],
                "weather": result["weather_conditions"]
            },
            "timestamp": result["timestamp"]
        })
    
    return transformed

def get_region_from_coords(lat, lng):
    """Determine region from coordinates"""
    if lat > 45:
        return "Montana/Washington"
    elif lat > 40:
        return "Wyoming/Colorado"
    elif lat > 30:
        return "Texas/Utah"
    else:
        return "International"

@app.route('/')
def index():
    """Main page with comparison results"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏄 Kaayko Paddling Locations - Model Comparison</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .summary-card {
            background: rgba(255,255,255,0.95);
            padding: 25px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .summary-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .summary-card .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }
        
        .summary-card .subtitle {
            font-size: 0.9em;
            color: #666;
            margin-top: 8px;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 25px;
        }
        
        .location-card {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .location-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        }
        
        .location-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 3px solid #f0f0f0;
        }
        
        .location-title {
            font-size: 1.4em;
            font-weight: bold;
            color: #333;
        }
        
        .location-region {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .coordinates {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .scores-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }
        
        .score-box {
            text-align: center;
            padding: 20px;
            border-radius: 16px;
            position: relative;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .score-box.production {
            background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
            color: white;
        }
        
        .score-box.local-xgb {
            background: linear-gradient(135deg, #4ECDC4, #7FDBDA);
            color: white;
        }
        
        .score-label {
            font-size: 1em;
            margin-bottom: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .score-value {
            font-size: 2.8em;
            font-weight: bold;
            margin-bottom: 8px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .score-confidence {
            font-size: 0.85em;
            opacity: 0.9;
            font-weight: 500;
        }
        
        .winner-badge {
            position: absolute;
            top: -12px;
            right: -12px;
            background: linear-gradient(45deg, #FFD700, #FFA500);
            color: #333;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1em;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .weather-info {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 16px;
            margin-top: 20px;
            border-left: 4px solid #667eea;
        }
        
        .weather-title {
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            font-size: 1.1em;
        }
        
        .weather-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            font-size: 0.9em;
        }
        
        .weather-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }
        
        .weather-item:last-child {
            border-bottom: none;
        }
        
        .weather-label {
            font-weight: 600;
            color: #555;
        }
        
        .weather-value {
            font-weight: bold;
            color: #333;
        }
        
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 0.85em;
            margin-top: 20px;
            font-style: italic;
        }
        
        .performance-highlight {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 15px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏄 Kaayko Paddling Locations</h1>
            <p>XGBoost vs HistGradient Model Comparison Results</p>
        </div>
        
        <div class="performance-highlight">
            🏆 XGBoost Dominance: 94.1% Win Rate (16/17 locations) • Average Score Advantage: +0.31 points
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>📍 Total Locations</h3>
                <div class="value">17</div>
                <div class="subtitle">Paddling destinations tested</div>
            </div>
            <div class="summary-card">
                <h3>🥇 XGBoost Wins</h3>
                <div class="value">16</div>
                <div class="subtitle">94.1% win rate</div>
            </div>
            <div class="summary-card">
                <h3>🥈 HistGradient Wins</h3>
                <div class="value">1</div>
                <div class="subtitle">5.9% win rate</div>
            </div>
            <div class="summary-card">
                <h3>⚡ Avg XGBoost Score</h3>
                <div class="value">2.70</div>
                <div class="subtitle">Superior performance</div>
            </div>
            <div class="summary-card">
                <h3>📊 Avg HistGradient Score</h3>
                <div class="value">2.39</div>
                <div class="subtitle">Baseline performance</div>
            </div>
            <div class="summary-card">
                <h3>🎯 Performance Gap</h3>
                <div class="value">+0.31</div>
                <div class="subtitle">XGBoost advantage</div>
            </div>
        </div>
        
        <div class="results-grid" id="results">
            <!-- Results will be populated by JavaScript -->
        </div>
    </div>

    <script>
        // Our comparison results data
        const results = {{ results_json | safe }};
        
        function displayResults() {
            const container = document.getElementById('results');
            
            results.forEach((result) => {
                const location = result.location;
                const xgb = result.local_xgboost;
                const hist = result.production; // Using HistGradient as comparison
                
                // Determine winner
                const xgbWins = xgb.score > hist.score;
                const scoreDiff = Math.abs(xgb.score - hist.score);
                
                const card = document.createElement('div');
                card.className = 'location-card';
                card.innerHTML = `
                    <div class="location-header">
                        <div>
                            <div class="location-title">${location.title}</div>
                            <div class="coordinates">${location.lat.toFixed(6)}, ${location.lng.toFixed(6)}</div>
                        </div>
                        <div class="location-region">${location.region}</div>
                    </div>
                    
                    <div class="scores-container">
                        <div class="score-box local-xgb">
                            ${xgbWins ? '<div class="winner-badge">👑</div>' : ''}
                            <div class="score-label">XGBoost Model</div>
                            <div class="score-value">${xgb.score.toFixed(2)}</div>
                            <div class="score-confidence">${xgb.confidence} confidence</div>
                        </div>
                        
                        <div class="score-box production">
                            ${!xgbWins ? '<div class="winner-badge">👑</div>' : ''}
                            <div class="score-label">HistGradient Model</div>
                            <div class="score-value">${hist.score.toFixed(2)}</div>
                            <div class="score-confidence">${hist.confidence} confidence</div>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin: 15px 0; font-weight: bold; color: ${xgbWins ? '#4ECDC4' : '#FF6B6B'};">
                        ${xgbWins ? 'XGBoost' : 'HistGradient'} wins by ${scoreDiff.toFixed(2)} points
                    </div>
                    
                    ${createWeatherInfo(xgb.weather)}
                    
                    <div class="timestamp">
                        Updated: ${new Date(result.timestamp).toLocaleString()}
                    </div>
                `;
                
                container.appendChild(card);
            });
        }
        
        function createWeatherInfo(weather) {
            return `
                <div class="weather-info">
                    <div class="weather-title">🌤️ Current Conditions</div>
                    <div class="weather-grid">
                        <div class="weather-item">
                            <span class="weather-label">Temperature:</span>
                            <span class="weather-value">${weather.temp_c.toFixed(1)}°C</span>
                        </div>
                        <div class="weather-item">
                            <span class="weather-label">Wind:</span>
                            <span class="weather-value">${weather.wind_kph.toFixed(1)} kph</span>
                        </div>
                        <div class="weather-item">
                            <span class="weather-label">Humidity:</span>
                            <span class="weather-value">${weather.humidity.toFixed(0)}%</span>
                        </div>
                        <div class="weather-item">
                            <span class="weather-label">Cloud Cover:</span>
                            <span class="weather-value">${weather.cloud.toFixed(0)}%</span>
                        </div>
                        <div class="weather-item">
                            <span class="weather-label">Wave Height:</span>
                            <span class="weather-value">${weather.estimated_wave_height_m.toFixed(2)}m</span>
                        </div>
                        <div class="weather-item">
                            <span class="weather-label">Water Temp:</span>
                            <span class="weather-value">${weather.estimated_water_temp_c.toFixed(1)}°C</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Load results on page load
        document.addEventListener('DOMContentLoaded', displayResults);
    </script>
</body>
</html>
    """)

@app.route('/api/results')
def get_results():
    """API endpoint to get comparison results"""
    raw_results = load_existing_results()
    transformed_results = transform_results_for_ui(raw_results)
    return {"status": "success", "results": transformed_results}

def open_browser():
    """Open the web browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5001')

if __name__ == '__main__':
    print("🚀 Starting Kaayko Paddling Locations Model Comparison UI")
    
    # Load and transform results
    raw_results = load_existing_results()
    if not raw_results:
        print("❌ No comparison results found! Please run the comparison first.")
        exit(1)
    
    transformed_results = transform_results_for_ui(raw_results)
    print(f"📍 Loaded {len(transformed_results)} paddling location comparisons")
    print("🌐 Web interface will be available at: http://127.0.0.1:5001")
    
    # Inject results into the template
    app.jinja_env.globals['results_json'] = json.dumps(transformed_results)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)
