#!/usr/bin/env python3
"""
M1 Max Optimized Training Pipeline for Kaayko Paddle Intelligence
FIXED: Only uses available columns, trains on ALL data (no splits)
"""

import argparse
import multiprocessing
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

# M1 Max Optimization Settings
M1_MAX_CORES = multiprocessing.cpu_count()
OPTIMAL_THREADS = min(M1_MAX_CORES, 8)
print(f"🚀 M1 Max Training Setup - Using {OPTIMAL_THREADS}/{M1_MAX_CORES} cores")

def load_lake_data(data_path: Path, max_lakes: int = None):
    """Load and process lake weather data from directory structure"""
    print(f"📂 Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    lake_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if max_lakes:
        lake_dirs = lake_dirs[:max_lakes]
    
    print(f"🏊 Processing {len(lake_dirs)} lakes...")
    
    all_data = []
    for i, lake_dir in enumerate(lake_dirs, 1):
        try:
            csv_files = list(lake_dir.glob("*.csv"))
            if not csv_files:
                print(f"⚠️  No CSV files in {lake_dir.name}")
                continue
                
            lake_data = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df['lake'] = lake_dir.name
                lake_data.append(df)
            
            if lake_data:
                lake_df = pd.concat(lake_data, ignore_index=True)
                all_data.append(lake_df)
                print(f"✅ {i:3d}/{len(lake_dirs)} - {lake_dir.name}: {len(lake_df):,} records")
            
        except Exception as e:
            print(f"❌ Error loading {lake_dir.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid lake data found!")
    
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"🎯 Total dataset: {len(final_df):,} records from {len(all_data)} lakes")
    
    return final_df

def engineer_features(df):
    """Engineer features - ONLY USING AVAILABLE COLUMNS"""
    print("🔧 Engineering features...")
    
    df = df.copy()
    
    # Check what columns we actually have
    print(f"📋 Available columns: {list(df.columns)}")
    
    weather_cols = {
        'temperature_c': ['temperature_c', 'temp_c', 'temperature', 'temp'],
        'wind_speed_kph': ['wind_speed_kph', 'wind_kph', 'windspeed', 'wind_speed'],
        'humidity': ['humidity', 'humidity_percent'],
        'pressure_hpa': ['pressure_hpa', 'pressure_mb', 'pressure'],
        'visibility_km': ['visibility_km', 'visibility', 'vis_km'],
        'cloud_cover': ['cloud_cover', 'cloud', 'cloudiness'],
        'precip_mm': ['precip_mm', 'precipitation', 'rain_mm'],
        'uv': ['uv', 'uv_index'],
        'dew_point_c': ['dew_point_c', 'dewpoint_c', 'dew_point'],
        'feelslike_c': ['feelslike_c', 'feels_like_c', 'apparent_temp'],
        'gust_kph': ['gust_kph', 'wind_gust_kph', 'gust']
    }
    
    # ONLY USE COLUMNS THAT EXIST
    feature_cols = []
    for standard_name, possible_names in weather_cols.items():
        found_col = None
        for possible_name in possible_names:
            if possible_name in df.columns:
                found_col = possible_name
                break
        
        if found_col:
            if found_col != standard_name:
                df[standard_name] = df[found_col]
            feature_cols.append(standard_name)
            print(f"✅ Found {standard_name}: using '{found_col}'")
        else:
            print(f"❌ Column {standard_name} not available - SKIPPING")
    
    # Engineer paddle safety score if not present
    if 'paddle_score' not in df.columns:
        print("🎯 Creating paddle safety scores...")
        df['paddle_score'] = calculate_paddle_score(df)
    
    # Engineering derived features from AVAILABLE data only
    derived_features = []
    
    if 'temperature_c' in feature_cols:
        df['temp_comfort'] = np.where(df['temperature_c'].between(15, 25), 1, 0)
        derived_features.append('temp_comfort')
    
    if 'wind_speed_kph' in feature_cols:
        df['wind_category'] = pd.cut(df['wind_speed_kph'], 
                                    bins=[0, 15, 30, 50, 100], 
                                    labels=[0, 1, 2, 3]).astype(float)
        derived_features.append('wind_category')
    
    if 'visibility_km' in feature_cols:
        df['visibility_good'] = np.where(df['visibility_km'] > 5, 1, 0)
        derived_features.append('visibility_good')
    
    # Lake encoding
    if 'lake' in df.columns:
        df['lake_encoded'] = pd.Categorical(df['lake']).codes
        derived_features.append('lake_encoded')
    
    feature_cols.extend(derived_features)
    
    print(f"✅ Feature engineering complete - {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {col}")
    
    return df, feature_cols

def calculate_paddle_score(df):
    """Calculate paddle safety score based on available weather conditions"""
    score = np.full(len(df), 3.0)  # Start with neutral score
    
    # Temperature penalties (if available)
    if 'temperature_c' in df.columns:
        score += np.where(df['temperature_c'].between(18, 26), 1.0, 0)
        score -= np.where(df['temperature_c'] < 5, 2.0, 0)
        score -= np.where(df['temperature_c'] > 35, 1.5, 0)
    
    # Wind penalties (if available)
    if 'wind_speed_kph' in df.columns or 'wind_kph' in df.columns:
        wind_col = 'wind_speed_kph' if 'wind_speed_kph' in df.columns else 'wind_kph'
        score -= np.where(df[wind_col] > 30, 2.0, 0)
        score -= np.where(df[wind_col] > 50, 1.5, 0)
        score -= np.where(df[wind_col] > 20, 0.5, 0)
    
    # Precipitation penalties (if available)
    if 'precip_mm' in df.columns:
        score -= np.where(df['precip_mm'] > 10, 1.5, 0)
        score -= np.where(df['precip_mm'] > 5, 0.5, 0)
    
    # Ensure score stays in valid range
    score = np.clip(score, 1.0, 5.0)
    
    return score

def train_m1_optimized_model(X, y, feature_names):
    """Train model optimized for M1 Max - USE ALL DATA"""
    print("🤖 Training M1 Max optimized models on ALL DATA...")
    
    print(f"�� Training set: {len(X):,} samples (100% of data)")
    print(f"🎯 Features: {len(feature_names)}")
    
    # M1 Max optimized models
    models = {
        'HistGradientBoosting': HistGradientBoostingRegressor(
            max_iter=100,  # Reduced for speed
            learning_rate=0.1,
            max_depth=10,  # Reduced for speed
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=50,  # Reduced for speed with massive data
            max_depth=15,
            n_jobs=OPTIMAL_THREADS,
            random_state=42
        )
    }
    
    best_model = None
    best_score = -np.inf
    best_name = None
    
    for name, model in models.items():
        print(f"\n🚀 Training {name} on {len(X):,} samples...")
        start_time = time.time()
        
        # Cross-validation on subset for speed
        print("📊 Running cross-validation on sample...")
        sample_size = min(100000, len(X))  # CV on 100k sample for speed
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        cv_scores = cross_val_score(model, X_sample, y_sample, 
                                   cv=3, scoring='r2', n_jobs=OPTIMAL_THREADS)
        
        print(f"📈 CV R² (on sample): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train on ALL data
        print("�� Training on full dataset...")
        model.fit(X, y)
        
        # Quick evaluation on full data
        train_pred = model.predict(X)
        train_mae = mean_absolute_error(y, train_pred)
        train_r2 = r2_score(y, train_pred)
        
        training_time = time.time() - start_time
        
        print(f"⏱️  Training time: {training_time:.2f}s ({training_time/60:.1f} min)")
        print(f"📊 Full Data MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        
        if train_r2 > best_score:
            best_model = model
            best_score = train_r2
            best_name = name
    
    print(f"\n🏆 Best model: {best_name} (R² = {best_score:.4f})")
    return best_model, best_name, best_score

def save_model(model, model_name, score, feature_names, output_dir):
    """Save trained model with metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "kaayko_paddle_model.pkl"
    joblib.dump(model, model_path)
    
    metadata = {
        'model_name': model_name,
        'score': float(score),
        'feature_names': feature_names,
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'cpu_cores': M1_MAX_CORES,
            'used_threads': OPTIMAL_THREADS,
            'platform': 'M1 Max'
        },
        'data_info': {
            'training_approach': 'Full dataset training (no splits)',
            'feature_engineering': 'Only available columns used'
        }
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model saved to: {model_path}")
    print(f"📄 Metadata saved to: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='M1 Max Optimized Kaayko Training - FIXED')
    parser.add_argument('--data-path', type=str, 
                       default='/Users/Rohan/data_lake_monthly',
                       help='Path to lake data directory')
    parser.add_argument('--max-lakes', type=int, 
                       help='Maximum number of lakes to process')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    print("�� KAAYKO PADDLE INTELLIGENCE - M1 MAX TRAINING (FIXED) 🚀")
    print("=" * 70)
    print(f"💻 System: M1 Max ({M1_MAX_CORES} cores, using {OPTIMAL_THREADS})")
    print(f"📂 Data path: {args.data_path}")
    print(f"🎯 Strategy: ALL data training (no splits)")
    print(f"✅ Fix: Only available columns used")
    if args.max_lakes:
        print(f"🔢 Processing max {args.max_lakes} lakes")
    print("=" * 70)
    
    try:
        # Load data
        df = load_lake_data(Path(args.data_path), args.max_lakes)
        
        # Engineer features
        df, feature_cols = engineer_features(df)
        
        # Prepare training data
        X = df[feature_cols].fillna(0)
        y = df['paddle_score']
        
        print(f"\n📊 Final dataset: {len(X):,} samples, {len(feature_cols)} features")
        print(f"🎯 Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Train model on ALL data
        model, model_name, score = train_m1_optimized_model(X, y, feature_cols)
        
        # Save model
        save_model(model, model_name, score, feature_cols, args.output_dir)
        
        print(f"\n🎉 Training complete! R² score: {score:.4f}")
        
    except Exception as e:
        print(f"💥 Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
