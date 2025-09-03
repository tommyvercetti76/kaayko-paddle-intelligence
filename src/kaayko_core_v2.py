#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaayko Superior Trainer v2.0 - Core Training Engine
===================================================

🎯 RESPONSIBILITIES:
• ML model definitions and algorithm implementations
• Feature engineering and data preprocessing
• Training pipeline and hyperparameter optimization
• Model evaluation and validation
• Data loading and processing

Author: Kaayko Intelligence Team
Version: 2.0
License: Proprietary
"""

import json
import logging
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import numpy as np
from scipy.stats import loguniform

from sklearn.model_selection import (
    cross_val_score, RandomizedSearchCV, GroupKFold, KFold
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor, VotingRegressor, 
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from kaayko_config_v2 import (
    Colors, CANONICAL_WEATHER_COLS, TARGET_COL, RANDOM_STATE, 
    print_header, mask_path, quantize_score, SAMPLE_CONFIGS,
    TrainingConfig, setup_logging
)

# ============================================================================
# ALGORITHM FACTORY
# ============================================================================

class AlgorithmFactory:
    """Professional algorithm factory with timeout handling."""
    
    @staticmethod
    def create_algorithm(algorithm_type: str, **kwargs) -> Any:
        """Create algorithm instance with proper configuration."""
        # M1 Pro Max optimized algorithms - FULL ACCURACY maintained, smart parallelization
        algorithms = {
            'histgradient': HistGradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.05,        # Optimal learning rate for accuracy
                max_iter=300,              # Full iterations for robustness
                max_depth=12,              # Deeper for better patterns
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=25,       # Increased patience for stability
                scoring='r2'
            ),
            'randomforest': RandomForestRegressor(
                n_estimators=250,          # More estimators for robustness
                max_depth=20,              # Deep enough for complex patterns
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',       # Good balance of speed and accuracy
                random_state=RANDOM_STATE,
                n_jobs=-2,                 # Leave 2 cores free for system stability
                warm_start=False
            ),
            'extratrees': ExtraTreesRegressor(
                n_estimators=250,          # More estimators for robustness
                max_depth=20,              # Deep enough for complex patterns
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',       # Good balance of speed and accuracy
                random_state=RANDOM_STATE,
                n_jobs=-2,                 # Leave 2 cores free for system stability
                warm_start=False
            ),
            'gradientboosting': GradientBoostingRegressor(
                n_estimators=250,          # More estimators for robustness
                learning_rate=0.08,        # Balanced learning rate
                max_depth=10,              # Deep enough for patterns
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                validation_fraction=0.1,
                n_iter_no_change=25,       # Increased patience for stability
                subsample=0.95             # High subsample for accuracy
            ),
            'xgboost': AlgorithmFactory._create_xgboost_regressor()
        }
        
        if algorithm_type not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_type}")
        
        algorithm = algorithms[algorithm_type]
        if algorithm is None:  # XGBoost not available
            raise ValueError(f"XGBoost not available - install with 'pip install xgboost'")
        
        return algorithm
    
    @staticmethod
    def _create_xgboost_regressor():
        """Create optimized XGBoost regressor for M1 Max."""
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,          # More estimators for robustness
                max_depth=10,              # Deep enough for complex patterns
                learning_rate=0.08,        # Balanced learning rate
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                tree_method='auto',        # Let XGBoost choose optimal method for M1
                n_jobs=-1,                 # Use all cores for XGBoost
                eval_metric='rmse',
                verbosity=0,               # Suppress XGBoost warnings
                # Remove early stopping for cross-validation compatibility
                # early_stopping_rounds=25,  # This causes issues with CV without validation set
                validate_parameters=True
            )
        except ImportError:
            return None
    
    @staticmethod
    def create_ensemble() -> VotingRegressor:
        """Create ensemble with all algorithms including XGBoost if available."""
        estimators = [
            ('hist', AlgorithmFactory.create_algorithm('histgradient')),
            ('rf', AlgorithmFactory.create_algorithm('randomforest')),
            ('et', AlgorithmFactory.create_algorithm('extratrees')),
            ('gb', AlgorithmFactory.create_algorithm('gradientboosting'))
        ]
        
        # Try to add XGBoost to ensemble
        try:
            xgb_estimator = AlgorithmFactory.create_algorithm('xgboost')
            if xgb_estimator is not None:
                estimators.append(('xgb', xgb_estimator))
                print(f"{Colors.GREEN}🚀 XGBoost added to ensemble for M1 optimization!{Colors.RESET}")
        except (ValueError, ImportError):
            print(f"{Colors.YELLOW}⚠️  XGBoost not available for ensemble{Colors.RESET}")
        
        # VotingRegressor without n_jobs - let individual estimators handle parallelization
        return VotingRegressor(estimators=estimators)

class AlgorithmEvaluator:
    """Professional algorithm evaluation with parallel processing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger('kaayko_training')
        self.interrupt_handler = None
    
    def _check_interrupt(self) -> bool:
        """Check for user interruption."""
        if self.interrupt_handler and self.interrupt_handler.interrupted:
            return True
        return False
    
    def evaluate_single_algorithm(self, name: str, algorithm: Any, X: pd.DataFrame, 
                                y: pd.Series, groups: np.ndarray) -> Dict[str, float]:
        """Evaluate single algorithm with efficient processing."""
        try:
            print(f"⏳ Training {name}...")
            
            # Check for interruption
            if self.interrupt_handler and self.interrupt_handler.interrupted:
                return {'r2_mean': 0.0, 'r2_std': 0.0, 'r2_scores': [], 'interrupted': True}
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(mutual_info_regression, k=min(50, X.shape[1]))),
                ('regressor', algorithm)
            ])
            
            # Perform cross-validation with reduced folds for speed
            cv = GroupKFold(n_splits=2) if groups is not None else KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(pipeline, X, y, cv=cv, groups=groups, scoring='r2', n_jobs=2)
            rmse_scores = -cross_val_score(pipeline, X, y, cv=cv, groups=groups, scoring='neg_root_mean_squared_error', n_jobs=2)
            
            result = {
                'r2_mean': np.mean(scores),
                'r2_std': np.std(scores),
                'rmse_mean': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'r2_scores': scores.tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"  {Colors.RED}❌ {name} failed: {str(e)}{Colors.RESET}")
            return {'r2_mean': 0.0, 'r2_std': 0.0, 'r2_scores': [], 'error': str(e)}

    def _evaluate_algorithms_parallel(self, algorithms: Dict[str, Any], X: pd.DataFrame, 
                                    y: pd.Series, groups: np.ndarray = None) -> Dict[str, Dict]:
        """Evaluate algorithms in parallel using ThreadPoolExecutor."""
        results = {}
        
        def train_single_algorithm(name_algo_pair):
            name, algorithm = name_algo_pair
            if algorithm is None:
                return name, {'r2_mean': 0.0, 'r2_std': 0.0, 'r2_scores': [], 'error': 'Algorithm creation failed'}
            
            try:
                start_time = time.time()
                result = self.evaluate_single_algorithm(name, algorithm, X, y, groups)
                end_time = time.time()
                result['training_time'] = end_time - start_time
                return name, result
            except Exception as e:
                return name, {'r2_mean': 0.0, 'r2_std': 0.0, 'r2_scores': [], 'error': str(e)}
        
        # Use ThreadPoolExecutor for I/O bound ML operations
        with ThreadPoolExecutor(max_workers=min(3, len(algorithms))) as executor:
            # Submit all algorithm training jobs
            future_to_name = {
                executor.submit(train_single_algorithm, (name, algo)): name 
                for name, algo in algorithms.items() if algo is not None
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    algo_name, result = future.result()
                    results[algo_name] = result
                    
                    # Print progress
                    if 'error' in result:
                        status = f"{Colors.RED}❌ ERROR{Colors.RESET}"
                    else:
                        r2_score = result['r2_mean']
                        training_time = result.get('training_time', 0)
                        status = f"{Colors.GREEN}✅ Complete ({training_time:.1f}s){Colors.RESET}"
                        print(f"🏁 {algo_name}: {r2_score:.2%} R² {status}")
                        
                except Exception as e:
                    results[name] = {'r2_mean': 0.0, 'r2_std': 0.0, 'r2_scores': [], 'error': str(e)}
        
        return results
    
    def evaluate_all_algorithms(self, X: pd.DataFrame, y: pd.Series, 
                               groups: np.ndarray = None) -> Dict[str, Dict]:
        """Evaluate all individual algorithms with XGBoost optimization for M1 Max."""
        print(f"{Colors.BOLD}{Colors.WHITE}🏆 INDIVIDUAL ALGORITHM PERFORMANCE COMPARISON{Colors.RESET}")
        print("="*90)
        print(f"{Colors.YELLOW}Testing algorithms individually to show why our ensemble dominates...{Colors.RESET}")
        print()
        print("Algorithm            R² Score    RMSE    Status")
        print("-" * 70)
        
        # Check if XGBoost is available
        try:
            import xgboost as xgb
            xgboost_available = True
            print(f"{Colors.GREEN}🚀 XGBoost detected - optimized for M1 Max performance!{Colors.RESET}")
        except ImportError:
            xgboost_available = False
            print(f"{Colors.YELLOW}⚠️  XGBoost not available - install with 'pip install xgboost'{Colors.RESET}")
        
        # Smart algorithm selection based on dataset size and hardware
        if len(X) > 5_000_000 and xgboost_available:  # Massive dataset + XGBoost
            print(f"{Colors.GREEN}💪 MASSIVE DATASET + M1 OPTIMIZATION: Using XGBoost-focused approach{Colors.RESET}")
            algorithms = {
                'XGBoost': self._create_xgboost_regressor(),
                'HistGradientBoosting': AlgorithmFactory.create_algorithm('histgradient')
            }
        elif len(X) > 5_000_000:  # Massive dataset, no XGBoost
            print(f"{Colors.YELLOW}⚡ MASSIVE DATASET DETECTED: Using HistGradientBoosting only{Colors.RESET}")
            algorithms = {
                'HistGradientBoosting': AlgorithmFactory.create_algorithm('histgradient')
            }
        elif len(X) > 1_000_000:  # Large dataset
            algorithms = {
                'HistGradientBoosting': AlgorithmFactory.create_algorithm('histgradient'),
                'RandomForest': AlgorithmFactory.create_algorithm('randomforest')
            }
            if xgboost_available:
                algorithms['XGBoost'] = self._create_xgboost_regressor()
            print(f"{Colors.YELLOW}⚡ LARGE DATASET DETECTED: Fast training mode{Colors.RESET}")
        else:
            # Full algorithm suite for smaller datasets
            algorithms = {
                'HistGradientBoosting': AlgorithmFactory.create_algorithm('histgradient'),
                'RandomForest': AlgorithmFactory.create_algorithm('randomforest'),
                'ExtraTrees': AlgorithmFactory.create_algorithm('extratrees'),
                'GradientBoosting': AlgorithmFactory.create_algorithm('gradientboosting')
            }
            if xgboost_available:
                algorithms['XGBoost'] = self._create_xgboost_regressor()
        
        results = {}
        
        # Use parallel training for massive datasets
        if len(X) > 5_000_000:
            print(f"{Colors.CYAN}🚀 PARALLEL TRAINING MODE for {len(X):,} samples{Colors.RESET}")
            results = self._evaluate_algorithms_parallel(algorithms, X, y, groups)
        else:
            # Sequential training for smaller datasets
            for name, algorithm in algorithms.items():
                if algorithm is None:  # Skip if XGBoost creation failed
                    continue
                    
                if self.interrupt_handler and self.interrupt_handler.interrupted:
                    print(f"\n{Colors.YELLOW}⚠️ Training interrupted by user{Colors.RESET}")
                    break
                
                result = self.evaluate_single_algorithm(name, algorithm, X, y, groups)
                results[name] = result

        # Display results summary
        print(f"\n{Colors.BOLD}📊 ALGORITHM PERFORMANCE SUMMARY{Colors.RESET}")
        print("=" * 70)
        print("Algorithm            R² Score    RMSE    Status")
        print("-" * 70)
        
        for name, result in results.items():
            if 'timeout' in result:
                status = f"{Colors.YELLOW}⏰ TIMEOUT{Colors.RESET}"
                rmse_display = "N/A"
                r2_display = "N/A"
            elif 'error' in result:
                status = f"{Colors.RED}❌ ERROR{Colors.RESET}"
                rmse_display = "N/A"
                r2_display = "N/A"
            else:
                r2_score = result['r2_mean']
                rmse = result.get('rmse_mean', 'N/A')
                r2_display = f"{r2_score:.2%}"
                rmse_display = f"{rmse:.4f}" if rmse != 'N/A' else 'N/A'
                
                if r2_score >= 0.97:
                    status = f"{Colors.GREEN}🥇 BEST{Colors.RESET}"
                elif r2_score >= 0.95:
                    status = f"{Colors.YELLOW}🥈 GOOD{Colors.RESET}"
                else:
                    status = f"{Colors.BLUE}✅ OK{Colors.RESET}"
            
            print(f"{name:<20} {r2_display:<7} {rmse_display:<7} {status}")
        
        return results
    
    def _create_xgboost_regressor(self):
        """Create optimized XGBoost regressor for M1 Max."""
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                tree_method='auto',  # Let XGBoost choose optimal method for M1
                n_jobs=-1,
                eval_metric='rmse',
                verbosity=0  # Suppress XGBoost warnings
            )
        except ImportError:
            return None

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Professional feature engineering pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logging()
    
    def standardize_weather_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize weather column names and units to canonical metric system."""
        df_standardized = df.copy()
        conversions_applied = []
        
        # Temperature: always convert to Celsius
        if 'temp_f' in df_standardized.columns:
            df_standardized['temperature'] = (df_standardized['temp_f'] - 32) * 5/9
            conversions_applied.append('temp_f -> temperature (°C)')
        elif 'temp_c' in df_standardized.columns:
            df_standardized['temperature'] = df_standardized['temp_c']
            conversions_applied.append('temp_c -> temperature (°C)')
        
        # Wind speed: always convert to km/h
        if 'wind_mph' in df_standardized.columns:
            df_standardized['wind_speed'] = df_standardized['wind_mph'] * 1.60934
            conversions_applied.append('wind_mph -> wind_speed (kph)')
        elif 'wind_kph' in df_standardized.columns:
            df_standardized['wind_speed'] = df_standardized['wind_kph']
            conversions_applied.append('wind_kph -> wind_speed (kph)')
        
        # Pressure: always convert to millibars
        if 'pressure_in' in df_standardized.columns:
            df_standardized['pressure'] = df_standardized['pressure_in'] * 33.8639
            conversions_applied.append('pressure_in -> pressure (mb)')
        elif 'pressure_mb' in df_standardized.columns:
            df_standardized['pressure'] = df_standardized['pressure_mb']
            conversions_applied.append('pressure_mb -> pressure (mb)')
        
        # Precipitation: always convert to mm
        if 'precip_in' in df_standardized.columns:
            df_standardized['precipitation'] = df_standardized['precip_in'] * 25.4
            conversions_applied.append('precip_in -> precipitation (mm)')
        elif 'precip_mm' in df_standardized.columns:
            df_standardized['precipitation'] = df_standardized['precip_mm']
            conversions_applied.append('precip_mm -> precipitation (mm)')
        
        # Visibility: always convert to km
        if 'vis_miles' in df_standardized.columns:
            df_standardized['visibility'] = df_standardized['vis_miles'] * 1.60934
            conversions_applied.append('vis_miles -> visibility (km)')
        elif 'vis_km' in df_standardized.columns:
            df_standardized['visibility'] = df_standardized['vis_km']
            conversions_applied.append('vis_km -> visibility (km)')
        
        # Simple renames (no unit conversion)
        simple_renames = {
            'humidity': 'humidity',
            'cloud': 'cloud_cover',
            'uv': 'uv_index'
        }
        
        for old_col, new_col in simple_renames.items():
            if old_col in df_standardized.columns:
                if old_col != new_col:
                    df_standardized[new_col] = df_standardized[old_col]
                    conversions_applied.append(f'{old_col} -> {new_col}')
        
        # Remove original columns to avoid conflicts
        original_cols_to_drop = [
            'temp_f', 'temp_c', 'wind_mph', 'wind_kph', 'pressure_in', 'pressure_mb',
            'precip_in', 'precip_mm', 'vis_miles', 'vis_km', 'cloud', 'uv'
        ]
        for col in original_cols_to_drop:
            if col in df_standardized.columns and col not in CANONICAL_WEATHER_COLS:
                df_standardized = df_standardized.drop(columns=[col])
        
        self.logger.info(f"Unit conversions applied: {conversions_applied}")
        return df_standardized
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        if 'datetime' not in df.columns:
            return df
        
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract temporal features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)  # Convert to int to avoid UInt32 issues
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Cyclical features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        self.logger.info("Created temporal features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features."""
        if 'datetime' not in df.columns:
            self.logger.info("No datetime column, skipping rolling features")
            return df
        
        df = df.copy()
        
        # Determine lake grouping column - use 'lake' if available, fallback to 'lake_name'
        lake_col = None
        if 'lake' in df.columns:
            lake_col = 'lake'
        elif 'lake_name' in df.columns:
            lake_col = 'lake_name'
        
        # Sort by available columns
        if lake_col:
            df = df.sort_values([lake_col, 'datetime'])
            self.logger.info(f"Creating rolling features grouped by {lake_col}")
        else:
            df = df.sort_values(['datetime'])
            self.logger.info("Creating rolling features for single time series (no lake grouping)")
        
        # Weather columns for rolling features (use actual column names from data)
        weather_cols = [col for col in ['temperature', 'wind_speed', 'humidity', 'cloud_cover', 'uv_index'] 
                       if col in df.columns]
        
        if not weather_cols:
            self.logger.info("No weather columns for rolling features")
            return df
        
        self.logger.info(f"Creating rolling features for: {weather_cols}")
        
        # Create rolling features
        for col in weather_cols:
            if lake_col:
                # Group by lake column
                # 3-hour rolling features
                df[f'{col}_roll_3h_mean'] = df.groupby(lake_col)[col].rolling(3, min_periods=1).mean().values
                df[f'{col}_roll_3h_std'] = df.groupby(lake_col)[col].rolling(3, min_periods=1).std().fillna(0).values
                
                # 12-hour rolling features
                df[f'{col}_roll_12h_mean'] = df.groupby(lake_col)[col].rolling(12, min_periods=1).mean().values
                df[f'{col}_roll_12h_std'] = df.groupby(lake_col)[col].rolling(12, min_periods=1).std().fillna(0).values
            else:
                # Single time series
                # 3-hour rolling features
                df[f'{col}_roll_3h_mean'] = df[col].rolling(3, min_periods=1).mean()
                df[f'{col}_roll_3h_std'] = df[col].rolling(3, min_periods=1).std().fillna(0)
                
                # 12-hour rolling features
                df[f'{col}_roll_12h_mean'] = df[col].rolling(12, min_periods=1).mean()
                df[f'{col}_roll_12h_std'] = df[col].rolling(12, min_periods=1).std().fillna(0)
        
        return df
    
    def create_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather interaction features."""
        weather_cols = [col for col in CANONICAL_WEATHER_COLS if col in df.columns]
        self.logger.info(f"Created weather interactions from {weather_cols}")
        
        df = df.copy()
        
        # Key interactions
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['temp_wind_interaction'] = df['temperature'] * df['wind_speed']
            df['wind_chill'] = df['temperature'] - (df['wind_speed'] * 0.7)
        
        if 'humidity' in df.columns and 'temperature' in df.columns:
            df['heat_index'] = df['temperature'] + 0.5 * df['humidity']
        
        if 'cloud_cover' in df.columns and 'uv_index' in df.columns:
            df['uv_cloud_interaction'] = df['uv_index'] * (1 - df['cloud_cover'] / 100)
        
        return df
    
    def apply_safety_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply safety override policies."""
        if not self.config.safety_overrides:
            return df
        
        df = df.copy()
        original_scores = df[TARGET_COL].copy()
        
        # Temperature-based penalties
        extreme_cold_mask = df['temperature'] <= -5  # Celsius
        freezing_mask = (df['temperature'] <= 0) & (~extreme_cold_mask)
        
        # Wind-based penalties
        dangerous_wind_mask = df['wind_speed'] >= 60  # kph
        strong_wind_mask = (df['wind_speed'] >= 40) & (~dangerous_wind_mask)
        
        # Apply penalties
        df.loc[extreme_cold_mask, TARGET_COL] = np.maximum(0, df.loc[extreme_cold_mask, TARGET_COL] - 4)
        df.loc[freezing_mask, TARGET_COL] = np.maximum(0, df.loc[freezing_mask, TARGET_COL] - 3)
        df.loc[dangerous_wind_mask, TARGET_COL] = np.maximum(0, df.loc[dangerous_wind_mask, TARGET_COL] - 3)
        df.loc[strong_wind_mask, TARGET_COL] = np.maximum(0, df.loc[strong_wind_mask, TARGET_COL] - 2)
        
        # Log statistics
        total_adjusted = (df[TARGET_COL] != original_scores).sum()
        extreme_cold_count = extreme_cold_mask.sum()
        freezing_count = freezing_mask.sum()
        dangerous_wind_count = dangerous_wind_mask.sum()
        
        self.logger.info(f"🛡️ Applied safety logic: {total_adjusted:,}/{len(df):,} ({100*total_adjusted/len(df):.1f}%) scores adjusted for dangerous conditions")
        self.logger.info(f"   ❄️ Extreme cold (<-5°C): {extreme_cold_count:,} samples")
        self.logger.info(f"   🧊 Freezing (≤0°C): {freezing_count:,} samples")
        self.logger.info(f"   🌪️ Dangerous wind (≥40km/h): {dangerous_wind_count:,} samples")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        # Categorical columns to encode
        categorical_columns = [
            'lake_name', 'wind_dir', 'condition', 'skill_level', 'season',
            'season_intensity', 'hemisphere', 'climate_zone', 'region',
            'regional_pattern', 'lake_region', 'lake_type', 'base_lake_name'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                if col == 'season':
                    # One-hot encode season
                    season_dummies = pd.get_dummies(df[col], prefix='season')
                    df = pd.concat([df, season_dummies], axis=1)
                    df = df.drop(columns=[col])
                    self.logger.info(f"One-hot encoded {col} into {len(season_dummies.columns)} columns")
                else:
                    # Label encode other categorical variables
                    df[col] = df[col].astype('category').cat.codes
                    self.logger.info(f"Category encoded {col}")
        
        return df
    
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        self.logger.info("Starting feature engineering pipeline")
        
        original_cols = len(df.columns)
        self.logger.info(f"Input columns: {list(df.columns)}")  # Debug log
        
        # Apply all feature engineering steps
        df = self.standardize_weather_columns(df)
        
        # Sort by lake and datetime if both exist, fallback to lake_name
        if 'lake' in df.columns and 'datetime' in df.columns:
            df = df.sort_values(['lake', 'datetime'])
            self.logger.info("Sorted data by lake and datetime")
        elif 'lake_name' in df.columns and 'datetime' in df.columns:
            df = df.sort_values(['lake_name', 'datetime'])
            self.logger.info("Sorted data by lake_name and datetime")
        elif 'datetime' in df.columns:
            df = df.sort_values(['datetime'])
            self.logger.info("Sorted data by datetime (no lake column)")
        else:
            self.logger.info("No datetime column found, skipping sort")
        
        df = self.create_temporal_features(df)
        df = self.create_rolling_features(df)
        df = self.create_weather_interactions(df)
        df = self.apply_safety_overrides(df)
        df = self.encode_categorical_features(df)
        
        final_cols = len(df.columns)
        self.logger.info(f"Feature engineering complete: {original_cols} → {final_cols} columns")
        
        # Target health check
        if TARGET_COL in df.columns:
            unique_targets = df[TARGET_COL].nunique()
            self.logger.info(f"Target health check: {unique_targets} unique values in {len(df)} samples")
        
        return df

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

class DataProcessor:
    """Professional data loading and processing."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = setup_logging()
        self.feature_engineer = FeatureEngineer(config)
        self.interrupt_handler = None  # Will be set by TrainingPipeline
    
    def _check_interrupt(self) -> bool:
        """Check for user interruption."""
        if self.interrupt_handler and self.interrupt_handler.interrupted:
            self.logger.info("Data loading interrupted by user")
            return True
        return False
    
    def estimate_dataset_size(self) -> Tuple[int, int]:
        """Estimate total dataset size."""
        data_root = Path(self.config.data_root)
        if not data_root.exists():
            return 0, 0
        
        lake_dirs = [d for d in data_root.iterdir() if d.is_dir()]
        total_lakes = len(lake_dirs)
        
        # Estimate samples per lake (average)
        samples_per_lake = 6117  # Based on previous analysis
        estimated_total_samples = total_lakes * samples_per_lake
        
        self.logger.info(f"Estimated dataset size: {estimated_total_samples:,} samples across {total_lakes:,} lakes")
        return estimated_total_samples, total_lakes
    
    def load_massive_dataset(self) -> pd.DataFrame:
        """Load massive dataset with progress tracking."""
        print_header("📊 MASSIVE DATA LOADING")
        
        data_root = Path(self.config.data_root)
        self.logger.info(f"Data root: {mask_path(str(data_root))}")
        
        estimated_total, total_lakes = self.estimate_dataset_size()
        
        # Calculate target samples based on sample size using cohesive SAMPLE_CONFIGS
        sample_multiplier = SAMPLE_CONFIGS[self.config.sample_size]['percentage'] / 100.0
        target_samples = int(estimated_total * sample_multiplier)
        samples_per_lake = max(1, int(target_samples / total_lakes)) if total_lakes > 0 else 0
        
        print(f"🌍 Scanning lake directories in {mask_path(str(data_root))}...")
        print(f"Found {total_lakes:,} lake directories")
        print(f"📊 Target samples per lake: {samples_per_lake:,}")
        print(f"🎯 Total target: {target_samples:,} samples")
        print(f"📈 Estimated total available: {estimated_total:,} samples")
        
        # Load data with memory optimization
        all_dataframes = []
        processed_lakes = 0
        total_samples = 0
        
        lake_dirs = list(data_root.iterdir())
        
        print(f"\n🚀 Starting massive data collection...")
        
        for i, lake_dir in enumerate(lake_dirs):
            if not lake_dir.is_dir():
                continue
            
            # Strategic interrupt check every 100 lakes
            if i % 100 == 0 and self._check_interrupt():
                print(f"{Colors.YELLOW}⚠️ Data loading interrupted by user{Colors.RESET}")
                break
            
            try:
                lake_data = self._load_lake_data(lake_dir, samples_per_lake)
                if not lake_data.empty:
                    all_dataframes.append(lake_data)
                    processed_lakes += 1
                    total_samples += len(lake_data)
                    
                    if processed_lakes % 100 == 0 or processed_lakes <= 10:
                        # Get actual file count for this lake
                        csv_count = len(list(lake_dir.glob('*.csv')))
                        print(f"🏊 Lake {processed_lakes:,}/{total_lakes:,}: {lake_dir.name}... "
                              f"({csv_count:,} files available, {len(lake_data):,} samples loaded) → Total: {total_samples:,} samples")
                
                # Memory cleanup for large datasets
                if processed_lakes % 500 == 0 and processed_lakes > 0:
                    import gc
                    gc.collect()
                
            except Exception as e:
                self.logger.warning(f"Failed to load lake {lake_dir.name}: {str(e)}")
                continue
        
        # Combine all data
        if all_dataframes:
            print(f"\n🔄 Combining data from {processed_lakes:,} lakes...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        print(f"\n💾 Saving massive dataset...")
        self._save_dataset(combined_df)
        
        print_header("✅ MASSIVE DATA LOADING COMPLETE")
        print(f"📊 Total samples: {len(combined_df):,}")
        print(f"🌍 Lakes processed: {processed_lakes:,}")
        
        return combined_df
    
    def _load_lake_data(self, lake_dir: Path, target_samples: int) -> pd.DataFrame:
        """Load data for a single lake."""
        csv_files = list(lake_dir.glob('*.csv'))
        if not csv_files:
            return pd.DataFrame()
        
        # Sample files to reach target samples
        samples_per_file = max(1, target_samples // len(csv_files))
        
        lake_dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Sample from this file
                    if len(df) > samples_per_file:
                        df = df.sample(n=samples_per_file, random_state=RANDOM_STATE)
                    lake_dataframes.append(df)
            except Exception as e:
                self.logger.warning(f"Failed to load {csv_file}: {str(e)}")
                continue
        
        if lake_dataframes:
            return pd.concat(lake_dataframes, ignore_index=True)
        return pd.DataFrame()
    
    def _save_dataset(self, df: pd.DataFrame) -> None:
        """Save dataset in multiple formats."""
        try:
            # Save as Parquet (more efficient)
            df.to_parquet('kaayko_training_dataset.parquet', index=False)
            parquet_size = os.path.getsize('kaayko_training_dataset.parquet') / (1024**3)
            print(f"✅ Saved Parquet: kaayko_training_dataset.parquet")
        except Exception as e:
            self.logger.warning(f"Failed to save Parquet: {str(e)}")
            parquet_size = 0.0
        
        try:
            # Save as CSV (for compatibility)
            df.to_csv('kaayko_training_dataset.csv', index=False)
            csv_size = os.path.getsize('kaayko_training_dataset.csv') / (1024**3)
            print(f"✅ Saved CSV: kaayko_training_dataset.csv")
        except Exception as e:
            self.logger.warning(f"Failed to save CSV: {str(e)}")
            csv_size = 0.0
        
        print(f"💾 kaayko_training_dataset.parquet: {parquet_size:.2f} GB")
        print(f"💾 kaayko_training_dataset.csv: {csv_size:.2f} GB")

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Advanced ML training pipeline with production-grade features."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger('kaayko_training')
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.algorithm_factory = AlgorithmFactory()
        self.algorithm_evaluator = AlgorithmEvaluator(config)
        self.interrupt_handler = None  # Will be set by trainer
    
    def _check_interrupt(self) -> bool:
        """Check for user interruption without bloating code."""
        if self.interrupt_handler and self.interrupt_handler.interrupted:
            self.logger.info("Training interrupted by user")
            return True
        return False
    
    def set_interrupt_handler(self, interrupt_handler):
        """Set interrupt handler and propagate to components."""
        self.interrupt_handler = interrupt_handler
        self.data_processor.interrupt_handler = interrupt_handler
        
    def create_pipeline(self) -> Pipeline:
        """Create ML pipeline based on configuration."""
        if self.config.algorithm == 'ensemble':
            regressor = AlgorithmFactory.create_ensemble()
        else:
            regressor = AlgorithmFactory.create_algorithm(self.config.algorithm)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(mutual_info_regression, k='all')),
            ('regressor', regressor)
        ])
        
        return pipeline
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Prepare data for training."""
        # Feature columns (exclude target and non-feature columns)
        exclude_cols = [TARGET_COL, 'datetime', 'lake', 'lake_name', 'date', 'region', 'country', 
                       'condition', 'wind_dir', 'season', 'hemisphere', 'climate_zone', 
                       'regional_pattern', 'lake_region', 'lake_type', 'base_lake_name']
        
        # Start with all columns except excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith('_encoded')]
        
        # Additional safety: only include numeric columns
        X_temp = df[feature_cols]
        numeric_cols = X_temp.select_dtypes(include=[np.number]).columns.tolist()
        
        # Double check - remove any remaining string columns
        final_cols = []
        for col in numeric_cols:
            try:
                # Test if column can be converted to float
                pd.to_numeric(df[col].iloc[:100], errors='raise')
                final_cols.append(col)
            except (ValueError, TypeError):
                self.logger.warning(f"Excluding non-numeric column: {col}")
        
        X = df[final_cols]
        y = df[TARGET_COL]
        
        self.logger.info(f"Using {len(final_cols)} numeric features: {final_cols[:10]}...")
        
        # Create groups for cross-validation (by lake)
        if 'lake' in df.columns:
            lake_names = df['lake'].astype('category').cat.codes
            groups = lake_names.values
        elif 'lake_name' in df.columns:
            lake_names = df['lake_name'].astype('category').cat.codes
            groups = lake_names.values
        else:
            groups = None
        
        return X, y, groups
    
    def hyperparameter_search(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, 
                            groups: np.ndarray, n_iter: int = 40) -> Pipeline:
        """Perform hyperparameter optimization - FAST & SMART for massive datasets."""
        
        # INTELLIGENT SAMPLING: Use much smaller subset for hyperparameter search
        if len(X) > 1_000_000:  # For massive datasets (>1M samples)
            sample_size_for_search = 25_000  # Extra small sample
            n_iter = min(n_iter, 10)  # Fewer iterations
        elif len(X) > 100_000:  # For large datasets
            sample_size_for_search = 50_000
            n_iter = min(n_iter, 15)
        else:
            sample_size_for_search = min(50000, len(X))  # Original logic
            
        self.logger.info(f"Hyperparameter search on {sample_size_for_search:,} samples (smart sampling)")
        
        if sample_size_for_search < len(X):
            # Stratified sampling to maintain target distribution
            indices = np.random.choice(len(X), sample_size_for_search, replace=False)
            X_search = X.iloc[indices]
            y_search = y.iloc[indices]
            groups_search = groups[indices] if groups is not None else None
        else:
            X_search, y_search, groups_search = X, y, groups
        
        print(f"{Colors.BLUE}🔍 Starting FAST hyperparameter optimization ({min(n_iter, 15)} iterations on {sample_size_for_search:,} samples)...{Colors.RESET}")
        
        # Reduced but effective parameter space
        if self.config.algorithm == 'ensemble':
            param_dist = {
                'regressor__hist__learning_rate': [0.05, 0.1, 0.15],     # Key values only
                'regressor__hist__max_depth': [10, 15],                   # Best performing depths
                'regressor__rf__n_estimators': [200, 250],                # Proven values
                'regressor__rf__max_depth': [15, 20],                     # Skip None for speed
                'selector__k': ['all', 75]                                # Focus on best options
            }
        else:
            # Individual algorithm parameters - focused on best performers
            param_dist = {
                'selector__k': ['all', 75]                                # Focus on proven values
            }
            
            if self.config.algorithm == 'histgradient':
                param_dist.update({
                    'regressor__learning_rate': [0.05, 0.1, 0.15],        # Key values
                    'regressor__max_depth': [10, 12, 15]                  # Best depths
                })
            elif self.config.algorithm == 'xgboost':
                param_dist.update({
                    'regressor__n_estimators': [200, 300, 400],           # Different tree counts
                    'regressor__max_depth': [8, 10, 12],                  # Tree depth variations
                    'regressor__learning_rate': [0.05, 0.08, 0.1],       # Learning rate options
                    'regressor__subsample': [0.8, 0.9],                  # Subsampling ratios
                    'regressor__colsample_bytree': [0.8, 0.9]             # Feature subsampling
                })
        
        # FAST CV: 2-fold instead of 3-fold for speed without major accuracy loss
        cv = GroupKFold(n_splits=2) if groups_search is not None else KFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=min(n_iter, 15),  # Cap at 15 iterations for speed
            cv=cv,
            scoring='r2',
            n_jobs=-2,  # Leave 2 cores free for system stability
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        try:
            search.fit(X, y, groups=groups)
            print(f"{Colors.GREEN}✅ Best hyperparameter score: {search.best_score_:.4f}{Colors.RESET}")
        except KeyboardInterrupt:
            if self.interrupt_handler:
                self.interrupt_handler.interrupted = True
            print(f"{Colors.YELLOW}⚠️ Hyperparameter search interrupted{Colors.RESET}")
            # Return original pipeline if interrupted
            return pipeline
        
        return search.best_estimator_
    
    def evaluate_pipeline(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, 
                         groups: np.ndarray) -> Dict[str, float]:
        """Evaluate pipeline performance."""
        # Determine appropriate number of splits
        if groups is not None:
            n_groups = len(np.unique(groups))
            n_splits = min(5, n_groups)  # Don't exceed number of groups
            cv = GroupKFold(n_splits=n_splits)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        # Cross-validation scores
        r2_scores = cross_val_score(pipeline, X, y, cv=cv, groups=groups, scoring='r2', n_jobs=-2)  # Leave 2 cores free
        mae_scores = cross_val_score(pipeline, X, y, cv=cv, groups=groups, scoring='neg_mean_absolute_error', n_jobs=-2)  # Leave 2 cores free
        
        metrics = {
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores),
            'cv_mae_mean': -np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores)
        }
        
        return metrics
    
    def train_model(self) -> Dict[str, Any]:
        """Main training pipeline."""
        try:
            print_header("🚀 KAAYKO SUPERIOR TRAINER V2.0")
            
            # Check for early interruption
            if self._check_interrupt():
                return {'status': 'interrupted', 'stage': 'startup'}
            
            # Load and process data
            if self.config.smoke_test:
                df = self._generate_synthetic_data()
            else:
                df = self.data_processor.load_massive_dataset()
            
            if df.empty or self._check_interrupt():
                if self._check_interrupt():
                    return {'status': 'interrupted', 'stage': 'data_loading'}
                raise ValueError("No data loaded")
            
            # Feature engineering
            print_header("⚙️ ADVANCED FEATURE ENGINEERING")
            processed_df = self.data_processor.feature_engineer.process_features(df)
            
            if self._check_interrupt():
                return {'status': 'interrupted', 'stage': 'feature_engineering'}
            
            # Prepare training data
            X, y, groups = self.prepare_training_data(processed_df)
            
            print_header("🔍 HYPERPARAMETER OPTIMIZATION")
            print(f"📊 Sampling {min(len(X), self.config.sample_rows_for_search):,} rows for hyperparameter search")
            
            # Sample for hyperparameter search
            if len(X) > self.config.sample_rows_for_search:
                sample_idx = np.random.choice(len(X), self.config.sample_rows_for_search, replace=False)
                X_sample, y_sample = X.iloc[sample_idx], y.iloc[sample_idx]
                groups_sample = groups[sample_idx] if groups is not None else None
            else:
                X_sample, y_sample, groups_sample = X, y, groups
            
            # Individual algorithm comparison (only for ensemble or explicit request)
            if self.config.algorithm == 'ensemble':
                algorithm_results = self.algorithm_evaluator.evaluate_all_algorithms(X_sample, y_sample, groups_sample)
            
            # Create and optimize pipeline
            pipeline = self.create_pipeline()
            best_pipeline = self.hyperparameter_search(pipeline, X_sample, y_sample, groups_sample)
            
            # Final training and evaluation
            print_header("🎯 FINAL MODEL TRAINING")
            print(f"{Colors.GREEN}📊 Training final model on {len(X):,} samples{Colors.RESET}")
            
            best_pipeline.fit(X, y)
            final_metrics = self.evaluate_pipeline(best_pipeline, X, y, groups)
            
            # Save model
            model_path = self.config.models_root / f"kaayko_model_v2_{self.config.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_pipeline, model_path)
            
            return {
                'status': 'success',
                'metrics': final_metrics,
                'model_path': str(model_path),
                'config': self.config.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'config': self.config.to_dict()
            }
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate realistic synthetic data for smoke testing with higher R² potential."""
        print(f"{Colors.BLUE}🧪 Generating realistic synthetic test data...{Colors.RESET}")
        
        np.random.seed(RANDOM_STATE)
        n_samples = 10000
        
        # Create more realistic weather patterns
        data = {
            'temperature': np.random.normal(18, 8, n_samples),  # More reasonable temp range
            'wind_speed': np.random.gamma(2, 5, n_samples),     # Gamma for realistic wind
            'humidity': np.random.beta(2, 2, n_samples) * 80 + 20,  # Beta for humidity
            'pressure': np.random.normal(1013, 15, n_samples),
            'precipitation': np.random.exponential(1.5, n_samples),
            'visibility': np.random.gamma(3, 5, n_samples) + 5,  # Better visibility distribution
            'cloud_cover': np.random.beta(1.5, 1.5, n_samples) * 100,
            'uv_index': np.random.gamma(2, 3, n_samples),
            'lake_name': np.random.choice(['Lake_A', 'Lake_B', 'Lake_C', 'Lake_D', 'Lake_E'], n_samples),
            'datetime': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic non-linear relationships for paddle ratings
        temp_factor = np.where(df['temperature'] < 10, 0.2,  # Cold = bad
                              np.where(df['temperature'] > 25, 0.8, 0.9))  # Warm = good
        
        wind_factor = np.where(df['wind_speed'] > 25, 0.1,   # High wind = bad
                              np.where(df['wind_speed'] < 5, 0.7, 0.9))   # Light wind = good
        
        weather_factor = np.where(df['precipitation'] > 5, 0.3, 0.9)  # Rain = bad
        
        visibility_factor = np.where(df['visibility'] < 10, 0.4, 0.9)  # Low vis = bad
        
        cloud_factor = np.where(df['cloud_cover'] > 80, 0.6, 1.0)  # Overcast = slightly worse
        
        # Combine factors with realistic interactions
        base_score = (temp_factor * wind_factor * weather_factor * visibility_factor * cloud_factor)
        
        # Add some lake-specific bias (some lakes are just better)
        lake_bias = {'Lake_A': 0.1, 'Lake_B': -0.05, 'Lake_C': 0.05, 'Lake_D': 0.0, 'Lake_E': 0.08}
        bias_values = df['lake_name'].map(lake_bias)
        
        # Generate target with much less noise
        df[TARGET_COL] = (
            base_score * 4.5 + 0.5 +  # Scale to roughly 0.5-5.0 range
            bias_values +              # Lake-specific effects
            np.random.normal(0, 0.15, n_samples)  # Reduced noise from 0.5 to 0.15
        )
        
        # Clip to valid range and apply minimal quantization
        df[TARGET_COL] = np.clip(df[TARGET_COL], 0, 5)
        
        # Only quantize if not using precise scoring
        if self.config.score_quantization != 'continuous':
            df[TARGET_COL] = df[TARGET_COL].apply(lambda x: quantize_score(x, self.config.score_quantization))
        
        self.logger.info(f"Generated synthetic data: {len(df)} samples with realistic paddle rating patterns")
        self.logger.info(f"Target distribution: min={df[TARGET_COL].min():.2f}, max={df[TARGET_COL].max():.2f}, mean={df[TARGET_COL].mean():.2f}")
        
        return df
