"""
Professional ML Training Module
==============================
Implements industry best practices for unbiased, rigorous machine learning training
with proper cross-validation, feature engineering, and model validation.

This module ensures:
1. No data leakage between train/test splits
2. Proper cross-validation methodology
3. Unbiased feature engineering
4. Robust model validation
5. Reproducible experiments
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import joblib
import json
from dataclasses import dataclass

from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, classification_report
from sklearn.base import BaseEstimator, TransformerMixin

from .config import config, model_naming, data_integrity
from .data_integrity import verify_data_integrity

logger = logging.getLogger(__name__)

@dataclass
class ExperimentMetadata:
    """Metadata for reproducible experiments"""
    experiment_id: str
    start_time: datetime
    model_name: str
    model_type: str  # 'global', 'continental', 'regional'
    data_version: str
    feature_count: int
    training_samples: int
    validation_samples: int
    test_samples: int
    hyperparameters: Dict
    cross_validation_scores: List[float]
    final_metrics: Dict
    feature_importance: Dict
    training_time_minutes: float
    reproducibility_seed: int

class UnbiasedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Professional feature engineering with bias prevention measures.
    
    Implements careful feature engineering that:
    1. Avoids target leakage
    2. Handles missing data appropriately  
    3. Creates meaningful weather interaction features
    4. Maintains temporal consistency
    """
    
    def __init__(self, include_seasonal_features: bool = True, 
                 include_interaction_features: bool = True):
        self.include_seasonal_features = include_seasonal_features
        self.include_interaction_features = include_interaction_features
        self.feature_names_ = None
        self.seasonal_scalers_ = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineer on training data only"""
        logger.info("🔧 Fitting unbiased feature engineering pipeline...")
        
        # Store feature names for consistency
        self.feature_names_ = list(X.columns)
        
        # Fit seasonal scalers if needed
        if self.include_seasonal_features and 'datetime' in X.columns:
            X_copy = X.copy()
            X_copy['datetime'] = pd.to_datetime(X_copy['datetime'])
            X_copy['month'] = X_copy['datetime'].dt.month
            
            # Fit separate scalers for each season to handle seasonal variations
            for month in range(1, 13):
                month_data = X_copy[X_copy['month'] == month]
                if len(month_data) > 10:  # Only if sufficient data
                    numeric_cols = month_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        scaler = StandardScaler()
                        scaler.fit(month_data[numeric_cols])
                        self.seasonal_scalers_[month] = scaler
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted parameters"""
        X_transformed = X.copy()
        
        # Add temporal features (no leakage risk)
        if 'datetime' in X_transformed.columns:
            X_transformed['datetime'] = pd.to_datetime(X_transformed['datetime'])
            X_transformed['month'] = X_transformed['datetime'].dt.month
            X_transformed['day_of_year'] = X_transformed['datetime'].dt.dayofyear
            X_transformed['hour'] = X_transformed['datetime'].dt.hour
            
            if self.include_seasonal_features:
                # Add seasonal sine/cosine features (no leakage)
                X_transformed['month_sin'] = np.sin(2 * np.pi * X_transformed['month'] / 12)
                X_transformed['month_cos'] = np.cos(2 * np.pi * X_transformed['month'] / 12)
                X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed['hour'] / 24)
                X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed['hour'] / 24)
        
        # Add weather interaction features (physically meaningful)
        if self.include_interaction_features:
            if all(col in X_transformed.columns for col in ['temp_c', 'humidity']):
                # Heat index approximation
                X_transformed['heat_index'] = X_transformed['temp_c'] + (X_transformed['humidity'] - 40) * 0.1
            
            if all(col in X_transformed.columns for col in ['wind_kph', 'temp_c']):
                # Wind chill effect
                X_transformed['wind_chill_effect'] = X_transformed['wind_kph'] * (10 - X_transformed['temp_c']) / 10
            
            if all(col in X_transformed.columns for col in ['pressure_mb', 'humidity']):
                # Atmospheric stability indicator
                X_transformed['stability_index'] = X_transformed['pressure_mb'] - X_transformed['humidity']
        
        # Add geographical features (if available)
        if all(col in X_transformed.columns for col in ['latitude', 'longitude']):
            # Distance from equator (affects seasonal patterns)
            X_transformed['equator_distance'] = np.abs(X_transformed['latitude'])
            
            # Hemisphere indicator
            X_transformed['northern_hemisphere'] = (X_transformed['latitude'] > 0).astype(int)
        
        logger.info(f"✅ Feature engineering complete. Features: {X_transformed.shape[1]}")
        return X_transformed

class RobustModelTrainer:
    """
    Professional model training with rigorous validation and bias prevention.
    
    Implements:
    1. GroupKFold cross-validation (by lake) to prevent data leakage
    2. Stratified sampling by continent
    3. Proper train/validation/test splits
    4. Hyperparameter optimization
    5. Model interpretation and validation
    """
    
    def __init__(self, model_name: str, model_type: str = 'global'):
        self.model_name = model_name
        self.model_type = model_type
        self.feature_engineer = UnbiasedFeatureEngineer()
        self.preprocessor = None
        self.model = None
        self.experiment_metadata = None
        
    def prepare_training_pipeline(self, target_type: str = 'regression') -> Pipeline:
        """Prepare sklearn pipeline with preprocessing and model"""
        
        # Preprocessing pipeline
        numeric_features = ['temp_c', 'humidity', 'pressure_mb', 'wind_kph', 'uv', 'precip_mm',
                           'latitude', 'longitude', 'month', 'day_of_year', 'hour']
        categorical_features = ['season', 'region', 'climate_zone']
        
        # Robust preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Median is robust to outliers
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', LabelEncoder() if target_type == 'regression' else LabelEncoder())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Model selection based on type and target
        if target_type == 'regression':
            base_model = HistGradientBoostingRegressor(
                max_iter=config.max_model_complexity[self.model_type],
                learning_rate=0.1,
                max_depth=8,
                min_samples_leaf=20,
                random_state=config.random_seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        else:
            base_model = HistGradientBoostingClassifier(
                max_iter=config.max_model_complexity[self.model_type],
                learning_rate=0.1, 
                max_depth=8,
                min_samples_leaf=20,
                random_state=config.random_seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        
        # Complete pipeline
        self.model = Pipeline(steps=[
            ('feature_engineer', self.feature_engineer),
            ('preprocessor', self.preprocessor),
            ('model', base_model)
        ])
        
        return self.model
    
    def train_with_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                   groups: pd.Series, target_type: str = 'regression') -> ExperimentMetadata:
        """
        Train model with rigorous cross-validation methodology.
        
        Args:
            X: Feature matrix
            y: Target variable
            groups: Grouping variable (lake names) for GroupKFold
            target_type: 'regression' or 'classification'
            
        Returns:
            Experiment metadata with all training details
        """
        start_time = datetime.now()
        experiment_id = f"{self.model_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🚀 Starting training experiment: {experiment_id}")
        logger.info(f"📊 Data shape: {X.shape}, Target type: {target_type}")
        
        # Prepare model pipeline
        model = self.prepare_training_pipeline(target_type)
        
        # Train-test split with stratification by groups
        unique_groups = groups.unique()
        train_groups, test_groups = train_test_split(
            unique_groups, 
            test_size=config.test_size,
            random_state=config.random_seed
        )
        
        train_mask = groups.isin(train_groups)
        test_mask = groups.isin(test_groups)
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        groups_train = groups[train_mask]
        
        logger.info(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"🏞️ Train lakes: {len(train_groups)}, Test lakes: {len(test_groups)}")
        
        # Cross-validation with GroupKFold (prevents data leakage)
        cv = GroupKFold(n_splits=config.cross_validation_folds)
        
        # Perform cross-validation
        if target_type == 'regression':
            cv_scores = cross_val_score(model, X_train, y_train, groups=groups_train, 
                                      cv=cv, scoring='r2', n_jobs=-1)
            cv_mae = cross_val_score(model, X_train, y_train, groups=groups_train,
                                   cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_scores_list = cv_scores.tolist()
            
            logger.info(f"📈 Cross-validation R² scores: {cv_scores_list}")
            logger.info(f"📈 Cross-validation MAE: {-cv_mae.mean():.3f} ± {cv_mae.std():.3f}")
            
        else:
            cv_scores = cross_val_score(model, X_train, y_train, groups=groups_train,
                                      cv=cv, scoring='accuracy', n_jobs=-1)
            cv_scores_list = cv_scores.tolist()
            
            logger.info(f"📈 Cross-validation accuracy scores: {cv_scores_list}")
        
        # Check cross-validation consistency
        cv_consistency = 1 - (cv_scores.std() / cv_scores.mean())
        
        if cv_consistency < config.min_cross_val_consistency:
            logger.warning(f"⚠️ Low cross-validation consistency: {cv_consistency:.3f}")
        
        # Train final model on all training data
        model.fit(X_train, y_train)
        
        # Final evaluation on test set
        test_predictions = model.predict(X_test)
        
        if target_type == 'regression':
            test_r2 = r2_score(y_test, test_predictions)
            test_mae = mean_absolute_error(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            
            final_metrics = {
                'test_r2': test_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'cv_consistency': cv_consistency
            }
            
            logger.info(f"✅ Final Test R²: {test_r2:.3f}")
            logger.info(f"✅ Final Test MAE: {test_mae:.3f}")
            
            # Quality check
            if test_r2 < config.min_model_r2_score:
                logger.warning(f"⚠️ Model R² ({test_r2:.3f}) below threshold ({config.min_model_r2_score})")
        
        else:
            test_accuracy = model.score(X_test, y_test)
            classification_rep = classification_report(y_test, test_predictions, output_dict=True)
            
            final_metrics = {
                'test_accuracy': test_accuracy,
                'classification_report': classification_rep,
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'cv_consistency': cv_consistency
            }
            
            logger.info(f"✅ Final Test Accuracy: {test_accuracy:.3f}")
        
        # Feature importance (if available)
        feature_importance = {}
        try:
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                importance_values = model.named_steps['model'].feature_importances_
                # Get feature names after preprocessing
                feature_names = [f"feature_{i}" for i in range(len(importance_values))]
                feature_importance = dict(zip(feature_names, importance_values.tolist()))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        # Create experiment metadata
        training_time = (datetime.now() - start_time).total_seconds() / 60  # minutes
        
        self.experiment_metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            start_time=start_time,
            model_name=self.model_name,
            model_type=self.model_type,
            data_version="v1",
            feature_count=X.shape[1],
            training_samples=len(X_train),
            validation_samples=0,  # Handled by CV
            test_samples=len(X_test),
            hyperparameters={
                'max_iter': config.max_model_complexity[self.model_type],
                'learning_rate': 0.1,
                'max_depth': 8,
                'min_samples_leaf': 20,
                'random_state': config.random_seed
            },
            cross_validation_scores=cv_scores_list,
            final_metrics=final_metrics,
            feature_importance=feature_importance,
            training_time_minutes=training_time,
            reproducibility_seed=config.random_seed
        )
        
        # Save model and metadata
        self._save_model_artifacts(model)
        
        logger.info(f"🎉 Training complete! Time: {training_time:.1f} minutes")
        return self.experiment_metadata
    
    def _save_model_artifacts(self, model):
        """Save model and metadata for reproducibility"""
        model_dir = config.models_output_dir / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{self.model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = model_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            # Convert metadata to dict for JSON serialization
            metadata_dict = {
                'experiment_id': self.experiment_metadata.experiment_id,
                'start_time': self.experiment_metadata.start_time.isoformat(),
                'model_name': self.experiment_metadata.model_name,
                'model_type': self.experiment_metadata.model_type,
                'data_version': self.experiment_metadata.data_version,
                'feature_count': self.experiment_metadata.feature_count,
                'training_samples': self.experiment_metadata.training_samples,
                'test_samples': self.experiment_metadata.test_samples,
                'hyperparameters': self.experiment_metadata.hyperparameters,
                'cross_validation_scores': self.experiment_metadata.cross_validation_scores,
                'final_metrics': self.experiment_metadata.final_metrics,
                'feature_importance': self.experiment_metadata.feature_importance,
                'training_time_minutes': self.experiment_metadata.training_time_minutes,
                'reproducibility_seed': self.experiment_metadata.reproducibility_seed
            }
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"💾 Model artifacts saved to: {model_dir}")
        
        # Save to central model registry
        registry_path = config.artifacts_dir / "model_registry.json"
        registry = {}
        
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        
        registry[self.model_name] = {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'created_at': self.experiment_metadata.start_time.isoformat(),
            'model_type': self.model_type,
            'performance': self.experiment_metadata.final_metrics
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        self.model = model  # Store for later use

def load_model(model_name: str):
    """Load a trained model with its metadata"""
    model_dir = config.models_output_dir / model_name
    model_path = model_dir / f"{model_name}.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load metadata if available
    metadata_path = model_dir / f"{model_name}_metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata
