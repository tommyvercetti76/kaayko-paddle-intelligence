"""
Main prediction interface for Kaayko Paddle Intelligence
"""
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .models import PaddleScore, SkillLevel, WeatherInput
from .exceptions import ModelNotFoundError, PredictionError, DataValidationError


class PaddlePredictor:
    """
    Main interface for paddle safety predictions
    
    Example:
        >>> predictor = PaddlePredictor()
        >>> result = predictor.predict(lat=40.7128, lon=-74.0060)
        >>> print(f"Paddle Score: {result.paddle_score}/5")
    """
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize predictor with model directory
        
        Args:
            models_dir: Path to directory containing trained models
                       Defaults to ./models relative to package
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / "models"
        
        self.models_dir = Path(models_dir)
        self._models = {}  # Model cache
        
        # Load global models (required)
        self._load_global_models()
    
    def _load_global_models(self):
        """Load required global models"""
        global_reg = self.models_dir / "global_reg.pkl"
        global_clf = self.models_dir / "global_clf.pkl"
        
        if not global_reg.exists() or not global_clf.exists():
            raise ModelNotFoundError(
                f"Global models not found in {self.models_dir}. "
                f"Please run training pipeline first."
            )
        
        self._models["global_reg"] = self._load_model(global_reg)
        self._models["global_clf"] = self._load_model(global_clf)
    
    def _load_model(self, path: Path):
        """Load a pickle model file"""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise ModelNotFoundError(f"Failed to load model {path}: {e}")
    
    def predict(
        self,
        latitude: float,
        longitude: float,
        datetime: Optional[datetime] = None,
        weather_data: Optional[WeatherInput] = None
    ) -> PaddleScore:
        """
        Predict paddle safety score for a location and time
        
        Args:
            latitude: Location latitude (-90 to 90)
            longitude: Location longitude (-180 to 180)  
            datetime: Prediction time (defaults to now)
            weather_data: Pre-fetched weather data (optional)
            
        Returns:
            PaddleScore object with prediction results
            
        Raises:
            DataValidationError: Invalid input parameters
            PredictionError: Prediction failed
        """
        # Validate inputs
        if not (-90 <= latitude <= 90):
            raise DataValidationError(f"Invalid latitude: {latitude}")
        if not (-180 <= longitude <= 180):
            raise DataValidationError(f"Invalid longitude: {longitude}")
        
        if datetime is None:
            datetime = datetime.now()
        
        # Get weather data if not provided
        if weather_data is None:
            # TODO: Integrate with weather API
            raise NotImplementedError("Weather API integration not yet implemented")
        
        # Convert to prediction format
        try:
            # Create feature vector from weather data
            features = self._create_feature_vector(weather_data)
            
            # Choose best model for this location
            model_tag, reg_model, clf_model = self._choose_models(latitude, longitude)
            
            # Make predictions
            paddle_score = float(reg_model["pipe"].predict(features)[0])
            skill_level = SkillLevel(clf_model["pipe"].predict(features)[0])
            
            # Calculate additional insights
            water_temp = self._estimate_water_temperature(
                weather_data.temp_c, latitude, datetime.month
            )
            wave_height = min(0.016 * (weather_data.wind_kph ** 1.2) / 10, 3.0)
            
            return PaddleScore(
                paddle_score=paddle_score,
                skill_level=skill_level,
                confidence=0.85,  # TODO: Calculate actual confidence
                model_used=model_tag,
                temperature_c=weather_data.temp_c,
                wind_kph=weather_data.wind_kph,
                precipitation_mm=weather_data.precip_mm,
                estimated_water_temp_c=water_temp,
                estimated_wave_height_m=wave_height,
                uv_index=weather_data.uv,
                location=(latitude, longitude),
                prediction_time=datetime,
                description=f"Paddle score {paddle_score:.1f}/5 for {skill_level.value} paddlers"
            )
            
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e}")
    
    def _create_feature_vector(self, weather: WeatherInput) -> pd.DataFrame:
        """Convert weather input to ML feature vector"""
        # TODO: Implement feature engineering matching training pipeline
        features = {
            "temp_c": weather.temp_c,
            "wind_kph": weather.wind_kph, 
            "humidity": weather.humidity,
            "cloud": weather.cloud,
            "uv": weather.uv,
            "precip_mm": weather.precip_mm,
            "pressure_mb": weather.pressure_mb,
            # Add more features as needed
        }
        return pd.DataFrame([features])
    
    def _choose_models(self, latitude: float, longitude: float):
        """Choose best models for this location"""
        # TODO: Implement specialist model selection logic
        # For now, always use global models
        return "global", self._models["global_reg"], self._models["global_clf"]
    
    def _estimate_water_temperature(self, air_temp: float, latitude: float, month: int) -> float:
        """Estimate water temperature from air temperature"""
        # Simplified version of the algorithm from training pipeline
        water_temp = air_temp - 3.0
        
        # Seasonal adjustment
        import math
        seasonal_factor = math.sin(math.radians((month - 4) * 30)) * 2.0
        if latitude < 0:  # Southern hemisphere
            seasonal_factor *= -1
        water_temp += seasonal_factor
        
        # Latitude adjustment
        latitude_factor = (abs(latitude) - 20) * 0.1
        water_temp -= max(0, latitude_factor)
        
        return round(water_temp, 1)
    
    def list_available_models(self) -> list[str]:
        """List all available models"""
        models = []
        for path in self.models_dir.glob("*.pkl"):
            models.append(path.stem)
        return sorted(models)
