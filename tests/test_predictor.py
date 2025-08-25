"""
Tests for Kaayko PaddlePredictor class
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from pathlib import Path

from kaayko import PaddlePredictor
from kaayko.models import WeatherInput, SkillLevel
from kaayko.exceptions import ModelNotFoundError, DataValidationError


class TestPaddlePredictor:
    """Test suite for PaddlePredictor"""
    
    def test_init_missing_models(self):
        """Test initialization with missing model files"""
        with pytest.raises(ModelNotFoundError):
            PaddlePredictor(models_dir="/nonexistent/path")
    
    @patch.object(PaddlePredictor, '_load_global_models')
    def test_init_success(self, mock_load):
        """Test successful initialization"""
        mock_load.return_value = None
        predictor = PaddlePredictor()
        assert predictor.models_dir.name in ["models", "kaayko_pipeline"]
        mock_load.assert_called_once()
    
    def test_predict_invalid_latitude(self):
        """Test prediction with invalid latitude"""
        with patch.object(PaddlePredictor, '_load_global_models'):
            predictor = PaddlePredictor()
            
            with pytest.raises(DataValidationError):
                predictor.predict(latitude=95.0, longitude=0.0)  # > 90
            
            with pytest.raises(DataValidationError):
                predictor.predict(latitude=-95.0, longitude=0.0)  # < -90
    
    def test_predict_invalid_longitude(self):
        """Test prediction with invalid longitude"""
        with patch.object(PaddlePredictor, '_load_global_models'):
            predictor = PaddlePredictor()
            
            with pytest.raises(DataValidationError):
                predictor.predict(latitude=0.0, longitude=185.0)  # > 180
                
            with pytest.raises(DataValidationError):
                predictor.predict(latitude=0.0, longitude=-185.0)  # < -180
    
    def test_estimate_water_temperature(self):
        """Test water temperature estimation"""
        with patch.object(PaddlePredictor, '_load_global_models'):
            predictor = PaddlePredictor()
            
            # Test northern hemisphere summer
            water_temp = predictor._estimate_water_temperature(25.0, 40.0, 7)  # July
            assert isinstance(water_temp, float)
            assert water_temp < 25.0  # Should be cooler than air
            
            # Test southern hemisphere winter  
            water_temp_south = predictor._estimate_water_temperature(15.0, -40.0, 7)  # July
            assert isinstance(water_temp_south, float)
    
    @patch.object(PaddlePredictor, '_load_global_models')
    def test_list_available_models(self, mock_load):
        """Test listing available models"""
        predictor = PaddlePredictor()
        
        # Mock the models directory
        with patch.object(predictor.models_dir, 'glob') as mock_glob:
            mock_glob.return_value = [
                Mock(stem="global_reg"),
                Mock(stem="global_clf"),
                Mock(stem="spec__region__North_America_reg")
            ]
            
            models = predictor.list_available_models()
            assert "global_reg" in models
            assert "global_clf" in models
            assert "spec__region__North_America_reg" in models


class TestWeatherInput:
    """Test suite for WeatherInput model"""
    
    def test_weather_input_creation(self):
        """Test creating WeatherInput object"""
        weather = WeatherInput(
            latitude=40.7128,
            longitude=-74.0060,
            datetime=datetime(2025, 8, 25, 10, 0),
            temp_c=22.5,
            wind_kph=12.0,
            wind_dir=270,
            humidity=65,
            cloud=25,
            uv=7.2,
            precip_mm=0.0,
            pressure_mb=1013.2
        )
        
        assert weather.latitude == 40.7128
        assert weather.temp_c == 22.5
        assert weather.wind_kph == 12.0
        assert weather.dew_point_c is None  # Optional field
