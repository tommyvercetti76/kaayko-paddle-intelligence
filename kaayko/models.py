"""
Data models for Kaayko predictions
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime

class SkillLevel(Enum):
    """Paddle skill level recommendations"""
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate" 
    ADVANCED = "Advanced"
    EXPERT_ONLY = "Expert Only"

@dataclass
class PaddleScore:
    """Paddle safety prediction result"""
    paddle_score: float  # 1.0 - 5.0 scale
    skill_level: SkillLevel
    confidence: float  # 0.0 - 1.0
    model_used: str  # e.g., "global", "spec__region__North_America"
    
    # Weather conditions used
    temperature_c: float
    wind_kph: float
    precipitation_mm: float
    
    # Additional insights
    estimated_water_temp_c: float
    estimated_wave_height_m: float
    uv_index: float
    
    # Metadata
    location: tuple[float, float]  # (lat, lon)
    prediction_time: datetime
    description: str
    
    @property
    def safety_rating(self) -> str:
        """Human-readable safety rating"""
        if self.paddle_score >= 4.5:
            return "Excellent conditions"
        elif self.paddle_score >= 3.5:
            return "Good conditions" 
        elif self.paddle_score >= 2.5:
            return "Moderate conditions"
        elif self.paddle_score >= 1.5:
            return "Poor conditions"
        else:
            return "Dangerous conditions"

@dataclass 
class WeatherInput:
    """Input weather data for predictions"""
    latitude: float
    longitude: float
    datetime: datetime
    
    # Required weather fields
    temp_c: float
    wind_kph: float
    wind_dir: int
    humidity: int
    cloud: int
    uv: float
    precip_mm: float
    pressure_mb: float
    
    # Optional fields
    dew_point_c: Optional[float] = None
    feelslike_c: Optional[float] = None
    gust_kph: Optional[float] = None
    vis_km: Optional[float] = None
    is_day: Optional[int] = None
    will_it_rain: Optional[int] = None
    will_it_snow: Optional[int] = None
