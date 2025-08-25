"""
Kaayko Paddle Intelligence

The world's first open-source paddle safety prediction system.
Transform weather data into actionable paddle safety scores using machine learning.
"""

__version__ = "1.0.0"
__author__ = "Kaayko Team"
__email__ = "team@kaayko.ai"

from .predictor import PaddlePredictor
from .models import PaddleScore, SkillLevel
from .exceptions import KaaykoError, ModelNotFoundError, PredictionError

__all__ = [
    "PaddlePredictor",
    "PaddleScore", 
    "SkillLevel",
    "KaaykoError",
    "ModelNotFoundError", 
    "PredictionError"
]
