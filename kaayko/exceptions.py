"""
Custom exceptions for Kaayko Paddle Intelligence
"""

class KaaykoError(Exception):
    """Base exception for all Kaayko errors"""
    pass

class ModelNotFoundError(KaaykoError):
    """Raised when a required model file is not found"""
    pass

class PredictionError(KaaykoError):
    """Raised when prediction fails"""
    pass

class DataValidationError(KaaykoError):
    """Raised when input data validation fails"""
    pass

class APIError(KaaykoError):
    """Raised when weather API calls fail"""
    pass
