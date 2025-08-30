"""
Kaayko Paddle Intelligence Training Suite
=========================================
A comprehensive machine learning training system for paddle safety prediction
based on weather data from 2,779 real lakes with 6M+ training records.

This suite implements industry best practices for:
- Unbiased data handling
- Rigorous cross-validation
- Hierarchical model architecture
- Reproducible experiments
- Collaborative development

Author: Kaayko Team
License: MIT
Version: 1.0.0
"""

# Training Suite Configuration
SUITE_VERSION = "1.0.0"
SUITE_NAME = "Kaayko Paddle Intelligence Training Suite"

# Data integrity verification
VERIFIED_LAKE_COUNT = 2779
VERIFIED_RECORD_COUNT = 6069336
VERIFIED_DATA_SIZE_GB = 35.21

# Model naming conventions
MODEL_NAMES = {
    "global": "kaayko_global_paddle_predictor_v1",
    "continental": {
        "europe": "kaayko_europe_specialist_v1", 
        "north_america": "kaayko_north_america_specialist_v1",
        "asia": "kaayko_asia_specialist_v1"
    },
    "regional": {
        "scotland": "kaayko_scotland_regional_v1",
        "switzerland": "kaayko_switzerland_regional_v1", 
        "finland": "kaayko_finland_regional_v1",
        "italy": "kaayko_italy_regional_v1",
        "india_kerala": "kaayko_india_kerala_regional_v1"
    }
}

# Collaboration metadata
CONTRIBUTORS = [
    {"name": "Kaayko Team", "role": "Primary Development"},
    {"github": "https://github.com/kaayko/paddle-intelligence", "role": "Open Source"},
]

DOCUMENTATION_URLS = {
    "api_reference": "https://docs.kaayko.ai/api",
    "training_guide": "https://docs.kaayko.ai/training", 
    "contribution_guide": "https://github.com/kaayko/paddle-intelligence/CONTRIBUTING.md"
}
