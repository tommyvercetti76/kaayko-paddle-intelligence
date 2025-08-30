"""
Kaayko Training Suite Configuration
==================================
Central configuration with professional naming conventions and data integrity checks.
"""

from pathlib import Path
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class DataIntegrityConfig:
    """Data integrity verification configuration"""
    expected_lake_count: int = 2779
    expected_min_records: int = 6000000  # 6M+ records minimum
    expected_data_size_gb: float = 35.0  # ~35GB minimum
    verified_continents: List[str] = field(default_factory=lambda: ["Europe", "North America", "Asia"])

@dataclass 
class ModelNamingConfig:
    """Professional model naming conventions"""
    project_prefix: str = "kaayko"
    version: str = "v1"
    
    # Model categories
    global_model_name: str = field(init=False)
    continental_models: Dict[str, str] = field(default_factory=dict)
    national_models: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self.global_model_name = f"{self.project_prefix}_global_paddle_predictor_{self.version}"
        
        if not self.continental_models:
            self.continental_models = {
                "europe": f"{self.project_prefix}_europe_specialist_{self.version}",
                "north_america": f"{self.project_prefix}_north_america_specialist_{self.version}", 
                "asia": f"{self.project_prefix}_asia_specialist_{self.version}"
            }
        
        if not self.national_models:
            self.national_models = {
                "usa": f"{self.project_prefix}_usa_national_{self.version}",
                "india": f"{self.project_prefix}_india_national_{self.version}",
                "canada": f"{self.project_prefix}_canada_national_{self.version}",
                "scotland": f"{self.project_prefix}_scotland_national_{self.version}",
                "switzerland": f"{self.project_prefix}_switzerland_national_{self.version}",
                "finland": f"{self.project_prefix}_finland_national_{self.version}",
                "italy": f"{self.project_prefix}_italy_national_{self.version}",
                "norway": f"{self.project_prefix}_norway_national_{self.version}"
            }

@dataclass
class TrainingConfig:
    """Professional ML training configuration with bias prevention"""
    
    # Data paths (verified real data only)
    raw_data_dir: Path = Path("/path/to/your/lake_data")
    processed_data_dir: Path = Path("./data/processed")
    models_output_dir: Path = Path("./models")
    logs_dir: Path = Path("./logs")
    artifacts_dir: Path = Path("./artifacts")
    
    # Data integrity
    integrity_config: DataIntegrityConfig = field(default_factory=DataIntegrityConfig)
    
    # Model naming
    naming_config: ModelNamingConfig = field(default_factory=ModelNamingConfig)
    
    # Training methodology (industry best practices)
    random_seed: int = 42  # Reproducibility
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    
    # Bias prevention
    stratify_by_continent: bool = True
    group_by_lake: bool = True  # Prevent data leakage
    balance_seasonal_data: bool = True
    
    # Model parameters (conservative for robustness)
    max_training_time_hours: int = 12
    early_stopping_patience: int = 10
    max_model_complexity: Dict[str, int] = None
    
    # Curated National Models Priority
    priority_national_models: List[str] = None
    
    # Memory management
    chunk_size_mb: int = 500  # Process in 500MB chunks
    max_memory_usage_gb: float = 8.0
    
    # Quality thresholds
    min_model_r2_score: float = 0.75
    min_cross_val_consistency: float = 0.90
    
    def __post_init__(self):
        if self.max_model_complexity is None:
            self.max_model_complexity = {
                "global": 1000,  # More complex for global model
                "continental": 800,  # Medium complexity
                "national": 900,   # High complexity for national specialists
                "regional": 600   # Simpler for regional models (legacy)
            }
        
        if self.priority_national_models is None:
            self.priority_national_models = [
                "usa",      # Priority 1: Largest North American dataset
                "india"     # Priority 2: Largest Asian dataset with diverse climate zones
            ]
        
        # Ensure directories exist
        for dir_path in [self.processed_data_dir, self.models_output_dir, 
                        self.logs_dir, self.artifacts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class CollaborationConfig:
    """Configuration for collaborative development"""
    
    # Version control
    git_repo_url: str = "https://github.com/kaayko/paddle-intelligence"
    documentation_url: str = "https://docs.kaayko.ai"
    
    # API endpoints
    model_registry_url: str = "https://models.kaayko.ai"
    experiment_tracking_url: str = "https://experiments.kaayko.ai"
    
    # Collaboration tools
    enable_experiment_tracking: bool = True
    enable_model_versioning: bool = True
    enable_automated_testing: bool = True
    
    # Code quality
    code_style: str = "black"
    max_line_length: int = 88
    docstring_style: str = "google"
    
    # CI/CD
    run_tests_on_commit: bool = True
    auto_deploy_on_merge: bool = False  # Manual approval for production

# Global configuration instance
config = TrainingConfig()
data_integrity = DataIntegrityConfig()  
model_naming = ModelNamingConfig()
collaboration = CollaborationConfig()

# Environment-specific overrides
if os.getenv("KAAYKO_ENV") == "production":
    config.max_training_time_hours = 24
    config.min_model_r2_score = 0.85
    collaboration.auto_deploy_on_merge = True

elif os.getenv("KAAYKO_ENV") == "development":
    config.cross_validation_folds = 3  # Faster for dev
    config.chunk_size_mb = 100  # Smaller chunks for dev
    
# Export key configurations
__all__ = [
    "config",
    "data_integrity", 
    "model_naming",
    "collaboration",
    "TrainingConfig",
    "DataIntegrityConfig",
    "ModelNamingConfig", 
    "CollaborationConfig"
]
