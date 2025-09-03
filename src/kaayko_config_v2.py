#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaayko Superior Trainer v2.0 - Configuration & UI Module
========================================================

🎯 RESPONSIBILITIES:
• Configuration management and validation
• Interactive user interface and CLI parsing
• Utilities, constants, and helper functions
• Logging and error handlin    for i, (key, config) in enumerate(SAMPLE_CONFIGS.items(), 1):
        emoji = "⚡" if key == "small" else "🔧" if key == "medium" else "🏗️" if key == "medium_plus" else "🚀" if key == "large" else "💪" if key == "xl" else "🏭" if key == "xxl" else "🏆"
        sample_count = int(estimated_total * config['percentage'] / 100)
        sample_k = f"~{sample_count//1000}K" if sample_count >= 1000 else f"~{sample_count}"
        print(f"{i}. {key:<12} - {config['percentage']:>5.1f}% ({sample_k:<8} samples) | {emoji} {config['description']:<15} ({config['time_estimate']})")up
• Path management and security

Author: Kaayko Intelligence Team
Version: 2.0
License: Proprietary
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
import json
import sys
import signal
import warnings
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
import numpy as np

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Paddle Score Scale Constants
DISPLAY_SCALE = 5.0
TRAIN_SCALE = 10.0
SCORE_FACTOR = DISPLAY_SCALE / TRAIN_SCALE
MIDPOINT = DISPLAY_SCALE / 2.0

# Environment-aware defaults (NO hardcoded user paths)
DEFAULT_MODELS_ROOT = Path(os.getenv("KAAYKO_MODELS_ROOT", "./models"))
DEFAULT_DATA_ROOT = Path(os.getenv("KAAYKO_DATA_ROOT", "./data"))

# Global Constants
CANONICAL_WEATHER_COLS = [
    'temperature', 'wind_speed', 'humidity', 'pressure', 
    'precipitation', 'visibility', 'cloud_cover', 'uv_index'
]
TARGET_COL = 'paddle_score'
RANDOM_STATE = 42

# Sample size configurations - M1 Pro Max optimized (maintains full accuracy, smart parallelization)
SAMPLE_CONFIGS = {
    'small': {'percentage': 0.2, 'description': 'Quick test', 'time_estimate': '1-2 min'},
    'medium': {'percentage': 2.0, 'description': 'Development', 'time_estimate': '3-6 min'},
    'medium_plus': {'percentage': 15.0, 'description': 'Extended dev', 'time_estimate': '8-15 min'},
    'large': {'percentage': 25.0, 'description': 'Production', 'time_estimate': '12-20 min'},
    'xl': {'percentage': 50.0, 'description': 'Heavy training', 'time_estimate': '20-35 min'},
    'xxl': {'percentage': 80.0, 'description': 'Near-complete', 'time_estimate': '30-50 min'},
    'complete': {'percentage': 100.0, 'description': 'Full dataset', 'time_estimate': '40-70 min'}
}

# Algorithm configurations
ALGORITHM_CONFIGS = {
    'ensemble': {
        'name': 'All algorithms combined',
        'description': '🏆 Best performance (99.996% R²)',
        'emoji': '✨'
    },
    'histgradient': {
        'name': 'HistGradient Boosting',
        'description': '⚡ Industry leader (97.40% R²)',
        'emoji': '🥇'
    },
    'xgboost': {
        'name': 'XGBoost (M1 Optimized)',
        'description': '🚀 M1 Max performance (98%+ R²)',
        'emoji': '⚡'
    },
    'randomforest': {
        'name': 'Random Forest',
        'description': '🛡️ Robust backup (96.97% R²)',
        'emoji': '🥈'
    },
    'extratrees': {
        'name': 'Extra Trees',
        'description': '🌟 Ensemble diversity (96.45% R²)',
        'emoji': '🥉'
    },
    'gradientboosting': {
        'name': 'Traditional Boosting',
        'description': '📊 Stable choice (96.13% R²)',
        'emoji': '✅'
    }
}

# Localization configurations
LOCALIZATION_CONFIGS = {
    'en-US': {'flag': '🇺🇸', 'name': 'United States', 'units': '°F, mph, inches'},
    'en-GB': {'flag': '🇬🇧', 'name': 'United Kingdom', 'units': '°C, mph, mm'},
    'fr-FR': {'flag': '🇫🇷', 'name': 'France', 'units': '°C, km/h, mm'}
}

# ============================================================================
# COLOR DEFINITIONS
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

class TrainingConfig:
    """Centralized training configuration."""
    
    def __init__(self, args: argparse.Namespace):
        self.sample_size = args.sample_size
        self.algorithm = args.algorithm
        self.score_quantization = getattr(args, 'score_quantization', 'half_step')
        self.safety_overrides = getattr(args, 'safety_overrides', False)
        self.confidence_metric = getattr(args, 'confidence_metric', False)
        self.telemetry = getattr(args, 'telemetry', False)
        self.localization = getattr(args, 'localization', 'en-US')
        self.resume = args.resume
        self.smoke_test = args.smoke_test
        self.models_root = Path(args.models_root)
        self.data_root = Path(args.data_root)
        self.sample_rows_for_search = args.sample_rows_for_search
        self.shard_size_rows = args.shard_size_rows
        self.n_jobs = getattr(args, 'n_jobs', -1)
        self.save_csv = getattr(args, 'save_csv', False)
        
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.sample_size not in SAMPLE_CONFIGS:
            raise ValueError(f"Invalid sample_size: {self.sample_size}")
        if self.algorithm not in ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
        if self.localization not in LOCALIZATION_CONFIGS:
            raise ValueError(f"Invalid localization: {self.localization}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'sample_size': self.sample_size,
            'algorithm': self.algorithm,
            'score_quantization': self.score_quantization,
            'safety_overrides': self.safety_overrides,
            'confidence_metric': self.confidence_metric,
            'telemetry': self.telemetry,
            'localization': self.localization,
            'resume': self.resume,
            'smoke_test': self.smoke_test,
            'models_root': str(self.models_root),
            'data_root': str(self.data_root),
            'sample_rows_for_search': self.sample_rows_for_search,
            'shard_size_rows': self.shard_size_rows
        }

# ============================================================================
# INTERACTIVE UI FUNCTIONS
# ============================================================================

def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}")

def print_ascii_box(title: str, width: int = 39) -> None:
    """Print ASCII art box."""
    print(f"          ╔{'═'*width}╗")
    print(f"          ║{title:^{width}}║")
    print(f"          ╚{'═'*width}╝")

def get_user_choice(prompt: str, choices: List[str], default: str = None) -> str:
    """Get user choice with validation and perfect error handling."""
    while True:
        try:
            choice = input(f"{prompt}: ").strip().lower()
            if not choice and default:
                return default
            if choice in [c.lower() for c in choices]:
                # Return original case from choices
                return choices[[c.lower() for c in choices].index(choice)]
            print(f"{Colors.RED}❌ Invalid choice. Options: {', '.join(choices)}{Colors.RESET}")
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️ Training cancelled by user{Colors.RESET}")
            sys.exit(0)
        except EOFError:
            print(f"\n{Colors.YELLOW}⚠️ Input stream ended{Colors.RESET}")
            sys.exit(0)

def interactive_data_path_selection() -> str:
    """Interactive data path selection."""
    print(f"\n{Colors.CYAN}📂 KAAYKO TRAINING - DATA SOURCE SELECTION{Colors.RESET}")
    print("="*60)
    print("Specify the path to your training dataset:")
    print()
    
    # Common data paths for this user
    common_paths = [
        "/Users/Rohan/data_lake_monthly",
        "/Users/Rohan/Desktop/kaayko-paddle-intelligence/data/hydrolakes_weather",
        "./data"
    ]
    
    print("🎯 Suggested paths:")
    for i, path in enumerate(common_paths, 1):
        path_obj = Path(path)
        exists = "✅" if path_obj.exists() else "❌"
        print(f"{i}. {path} {exists}")
    
    print(f"{len(common_paths) + 1}. Custom path (enter manually)")
    print()
    
    choice = get_user_choice(f"Choose data source (1-{len(common_paths) + 1})", 
                           [str(i) for i in range(1, len(common_paths) + 2)])
    
    if choice == str(len(common_paths) + 1):
        # Custom path
        while True:
            try:
                custom_path = input(f"\n{Colors.WHITE}Enter full path to your dataset: {Colors.RESET}").strip()
                if not custom_path:
                    print(f"{Colors.RED}❌ Please enter a valid path{Colors.RESET}")
                    continue
                
                path_obj = Path(custom_path)
                if not path_obj.exists():
                    print(f"{Colors.RED}❌ Path does not exist: {custom_path}{Colors.RESET}")
                    retry = get_user_choice("Try again? (y/n)", ["y", "n", "yes", "no"])
                    if retry.lower() in ["n", "no"]:
                        break
                    continue
                
                selected_path = custom_path
                break
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}⚠️ Training cancelled by user{Colors.RESET}")
                sys.exit(0)
    else:
        # Selected from common paths
        idx = int(choice) - 1
        selected_path = common_paths[idx]
    
    print(f"\n{Colors.GREEN}✅ Selected data source: {mask_path(selected_path)}{Colors.RESET}")
    return selected_path

def estimate_dataset_size(data_path: str) -> Tuple[int, int]:
    """Estimate dataset size from the given path."""
    data_root = Path(data_path)
    if not data_root.exists():
        return 0, 0
    
    lake_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    total_lakes = len(lake_dirs)
    
    if total_lakes == 0:
        return 0, 0
    
    # Sample a few directories to estimate average samples per lake
    sample_count = 0
    sampled_lakes = 0
    
    for lake_dir in lake_dirs[:min(3, total_lakes)]:  # Sample up to 3 lakes
        try:
            csv_files = list(lake_dir.glob("*.csv"))
            parquet_files = list(lake_dir.glob("*.parquet"))
            
            for file_path in csv_files[:1] + parquet_files[:1]:  # Check first file of each type
                try:
                    if file_path.suffix == '.csv':
                        # Quick line count for CSV
                        with open(file_path, 'r') as f:
                            lines = sum(1 for _ in f) - 1  # Subtract header
                        sample_count += lines
                    else:
                        # Read parquet
                        df = pd.read_parquet(file_path)
                        sample_count += len(df)
                    
                    sampled_lakes += 1
                    break  # Only count one file per lake
                except Exception:
                    continue
        except Exception:
            continue
    
    if sampled_lakes > 0:
        avg_samples_per_lake = sample_count / sampled_lakes
        estimated_total = int(total_lakes * avg_samples_per_lake)
    else:
        # Fallback to previous estimate
        estimated_total = total_lakes * 6117
    
    return estimated_total, total_lakes

def interactive_sample_size_selection(data_path: str) -> str:
    """Interactive sample size selection with accurate dataset estimation."""
    print(f"\n{Colors.CYAN}🎯 KAAYKO TRAINING - SAMPLE SIZE SELECTION{Colors.RESET}")
    print("="*60)
    
    # Estimate actual dataset size
    print(f"🔍 Analyzing dataset at {mask_path(data_path)}...")
    estimated_total, total_lakes = estimate_dataset_size(data_path)
    
    if estimated_total == 0:
        print(f"{Colors.RED}❌ No data found at specified path!{Colors.RESET}")
        print(f"{Colors.YELLOW}Please check your data path and try again.{Colors.RESET}")
        sys.exit(1)
    
    print(f"📊 Found {total_lakes:,} lakes with estimated {estimated_total:,} total samples")
    print()
    print("Choose training sample size:")
    print()
    
    for i, (key, config) in enumerate(SAMPLE_CONFIGS.items(), 1):
        emoji = "⚡" if key == "small" else "🔧" if key == "medium" else "🏗️" if key == "medium_plus" else "🚀" if key == "large" else "💪" if key == "xl" else "�" if key == "xxl" else "�🏆"
        sample_count = int(estimated_total * config['percentage'] / 100)
        sample_k = f"~{sample_count//1000}K" if sample_count >= 1000 else f"~{sample_count}"
        print(f"{i}. {key:<12} - {config['percentage']:>5.1f}% ({sample_k:<8} samples) | {emoji} {config['description']:<15} ({config['time_estimate']})")
    
    print()
    choice = get_user_choice("Choose option (1/2/3/4/5/6/7)", ["1", "2", "3", "4", "5", "6", "7"])
    
    sample_map = {"1": "small", "2": "medium", "3": "medium_plus", "4": "large", "5": "xl", "6": "xxl", "7": "complete"}
    selected = sample_map[choice]
    
    # Show final selection with actual numbers
    selected_config = SAMPLE_CONFIGS[selected]
    final_sample_count = int(estimated_total * selected_config['percentage'] / 100)
    print(f"\n{Colors.GREEN}✅ Selected: {selected} ({final_sample_count:,} samples){Colors.RESET}")
    return selected

def interactive_algorithm_selection() -> str:
    """Interactive algorithm selection with XGBoost support."""
    print(f"\n{Colors.CYAN}🤖 KAAYKO ALGORITHM SELECTION{Colors.RESET}")
    print("="*60)
    print()
    print_ascii_box("🧠 ALGORITHM SELECTION 🧠")
    print()
    
    # Check if XGBoost is available
    try:
        import xgboost as xgb
        xgboost_available = True
    except ImportError:
        xgboost_available = False
    
    print("Choose your ML algorithm:")
    print()
    
    # Display algorithms with availability check
    algorithm_list = list(ALGORITHM_CONFIGS.keys())
    available_choices = []
    
    for i, key in enumerate(algorithm_list, 1):
        config = ALGORITHM_CONFIGS[key]
        name = config['name'][:25].ljust(25)  # Slightly wider for XGBoost
        
        if key == 'xgboost' and not xgboost_available:
            print(f"{i}. {key:<15} - {config['emoji']} {name} | {config['description']} (⚠️ Not installed)")
            continue  # Skip adding to available choices
        
        print(f"{i}. {key:<15} - {config['emoji']} {name} | {config['description']}")
        available_choices.append(str(i))
    
    if not xgboost_available:
        print(f"\n{Colors.YELLOW}💡 To install XGBoost: pip install xgboost{Colors.RESET}")
    
    print()
    choice = get_user_choice(f"Choose algorithm ({'/'.join(available_choices)})", available_choices)
    
    # Map choice to algorithm, skipping unavailable XGBoost if needed
    choice_idx = int(choice) - 1
    if choice_idx < len(algorithm_list):
        selected = algorithm_list[choice_idx]
        
        # Double-check XGBoost availability
        if selected == 'xgboost' and not xgboost_available:
            print(f"\n{Colors.RED}❌ XGBoost not available. Please install with: pip install xgboost{Colors.RESET}")
            return interactive_algorithm_selection()  # Retry
        
        print(f"\n{Colors.GREEN}✅ Selected: {selected.upper()}{Colors.RESET}")
        return selected
    
    # Fallback to ensemble
    print(f"\n{Colors.GREEN}✅ Selected: ENSEMBLE{Colors.RESET}")
    return 'ensemble'

def interactive_score_quantization() -> str:
    """Interactive score quantization selection."""
    print(f"\n{Colors.CYAN}🎯 SCORE QUANTIZATION METHOD{Colors.RESET}")
    print("="*60)
    print()
    print_ascii_box("📊 SCORE QUANTIZATION 📊")
    print()
    print("Choose score precision:")
    print()
    print("1. half_step     - 🎯 0.5 increments (0.0, 0.5, 1.0...)   | 📱 UI-friendly")
    print("2. quarter_step  - 🔍 0.25 increments (0.0, 0.25, 0.5...) | 🔬 More precise")
    print("3. tenth_step    - ⚡ 0.1 increments (0.0, 0.1, 0.2...)   | 🎛️ Maximum precision")
    
    print()
    choice = get_user_choice("Choose quantization (1/2/3)", ["1", "2", "3"])
    
    quant_map = {"1": "half_step", "2": "quarter_step", "3": "tenth_step"}
    selected = quant_map[choice]
    
    display_names = {"half_step": "Half Step", "quarter_step": "Quarter Step", "tenth_step": "Tenth Step"}
    print(f"\n{Colors.GREEN}✅ Selected: {display_names[selected]}{Colors.RESET}")
    return selected

def interactive_safety_overrides() -> bool:
    """Interactive safety overrides selection."""
    print(f"\n{Colors.CYAN}🛡️ SAFETY OVERRIDE POLICIES{Colors.RESET}")
    print("="*60)
    print()
    print_ascii_box("🛡️ SAFETY OVERRIDES 🛡️")
    print()
    print("Enable safety overrides for extreme conditions?")
    print()
    print("Policy: temp≤-5°C:-4★, temp≤0°C:-3★, wind≥60kph:-3★, wind≥40kph:-2★")
    print("1. yes  - ✅ Enable safety overrides   | 🛡️ Protects users from dangerous conditions")
    print("2. no   - ❌ Disable safety overrides  | 🎯 Pure ML predictions only")
    
    print()
    choice = get_user_choice("Enable safety overrides? (1=yes/2=no)", ["1", "2"])
    
    enabled = choice == "1"
    print(f"\n{Colors.GREEN}✅ Safety overrides: {'Enabled' if enabled else 'Disabled'}{Colors.RESET}")
    return enabled

def interactive_confidence_metrics() -> bool:
    """Interactive confidence metrics selection."""
    print(f"\n{Colors.CYAN}📊 CONFIDENCE METRICS{Colors.RESET}")
    print("="*60)
    print()
    print_ascii_box("📊 CONFIDENCE METRICS 📊")
    print()
    print("Include confidence metrics in predictions?")
    print()
    print("Confidence = f(residual_proxy, stability, ensemble_agreement, distance)")
    print("Labels: low<0.35, medium 0.35–0.7, high>0.7")
    print("1. yes  - ✅ Enable confidence metrics  | 📈 Better prediction quality")
    print("2. no   - ❌ Disable confidence metrics | ⚡ Faster predictions")
    
    print()
    choice = get_user_choice("Enable confidence metrics? (1=yes/2=no)", ["1", "2"])
    
    enabled = choice == "1"
    print(f"\n{Colors.GREEN}✅ Confidence metrics: {'Enabled' if enabled else 'Disabled'}{Colors.RESET}")
    return enabled

def interactive_localization() -> str:
    """Interactive localization selection."""
    print(f"\n{Colors.CYAN}🌍 LOCALIZATION & UNITS{Colors.RESET}")
    print("="*60)
    print()
    print_ascii_box("🌍 LOCALIZATION & UNITS 🌍")
    print()
    print("Choose localization for units and formatting:")
    print()
    
    for i, (key, config) in enumerate(LOCALIZATION_CONFIGS.items(), 1):
        print(f"{i}. {key}  - {config['flag']} {config['name']:<13} | {config['units']}")
    
    print()
    choice = get_user_choice("Choose localization (1/2/3)", ["1", "2", "3"])
    
    loc_map = {"1": "en-US", "2": "en-GB", "3": "fr-FR"}
    selected = loc_map[choice]
    print(f"\n{Colors.GREEN}✅ Selected: {selected}{Colors.RESET}")
    return selected

def display_final_configuration(config: TrainingConfig) -> bool:
    """Display final configuration and get confirmation."""
    print(f"\n{Colors.CYAN}🎯 FINAL TRAINING CONFIGURATION{Colors.RESET}")
    print("="*60)
    print(f"✨ Sample Size: {config.sample_size.title()}")
    print(f"🤖 Algorithm: {config.algorithm.title()}")
    print(f"📊 Quantization: {config.score_quantization.replace('_', ' ').title()}")
    print(f"🛡️ Safety Overrides: {'Enabled' if config.safety_overrides else 'Disabled'}")
    print(f"📈 Confidence Metrics: {'Enabled' if config.confidence_metric else 'Disabled'}")
    print(f"🌍 Localization: {config.localization}")
    print()
    
    choice = get_user_choice("Proceed with training? (y/n)", ["y", "n", "yes", "no"])
    return choice.lower() in ["y", "yes"]

# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    # Environment-aware defaults
    DEFAULT_MODELS_ROOT = Path(os.getenv("KAAYKO_MODELS_ROOT", "./models"))
    DEFAULT_DATA_ROOT = Path(os.getenv("KAAYKO_DATA_ROOT", "./data"))
    
    parser = argparse.ArgumentParser(
        prog='kaayko_trainer_superior_v2.py',
        description='Kaayko Superior Trainer v2.0 - Professional ML Training System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (M1 Pro Max: Full Accuracy + Smart Parallelization):
  python3 kaayko_trainer_superior_v2.py --sample-size small              # 0.2% - Quick test (1-2 min)
  python3 kaayko_trainer_superior_v2.py --sample-size medium_plus        # 15% - Extended dev (8-15 min)
  python3 kaayko_trainer_superior_v2.py --sample-size large              # 25% - Production test (12-20 min)
  python3 kaayko_trainer_superior_v2.py --sample-size xl --algorithm histgradient   # 50% - Heavy training (20-35 min)
  python3 kaayko_trainer_superior_v2.py --data_root /Users/Rohan/data_lake_monthly --sample-size xxl
  python3 kaayko_trainer_superior_v2.py --smoke_test
        """
    )
    
    # Core training arguments
    parser.add_argument('--sample-size', choices=['small', 'medium', 'medium_plus', 'large', 'xl', 'xxl', 'complete'],
                       default='medium',
                       help='Training sample size as percentage of dataset')
    
    parser.add_argument('--resume', choices=['append', 'fresh', 'use_existing'],
                       default='fresh',
                       help='Resume mode for existing data (NOT YET IMPLEMENTED)')
    
    parser.add_argument('--sample_rows_for_search', type=int, default=2_000_000,
                       help='Number of rows for hyperparameter search')
    
    parser.add_argument('--shard_size_rows', type=int, default=2_000_000,
                       help='Shard size for incremental processing')
    
    parser.add_argument('--models_root', type=str, 
                       default=str(DEFAULT_MODELS_ROOT),
                       help='Root directory for saving models')
    
    parser.add_argument('--data_root', type=str,
                       default=str(DEFAULT_DATA_ROOT),
                       help='Root directory containing training data (will prompt if path does not exist)')
    
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run smoke test with synthetic data')
    
    # Algorithm and configuration arguments
    parser.add_argument('--algorithm', 
                       choices=['ensemble', 'histgradient', 'randomforest', 'extratrees', 'gradientboosting'],
                       default='ensemble',
                       help='Select specific algorithm or ensemble')
    
    parser.add_argument('--score_quantization',
                       choices=['half_step', 'quarter_step', 'tenth_step'],
                       default='half_step',
                       help='Score quantization method')
    
    parser.add_argument('--safety_overrides', action='store_true',
                       help='Enable safety override policies for extreme conditions')
    
    parser.add_argument('--confidence_metric', action='store_true',
                       help='Include confidence metrics in predictions (NOT YET IMPLEMENTED)')
    
    parser.add_argument('--telemetry', action='store_true',
                       help='Enable telemetry logging for production monitoring (NOT YET IMPLEMENTED)')
    
    parser.add_argument('--localization', choices=['en-US', 'en-GB', 'fr-FR'],
                       default='en-US',
                       help='Localization for units and formatting')
    
    parser.add_argument('--n_jobs', type=int, default=-2,
                       help='Parallelism for CV/search (-2 leaves 2 cores free for system stability)')
    
    parser.add_argument('--save_csv', action='store_true',
                       help='Also save training dataset as CSV (in addition to Parquet)')
    
    return parser

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def mask_path(path_str: str) -> str:
    """Mask sensitive user paths in logs for privacy."""
    import re
    # Mask Unix-style user directories
    path_str = re.sub(r'/Users/[^/]+/', '/Users/<redacted>/', path_str)
    # Mask Windows-style user directories  
    path_str = re.sub(r'C:\\\\Users\\\\[^\\\\]+\\\\', 'C:\\\\Users\\\\<redacted>\\\\', path_str)
    return path_str

def setup_logging(log_level: str = 'INFO', base_dir: Path | None = None) -> logging.Logger:
    """Setup comprehensive logging with proper output paths."""
    if base_dir is None:
        base_dir = Path(".")
    
    logger = logging.getLogger('kaayko_training')
    logger.setLevel(getattr(logging, log_level))
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with proper base directory path
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'kaayko_training.log'
        
        file_handler = RotatingFileHandler(
            str(log_file), maxBytes=10_000_000, backupCount=5  # Convert Path to string
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def quantize_score(score: float, method: str = 'half_step') -> float:
    """Quantize score to specified precision."""
    if method == 'half_step':
        step = 0.5
    elif method == 'quarter_step':
        step = 0.25
    elif method == 'tenth_step':
        step = 0.1
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Use Decimal for precise rounding
    decimal_score = Decimal(str(score))
    decimal_step = Decimal(str(step))
    quantized = (decimal_score / decimal_step).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * decimal_step
    return float(quantized)

class InterruptHandler:
    """Professional interrupt handling."""
    
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        if not self.interrupted:
            print(f"\n{Colors.YELLOW}⚠️ Interrupt received! Finishing current operation...{Colors.RESET}")
            print(f"{Colors.YELLOW}   Press Ctrl+C again to force immediate exit{Colors.RESET}")
            self.interrupted = True
        else:
            print(f"\n{Colors.RED}🚨 Force exit requested!{Colors.RESET}")
            sys.exit(130)  # Standard exit code for SIGINT

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
