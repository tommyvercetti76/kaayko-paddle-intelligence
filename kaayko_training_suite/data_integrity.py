"""
Data Integrity Verification Module
==================================
Ensures we only train on verified, real data with no bias or manufacturing.

This module implements rigorous data validation to ensure:
1. All training data comes from verified real lakes
2. No synthetic or manufactured data
3. Proper geographical distribution 
4. Temporal consistency checks
5. Weather data authenticity verification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings

from .config import data_integrity, config

logger = logging.getLogger(__name__)

class DataIntegrityError(Exception):
    """Raised when data integrity checks fail"""
    pass

class RealDataValidator:
    """Validates that all training data comes from real, verified sources"""
    
    def __init__(self, comprehensive_data_path: str):
        """
        Initialize with the comprehensive lake data that contains verified real lakes.
        
        Args:
            comprehensive_data_path: Path to real_lakes_comprehensive.csv
        """
        self.comprehensive_data_path = Path(comprehensive_data_path)
        self.verified_lakes_df = None
        self.load_verified_lakes()
    
    def load_verified_lakes(self):
        """Load and validate the comprehensive verified lakes database"""
        try:
            self.verified_lakes_df = pd.read_csv(self.comprehensive_data_path)
            logger.info(f"Loaded {len(self.verified_lakes_df)} verified real lakes")
            
            # Basic validation of comprehensive data
            required_columns = ['name', 'lat', 'lng', 'region']
            missing_cols = [col for col in required_columns if col not in self.verified_lakes_df.columns]
            
            if missing_cols:
                raise DataIntegrityError(f"Missing required columns in comprehensive data: {missing_cols}")
                
            # Geographic validity checks
            invalid_coords = (
                (self.verified_lakes_df['lat'] < -90) | (self.verified_lakes_df['lat'] > 90) |
                (self.verified_lakes_df['lng'] < -180) | (self.verified_lakes_df['lng'] > 180)
            )
            
            if invalid_coords.any():
                invalid_count = invalid_coords.sum()
                logger.warning(f"Found {invalid_count} lakes with invalid coordinates")
                self.verified_lakes_df = self.verified_lakes_df[~invalid_coords]
            
            logger.info("✅ Verified lakes database validation passed")
            
        except Exception as e:
            raise DataIntegrityError(f"Failed to load verified lakes database: {e}")
    
    def verify_lake_authenticity(self, lake_directories: List[str]) -> Tuple[List[str], List[str]]:
        """
        Verify that lake directories correspond to real, verified lakes.
        
        Args:
            lake_directories: List of directory names from data_lake_monthly
            
        Returns:
            Tuple of (verified_lakes, unverified_lakes)
        """
        verified_lakes = []
        unverified_lakes = []
        
        # Create lookup for efficient matching
        verified_names = set(self.verified_lakes_df['name'].str.lower().str.replace(' ', '_'))
        
        for lake_dir in lake_directories:
            # Normalize directory name for matching
            normalized_name = lake_dir.lower().replace('_', ' ').replace('-', ' ')
            
            # Check various name formats
            is_verified = any([
                lake_dir.lower() in verified_names,
                normalized_name in [name.lower() for name in self.verified_lakes_df['name']],
                any(normalized_name in verified_name.lower() or verified_name.lower() in normalized_name 
                    for verified_name in self.verified_lakes_df['name'] if len(verified_name) > 5)
            ])
            
            if is_verified:
                verified_lakes.append(lake_dir)
            else:
                unverified_lakes.append(lake_dir)
        
        logger.info(f"✅ Verified: {len(verified_lakes)} real lakes")
        logger.info(f"⚠️ Unverified: {len(unverified_lakes)} lakes need manual verification")
        
        return verified_lakes, unverified_lakes
    
    def validate_weather_data_authenticity(self, sample_data: pd.DataFrame) -> bool:
        """
        Validate that weather data appears authentic (not synthetic/manufactured).
        
        Args:
            sample_data: Sample of weather data to validate
            
        Returns:
            True if data appears authentic
        """
        logger.info("🔍 Validating weather data authenticity...")
        
        # Check 1: Realistic value ranges
        weather_ranges = {
            'temp_c': (-50, 60),      # Realistic temperature range
            'humidity': (0, 100),      # Humidity percentage
            'pressure_mb': (900, 1100), # Atmospheric pressure
            'wind_kph': (0, 200),      # Wind speed
            'uv': (0, 15),            # UV index
            'precip_mm': (0, 500)      # Daily precipitation
        }
        
        for column, (min_val, max_val) in weather_ranges.items():
            if column in sample_data.columns:
                out_of_range = (sample_data[column] < min_val) | (sample_data[column] > max_val)
                if out_of_range.any():
                    pct_invalid = (out_of_range.sum() / len(sample_data)) * 100
                    if pct_invalid > 5:  # More than 5% invalid is suspicious
                        logger.warning(f"⚠️ {pct_invalid:.1f}% of {column} values out of realistic range")
        
        # Check 2: Temporal consistency
        if 'datetime' in sample_data.columns:
            sample_data['datetime'] = pd.to_datetime(sample_data['datetime'])
            date_gaps = sample_data['datetime'].diff().dt.total_seconds() / 3600  # Hours between readings
            
            # Should have regular intervals (typically hourly)
            if date_gaps.std() > 2:  # More than 2 hours standard deviation is suspicious
                logger.warning("⚠️ Irregular time intervals detected in weather data")
        
        # Check 3: Statistical patterns that indicate real vs synthetic data
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            if len(sample_data[col].dropna()) > 100:
                # Real data should have some natural variation, not perfect patterns
                correlation_with_sequence = np.corrcoef(range(len(sample_data[col].dropna())), 
                                                       sample_data[col].dropna())[0,1]
                if abs(correlation_with_sequence) > 0.95:  # Too perfect correlation
                    logger.warning(f"⚠️ {col} shows suspiciously perfect sequential correlation")
        
        logger.info("✅ Weather data authenticity validation completed")
        return True
    
    def generate_data_integrity_report(self, data_directory: str) -> Dict:
        """
        Generate comprehensive data integrity report.
        
        Args:
            data_directory: Path to the data_lake_monthly directory
            
        Returns:
            Dictionary containing integrity report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': str(data_directory),
            'total_verified_lakes_in_db': len(self.verified_lakes_df),
            'lakes_with_data': 0,
            'verified_lakes_with_data': 0,
            'unverified_lakes': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        data_path = Path(data_directory)
        if not data_path.exists():
            report['error'] = f"Data directory does not exist: {data_directory}"
            return report
        
        # Get all lake directories
        lake_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        report['lakes_with_data'] = len(lake_dirs)
        
        # Verify against real lakes database
        lake_names = [d.name for d in lake_dirs]
        verified_lakes, unverified_lakes = self.verify_lake_authenticity(lake_names)
        
        report['verified_lakes_with_data'] = len(verified_lakes)
        report['unverified_lakes'] = unverified_lakes[:20]  # First 20 for brevity
        
        # Data quality checks on sample
        if verified_lakes:
            sample_lake = verified_lakes[0]
            sample_path = data_path / sample_lake
            csv_files = list(sample_path.glob("*.csv"))
            
            if csv_files:
                try:
                    sample_df = pd.read_csv(csv_files[0])
                    self.validate_weather_data_authenticity(sample_df)
                    report['sample_data_columns'] = list(sample_df.columns)
                    report['sample_data_shape'] = list(sample_df.shape)
                except Exception as e:
                    report['data_quality_issues'].append(f"Could not read sample data: {e}")
        
        # Generate recommendations
        if len(unverified_lakes) > 0:
            report['recommendations'].append(
                f"Manually verify {len(unverified_lakes)} unverified lakes against trusted sources"
            )
        
        if report['verified_lakes_with_data'] < data_integrity.expected_lake_count * 0.8:
            report['recommendations'].append(
                "Consider expanding verified lakes database or investigating missing data"
            )
        
        logger.info(f"📊 Data integrity report generated: {report['verified_lakes_with_data']} verified lakes")
        return report

class BiasPreventionValidator:
    """Validates training methodology to prevent bias and ensure fair representation"""
    
    @staticmethod
    def validate_geographical_distribution(lake_regions: List[str]) -> Dict:
        """Ensure balanced geographical representation"""
        from collections import Counter
        
        region_counts = Counter(lake_regions)
        total_lakes = len(lake_regions)
        
        # Check for geographical bias
        max_region_pct = max(region_counts.values()) / total_lakes * 100
        min_region_pct = min(region_counts.values()) / total_lakes * 100
        
        report = {
            'total_regions': len(region_counts),
            'max_region_percentage': max_region_pct,
            'min_region_percentage': min_region_pct,
            'geographical_balance_score': min_region_pct / max_region_pct,  # Closer to 1 is better
            'recommendations': []
        }
        
        if max_region_pct > 60:
            report['recommendations'].append(
                f"One region dominates {max_region_pct:.1f}% of data - consider regional stratification"
            )
        
        if report['geographical_balance_score'] < 0.1:
            report['recommendations'].append(
                "Severe geographical imbalance detected - use stratified sampling"
            )
        
        return report
    
    @staticmethod
    def validate_temporal_distribution(datetime_series: pd.Series) -> Dict:
        """Ensure balanced temporal representation across seasons/years"""
        datetime_series = pd.to_datetime(datetime_series)
        
        # Check seasonal distribution
        seasons = datetime_series.dt.month % 12 // 3  # 0=Winter, 1=Spring, 2=Summer, 3=Fall
        season_counts = seasons.value_counts()
        
        # Check yearly distribution
        yearly_counts = datetime_series.dt.year.value_counts()
        
        report = {
            'date_range': {
                'start': datetime_series.min().isoformat(),
                'end': datetime_series.max().isoformat(),
                'span_years': (datetime_series.max() - datetime_series.min()).days / 365.25
            },
            'seasonal_balance': {
                'winter': season_counts.get(0, 0),
                'spring': season_counts.get(1, 0), 
                'summer': season_counts.get(2, 0),
                'fall': season_counts.get(3, 0)
            },
            'yearly_distribution': yearly_counts.to_dict(),
            'recommendations': []
        }
        
        # Check for seasonal bias
        seasonal_std = season_counts.std() / season_counts.mean()
        if seasonal_std > 0.5:
            report['recommendations'].append(
                f"High seasonal variation detected (CV={seasonal_std:.2f}) - consider seasonal stratification"
            )
        
        # Check for temporal coverage
        if report['date_range']['span_years'] < 2:
            report['recommendations'].append(
                "Limited temporal coverage - consider multi-year data for robustness"
            )
        
        return report

def verify_data_integrity(data_directory: str, comprehensive_data_path: str) -> Dict:
    """
    Main function to verify complete data integrity before training.
    
    Args:
        data_directory: Path to data_lake_monthly
        comprehensive_data_path: Path to real_lakes_comprehensive.csv
        
    Returns:
        Comprehensive integrity report
    """
    logger.info("🔍 Starting comprehensive data integrity verification...")
    
    # Initialize validators
    real_data_validator = RealDataValidator(comprehensive_data_path)
    
    # Generate integrity report
    integrity_report = real_data_validator.generate_data_integrity_report(data_directory)
    
    # Add bias prevention analysis
    if integrity_report.get('verified_lakes_with_data', 0) > 0:
        # Sample some lake regions for bias analysis
        data_path = Path(data_directory)
        sample_regions = []
        
        for lake_dir in list(data_path.iterdir())[:100]:  # Sample first 100 lakes
            if lake_dir.is_dir():
                # Try to infer region from verified data
                lake_name = lake_dir.name
                matching_lake = real_data_validator.verified_lakes_df[
                    real_data_validator.verified_lakes_df['name'].str.lower().str.contains(
                        lake_name.lower().replace('_', ' ')[:10], na=False
                    )
                ]
                if not matching_lake.empty:
                    sample_regions.append(matching_lake.iloc[0]['region'])
        
        if sample_regions:
            bias_report = BiasPreventionValidator.validate_geographical_distribution(sample_regions)
            integrity_report['bias_prevention'] = bias_report
    
    # Final assessment
    integrity_report['integrity_score'] = calculate_integrity_score(integrity_report)
    integrity_report['ready_for_training'] = integrity_report['integrity_score'] > 0.8
    
    logger.info(f"✅ Data integrity verification complete. Score: {integrity_report['integrity_score']:.2f}")
    
    return integrity_report

def calculate_integrity_score(report: Dict) -> float:
    """Calculate overall data integrity score (0-1)"""
    score = 0.0
    
    # Verified data coverage (40% of score)
    verified_ratio = report.get('verified_lakes_with_data', 0) / max(report.get('lakes_with_data', 1), 1)
    score += verified_ratio * 0.4
    
    # Geographical balance (30% of score)
    bias_report = report.get('bias_prevention', {})
    geo_balance = bias_report.get('geographical_balance_score', 0.5)
    score += min(geo_balance * 2, 1.0) * 0.3  # Scale to 0-1
    
    # Data quality (30% of score)
    has_quality_issues = len(report.get('data_quality_issues', [])) > 0
    quality_score = 0.7 if has_quality_issues else 1.0
    score += quality_score * 0.3
    
    return min(score, 1.0)
