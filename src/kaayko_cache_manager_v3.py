#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaayko Cache Manager v3.0 - Configuration & Checkpoint Caching
==============================================================

🎯 RESPONSIBILITIES:
• Interactive configuration caching and restore
• Training checkpoint management and cleanup
• Session state persistence and recovery
• Background job queue management

🔄 MAXIMUM REUSE:
• Works with existing v2 TrainingConfig objects
• Compatible with v2 interactive functions
• Lightweight wrapper around proven v2 systems

Author: Kaayko Intelligence Team
Version: 3.0
License: Proprietary
"""

import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import logging

from kaayko_config_v2 import Colors, setup_logging


class ConfigurationCache:
    """Lightweight configuration caching for interactive sessions."""
    
    def __init__(self, cache_dir: Path = Path("checkpoints")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.config_file = self.cache_dir / "last_interactive_config.json"
        
    def save_interactive_config(self, config_dict: Dict[str, Any]) -> None:
        """Save interactive configuration choices."""
        config_dict['timestamp'] = datetime.now().isoformat()
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def load_interactive_config(self) -> Optional[Dict[str, Any]]:
        """Load cached interactive configuration."""
        if not self.config_file.exists():
            return None
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
            
    def clear_config_cache(self) -> None:
        """Clear cached configuration."""
        if self.config_file.exists():
            self.config_file.unlink()


class CheckpointManager:
    """Advanced checkpoint management system."""
    
    def __init__(self, checkpoint_dir: Path = Path("checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = setup_logging("INFO")
        
    def save_checkpoint(self, session_id: str, checkpoint_data: Dict[str, Any]) -> bool:
        """Save training checkpoint."""
        try:
            session_dir = self.checkpoint_dir / f"session_{session_id}"
            session_dir.mkdir(exist_ok=True)
            
            # Save JSON metadata
            metadata = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'stage': checkpoint_data.get('stage', 'unknown'),
                'progress': checkpoint_data.get('progress', 0.0)
            }
            
            with open(session_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save binary checkpoint data
            with open(session_dir / "checkpoint.pkl", 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False
            
    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load training checkpoint."""
        try:
            session_dir = self.checkpoint_dir / f"session_{session_id}"
            checkpoint_file = session_dir / "checkpoint.pkl"
            
            if not checkpoint_file.exists():
                return None
                
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load checkpoint {session_id}: {e}")
            return None
            
    def can_resume(self) -> Optional[Dict[str, Any]]:
        """Check if there's a resumable session."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
            
        # Find most recent incomplete session
        for session_id, metadata in checkpoints:
            if metadata.get('progress', 100) < 100:
                return {
                    'session_id': session_id,
                    'progress': metadata.get('progress', 0),
                    'timestamp': metadata.get('timestamp', 'unknown')
                }
        return None
        
    def start_training_session(self) -> str:
        """Start a new training session."""
        import uuid
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        return session_id
        
    def save_training_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Save training checkpoint with session ID."""
        session_id = checkpoint_data.get('session_id', self.start_training_session())
        return self.save_checkpoint(session_id, checkpoint_data)
        
    def complete_training_session(self) -> None:
        """Mark training session as completed."""
        # Clean up completed sessions older than 7 days
        self.cleanup_old_checkpoints(7)
        
    def get_current_progress(self) -> float:
        """Get current training progress."""
        return 0.0  # Placeholder - would track actual progress
        
    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints in simplified format."""
        checkpoints = self.list_checkpoints()
        return [
            {
                'session_id': session_id,
                'timestamp': metadata.get('timestamp', 'unknown'),
                'progress': metadata.get('progress', 0),
                'stage': metadata.get('stage', 'unknown')
            }
            for session_id, metadata in checkpoints
        ]
            
    def list_checkpoints(self) -> List[Tuple[str, Dict[str, Any]]]:
        """List all available checkpoints."""
        checkpoints = []
        for session_dir in self.checkpoint_dir.glob("session_*"):
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        checkpoints.append((session_dir.name, metadata))
                    except Exception:
                        continue
        
        # Sort by timestamp, newest first
        checkpoints.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
        return checkpoints
        
    def cleanup_old_checkpoints(self, days: int) -> int:
        """Clean up checkpoints older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        cleaned = 0
        
        for session_dir in self.checkpoint_dir.glob("session_*"):
            if session_dir.is_dir():
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        checkpoint_time = datetime.fromisoformat(metadata.get('timestamp', ''))
                        if checkpoint_time < cutoff:
                            import shutil
                            shutil.rmtree(session_dir)
                            cleaned += 1
                    except Exception:
                        continue
                        
        return cleaned


class TrainingJobQueue:
    """Background training job management."""
    
    def __init__(self):
        self.active_jobs: List[Dict[str, Any]] = []
        self.completed_jobs: List[Dict[str, Any]] = []
        self.failed_jobs: List[Dict[str, Any]] = []
        self.job_queue: List[Dict[str, Any]] = []
        
    def add_job(self, job_config: Dict[str, Any]) -> str:
        """Add job to queue."""
        job_id = f"job_{int(time.time())}"
        job_config['job_id'] = job_id
        job_config['created_at'] = datetime.now().isoformat()
        self.job_queue.append(job_config)
        return job_id
        
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status summary."""
        return {
            'active': len(self.active_jobs),
            'queued': len(self.job_queue),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs)
        }


class KaaykoCacheManager:
    """Unified cache management for v3 features."""
    
    def __init__(self, base_dir: Path = Path("checkpoints")):
        self.base_dir = base_dir
        self.config_cache = ConfigurationCache(base_dir)
        self.checkpoint_manager = CheckpointManager(base_dir)
        self.job_queue = TrainingJobQueue()
        
    def offer_cached_config(self) -> Optional[Dict[str, Any]]:
        """Check and offer to use cached interactive configuration."""
        cached_config = self.config_cache.load_interactive_config()
        
        if cached_config is None:
            return None
            
        print(f"\n{Colors.CYAN}🔄 PREVIOUS CONFIGURATION FOUND{Colors.RESET}")
        print("=" * 50)
        print(f"Data Source: {cached_config.get('data_root', 'N/A')}")
        print(f"Sample Size: {cached_config.get('sample_size', 'N/A')}")
        print(f"Algorithm: {cached_config.get('algorithm', 'N/A')}")
        print(f"Quantization: {cached_config.get('score_quantization', 'N/A')}")
        print(f"Safety Overrides: {'Yes' if cached_config.get('safety_overrides') else 'No'}")
        print(f"Confidence Metrics: {'Yes' if cached_config.get('confidence_metric') else 'No'}")
        print(f"Localization: {cached_config.get('localization', 'N/A')}")
        
        print(f"\n{Colors.YELLOW}Options:{Colors.RESET}")
        print("1. Use previous configuration")
        print("2. Start fresh configuration")
        
        while True:
            choice = input(f"\n{Colors.CYAN}Choose option (1-2): {Colors.RESET}").strip()
            if choice == "1":
                print(f"\n{Colors.GREEN}✅ Using previous configuration{Colors.RESET}")
                return cached_config
            elif choice == "2":
                return None  # Signal to run fresh interactive config
            else:
                print(f"{Colors.RED}❌ Invalid choice{Colors.RESET}")
                
    def save_config_after_interactive(self, config_dict: Dict[str, Any]) -> None:
        """Save configuration after successful interactive session."""
        self.config_cache.save_interactive_config(config_dict)
        print(f"\n{Colors.GREEN}💾 Configuration saved for future use{Colors.RESET}")
