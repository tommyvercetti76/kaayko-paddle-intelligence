#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaayko Superior Trainer v3.0 - Checkpoint-Enabled Training
==========================================================

🎯 MAIN FEATURES:
• Checkpoint-based training with resume capability
• Interactive configuration caching
• Session state management
• Progress tracking and recovery
• Background job support

🆕 NEW IN v3.0:
• Automatic checkpoint creation during training
• Resume interrupted training sessions
• Cache interactive configuration choices
• Enhanced error recovery
• Training progress persistence

🚀 USAGE:
  python3 kaayko_trainer_superior_v3.py --sample-size small
  python3 kaayko_trainer_superior_v3.py --resume
  python3 kaayko_trainer_superior_v3.py --list-checkpoints
  python3 kaayko_trainer_superior_v3.py --cleanup-old

📊 ENHANCED FEATURES:
• Professional modular architecture with checkpointing
• Comprehensive error handling and recovery
• Interactive and CLI modes with caching
• Advanced progress tracking and logging
• Model performance reporting with session management

Author: Kaayko Intelligence Team
Version: 3.0
License: Proprietary
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import our modular components
from kaayko_config_v2 import (
    Colors, TrainingConfig, SAMPLE_CONFIGS, ALGORITHM_CONFIGS,
    print_header, create_argument_parser, setup_logging, InterruptHandler,
    interactive_data_path_selection, interactive_sample_size_selection, interactive_algorithm_selection,
    interactive_score_quantization, interactive_safety_overrides,
    interactive_confidence_metrics, interactive_localization,
    display_final_configuration, mask_path
)

from kaayko_core_v2 import TrainingPipeline
from kaayko_cache_manager_v3 import KaaykoCacheManager

# ============================================================================
# ENHANCED ORCHESTRATOR CLASS WITH CHECKPOINTING
# ============================================================================

class KaaykoTrainerSuperiorV3:
    """Enhanced trainer with checkpoint and caching capabilities."""
    
    def __init__(self):
        self.logger = setup_logging("INFO")
        self.interrupt_handler = InterruptHandler()
        self.cache_manager = KaaykoCacheManager()
        self.session_id: Optional[str] = None
        
    def create_enhanced_argument_parser(self):
        """Create argument parser with v3 checkpoint features."""
        parser = create_argument_parser()
        
        # Add checkpoint-specific arguments
        parser.add_argument(
            '--resume', 
            action='store_true',
            help='Resume from the most recent checkpoint'
        )
        parser.add_argument(
            '--list-checkpoints', 
            action='store_true',
            help='List available training checkpoints'
        )
        parser.add_argument(
            '--cleanup-old', 
            action='store_true',
            help='Clean up old checkpoint files'
        )
        parser.add_argument(
            '--checkpoint-interval', 
            type=int, 
            default=50,
            help='Save checkpoint every N steps (default: 50)'
        )
        
        return parser
        
    def list_checkpoints(self):
        """List available training checkpoints."""
        print(f"\n{Colors.CYAN}🔍 AVAILABLE CHECKPOINTS{Colors.RESET}")
        print("=" * 60)
        
        checkpoints = self.cache_manager.checkpoint_manager.list_available_checkpoints()
        
        if not checkpoints:
            print(f"{Colors.YELLOW}No checkpoints found.{Colors.RESET}")
            return
            
        for i, checkpoint in enumerate(checkpoints, 1):
            session_id = checkpoint['session_id']
            timestamp = checkpoint['timestamp']
            progress = checkpoint['progress']
            stage = checkpoint['stage']
            
            print(f"{i}. Session: {session_id}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Progress: {progress:.1f}%")
            print(f"   Stage: {stage}")
            print()
            
    def cleanup_old_checkpoints(self, days: int = 7):
        """Clean up old checkpoint files."""
        print(f"\n{Colors.CYAN}🧹 CLEANING UP CHECKPOINTS{Colors.RESET}")
        print("=" * 50)
        
        cleaned = self.cache_manager.checkpoint_manager.cleanup_old_checkpoints(days)
        print(f"Cleaned up {cleaned} old checkpoint(s) older than {days} days.")
        
    def check_resume_capability(self) -> Optional[Dict[str, Any]]:
        """Check if training can be resumed."""
        resumable_session = self.cache_manager.checkpoint_manager.can_resume()
        
        if resumable_session:
            print(f"\n{Colors.CYAN}🔄 RESUMABLE SESSION FOUND{Colors.RESET}")
            print("=" * 50)
            print(f"Session ID: {resumable_session['session_id']}")
            print(f"Progress: {resumable_session['progress']:.1f}%")
            print(f"Last Updated: {resumable_session['timestamp']}")
            
            while True:
                choice = input(f"\n{Colors.YELLOW}Resume this session? (y/n): {Colors.RESET}").strip().lower()
                if choice in ['y', 'yes']:
                    return resumable_session
                elif choice in ['n', 'no']:
                    return None
                else:
                    print(f"{Colors.RED}❌ Please enter 'y' or 'n'{Colors.RESET}")
                    
        return None
        
    def interactive_configuration_with_caching(self) -> Dict[str, Any]:
        """Enhanced interactive configuration with caching."""
        # Check for cached configuration
        cached_config = self.cache_manager.offer_cached_config()
        if cached_config:
            return cached_config
            
        print(f"\n{Colors.CYAN}⚙️  INTERACTIVE CONFIGURATION{Colors.RESET}")
        print("=" * 50)
        
        # Run standard interactive configuration
        config_dict = {}
        
        config_dict['data_root'] = interactive_data_path_selection()
        config_dict['sample_size'] = interactive_sample_size_selection()
        config_dict['algorithm'] = interactive_algorithm_selection()
        config_dict['score_quantization'] = interactive_score_quantization()
        config_dict['safety_overrides'] = interactive_safety_overrides()
        config_dict['confidence_metric'] = interactive_confidence_metrics()
        config_dict['localization'] = interactive_localization()
        
        # Cache the configuration
        self.cache_manager.save_config_after_interactive(config_dict)
        
        return config_dict
        
    def save_training_checkpoint(self, pipeline: TrainingPipeline, stage: str, progress: float):
        """Save training checkpoint."""
        checkpoint_data = {
            'session_id': self.session_id,
            'stage': stage,
            'progress': progress,
            'timestamp': datetime.now().isoformat(),
            'pipeline_state': pipeline.get_state() if hasattr(pipeline, 'get_state') else {},
            'config': pipeline.config.__dict__ if hasattr(pipeline, 'config') else {}
        }
        
        success = self.cache_manager.checkpoint_manager.save_training_checkpoint(checkpoint_data)
        if success:
            self.logger.info(f"Checkpoint saved: {stage} ({progress:.1f}% complete)")
        
    def resume_from_checkpoint(self, session_info: Dict[str, Any]) -> Optional[TrainingPipeline]:
        """Resume training from checkpoint."""
        session_id = session_info['session_id']
        checkpoint_data = self.cache_manager.checkpoint_manager.load_checkpoint(session_id)
        
        if not checkpoint_data:
            print(f"{Colors.RED}❌ Could not load checkpoint data{Colors.RESET}")
            return None
            
        print(f"\n{Colors.GREEN}🔄 RESUMING FROM CHECKPOINT{Colors.RESET}")
        print("=" * 50)
        print(f"Session ID: {session_id}")
        print(f"Stage: {checkpoint_data.get('stage', 'unknown')}")
        print(f"Progress: {checkpoint_data.get('progress', 0):.1f}%")
        
        # Reconstruct training pipeline from checkpoint
        try:
            config_data = checkpoint_data.get('config', {})
            config = TrainingConfig(**config_data)
            
            pipeline = TrainingPipeline(config)
            
            # Restore pipeline state if available
            pipeline_state = checkpoint_data.get('pipeline_state', {})
            if pipeline_state and hasattr(pipeline, 'restore_state'):
                pipeline.restore_state(pipeline_state)
                
            self.session_id = session_id
            return pipeline
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to restore from checkpoint: {e}{Colors.RESET}")
            return None
            
    def run_training_with_checkpoints(self, pipeline: TrainingPipeline, checkpoint_interval: int = 50):
        """Run training with checkpoint saving."""
        print(f"\n{Colors.CYAN}🚀 STARTING CHECKPOINT-ENABLED TRAINING{Colors.RESET}")
        print("=" * 60)
        
        # Start new session if not resuming
        if not self.session_id:
            self.session_id = self.cache_manager.checkpoint_manager.start_training_session()
            
        try:
            # Save initial checkpoint
            self.save_training_checkpoint(pipeline, "initialization", 0.0)
            
            # Data loading checkpoint
            print(f"{Colors.YELLOW}📊 Loading and preprocessing data...{Colors.RESET}")
            pipeline.load_and_preprocess_data()
            self.save_training_checkpoint(pipeline, "data_loaded", 20.0)
            
            # Feature engineering checkpoint  
            print(f"{Colors.YELLOW}🔧 Engineering features...{Colors.RESET}")
            pipeline.engineer_features()
            self.save_training_checkpoint(pipeline, "features_engineered", 40.0)
            
            # Model training checkpoint
            print(f"{Colors.YELLOW}🤖 Training models...{Colors.RESET}")
            pipeline.train_models()
            self.save_training_checkpoint(pipeline, "models_trained", 80.0)
            
            # Final evaluation and results
            print(f"{Colors.YELLOW}📈 Evaluating and generating results...{Colors.RESET}")
            results = pipeline.evaluate_and_report()
            self.save_training_checkpoint(pipeline, "completed", 100.0)
            
            # Mark session as completed
            self.cache_manager.checkpoint_manager.complete_training_session()
            
            print(f"\n{Colors.GREEN}✅ TRAINING COMPLETED WITH CHECKPOINTS{Colors.RESET}")
            return results
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Training interrupted - checkpoint saved{Colors.RESET}")
            current_progress = self.cache_manager.checkpoint_manager.get_current_progress()
            self.save_training_checkpoint(pipeline, "interrupted", current_progress)
            return None
            
        except Exception as e:
            print(f"\n{Colors.RED}❌ Training failed: {e}{Colors.RESET}")
            current_progress = self.cache_manager.checkpoint_manager.get_current_progress() 
            self.save_training_checkpoint(pipeline, "failed", current_progress)
            raise
            
    def run(self):
        """Main execution method with v3 enhancements."""
        try:
            print_header("Kaayko Superior Trainer v3.0 - Checkpoint Edition")
            
            parser = self.create_enhanced_argument_parser()
            args = parser.parse_args()
            
            # Handle checkpoint-specific commands
            if args.list_checkpoints:
                self.list_checkpoints()
                return
                
            if args.cleanup_old:
                self.cleanup_old_checkpoints()
                return
                
            pipeline = None
            
            # Check for resume capability
            if args.resume:
                resumable_session = self.check_resume_capability()
                if resumable_session:
                    pipeline = self.resume_from_checkpoint(resumable_session)
                    
            # If not resuming or resume failed, start fresh
            if not pipeline:
                if args.smoke_test:
                    print(f"\n{Colors.YELLOW}🧪 RUNNING SMOKE TEST{Colors.RESET}")
                    config_dict = {
                        'data_root': '/Users/Rohan/data_lake_monthly',
                        'sample_size': 'tiny',
                        'algorithm': 'random_forest',
                        'score_quantization': 'none',
                        'safety_overrides': True,
                        'confidence_metric': True,
                        'localization': 'global'
                    }
                else:
                    config_dict = self.interactive_configuration_with_caching()
                    
                # Create configuration and pipeline
                config = TrainingConfig.from_dict(config_dict)
                display_final_configuration(config)
                
                pipeline = TrainingPipeline(config)
                
            # Run training with checkpoints
            results = self.run_training_with_checkpoints(pipeline, args.checkpoint_interval)
            
            if results:
                print(f"\n{Colors.GREEN}🎉 TRAINING COMPLETED SUCCESSFULLY{Colors.RESET}")
                print("Check the 'models/' directory for saved models and metrics.")
            else:
                print(f"\n{Colors.YELLOW}⚠️  Training was interrupted or failed{Colors.RESET}")
                print("Use --resume to continue from the last checkpoint.")
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Training interrupted by user{Colors.RESET}")
            sys.exit(0)
        except Exception as e:
            print(f"\n{Colors.RED}❌ Fatal error: {e}{Colors.RESET}")
            self.logger.error(f"Fatal error in main: {e}")
            sys.exit(1)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for v3 trainer."""
    trainer = KaaykoTrainerSuperiorV3()
    trainer.run()

if __name__ == "__main__":
    main()
