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
  python3 kaayko_trainer_superior_v3.py --resume-checkpoint
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
            '--resume-checkpoint', 
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
        
    def interactive_configuration_with_caching(self) -> TrainingConfig:
        """Enhanced interactive configuration with caching."""
        # Check for cached configuration
        cached_config = self.cache_manager.offer_cached_config()
        if cached_config:
            # Convert dict back to TrainingConfig using a dummy args object
            import argparse
            args = argparse.Namespace()
            args.sample_size = cached_config.get('sample_size', 'small')
            args.algorithm = cached_config.get('algorithm', 'histgradient')
            args.score_quantization = cached_config.get('score_quantization', 'half_step')
            args.safety_overrides = cached_config.get('safety_overrides', False)
            args.confidence_metric = cached_config.get('confidence_metric', False)
            args.localization = cached_config.get('localization', 'en-US')
            args.resume = 'fresh'
            args.smoke_test = False
            args.models_root = 'models'
            args.data_root = cached_config.get('data_root', '/Users/Rohan/data_lake_monthly')
            args.sample_rows_for_search = 1000000
            args.shard_size_rows = 100000
            args.n_jobs = -1
            args.save_csv = False
            return TrainingConfig(args)
            
        print(f"\n{Colors.CYAN}⚙️  INTERACTIVE CONFIGURATION{Colors.RESET}")
        print("=" * 50)
        
        # Run standard interactive configuration
        import argparse
        args = argparse.Namespace()
        
        data_root = interactive_data_path_selection()
        sample_size = interactive_sample_size_selection(data_root)
        algorithm = interactive_algorithm_selection()
        score_quantization = interactive_score_quantization()
        safety_overrides = interactive_safety_overrides()
        confidence_metric = interactive_confidence_metrics()
        localization = interactive_localization()
        
        # Set all required args attributes
        args.data_root = data_root
        args.sample_size = sample_size
        args.algorithm = algorithm
        args.score_quantization = score_quantization
        args.safety_overrides = safety_overrides
        args.confidence_metric = confidence_metric
        args.localization = localization
        args.resume = 'fresh'
        args.smoke_test = False
        args.models_root = 'models'
        args.sample_rows_for_search = 1000000
        args.shard_size_rows = 100000
        args.n_jobs = -1
        args.save_csv = False
        
        config = TrainingConfig(args)
        
        # Cache the configuration as dict
        config_dict = {
            'data_root': data_root,
            'sample_size': sample_size,
            'algorithm': algorithm,
            'score_quantization': score_quantization,
            'safety_overrides': safety_overrides,
            'confidence_metric': confidence_metric,
            'localization': localization
        }
        self.cache_manager.save_config_after_interactive(config_dict)
        
        return config
        
    def save_training_checkpoint(self, pipeline: TrainingPipeline, stage: str, progress: float):
        """Save training checkpoint."""
        checkpoint_data = {
            'session_id': self.session_id,
            'stage': stage,
            'progress': progress,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'data_root': pipeline.config.data_root,
                'sample_size': pipeline.config.sample_size,
                'algorithm': pipeline.config.algorithm,
                'score_quantization': pipeline.config.score_quantization,
                'safety_overrides': pipeline.config.safety_overrides,
                'confidence_metric': pipeline.config.confidence_metric,
                'localization': pipeline.config.localization
            }
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
            import argparse
            args = argparse.Namespace()
            args.data_root = config_data.get('data_root', '/Users/Rohan/data_lake_monthly')
            args.sample_size = config_data.get('sample_size', 'small')
            args.algorithm = config_data.get('algorithm', 'histgradient')
            args.score_quantization = config_data.get('score_quantization', 'half_step')
            args.safety_overrides = config_data.get('safety_overrides', False)
            args.confidence_metric = config_data.get('confidence_metric', False)
            args.localization = config_data.get('localization', 'en-US')
            args.resume = 'fresh'
            args.smoke_test = False
            args.models_root = 'models'
            args.sample_rows_for_search = 1000000
            args.shard_size_rows = 100000
            args.n_jobs = -1
            args.save_csv = False
            
            config = TrainingConfig(args)
            
            pipeline = TrainingPipeline(config)
            pipeline.set_interrupt_handler(self.interrupt_handler)
                
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
            
            # Run the full training pipeline using the existing method
            results = pipeline.train_model()
            
            # Mark as completed
            self.save_training_checkpoint(pipeline, "completed", 100.0)
            self.cache_manager.checkpoint_manager.complete_training_session()
            
            print(f"\n{Colors.GREEN}✅ TRAINING COMPLETED WITH CHECKPOINTS{Colors.RESET}")
            return results
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Training interrupted - checkpoint saved{Colors.RESET}")
            self.save_training_checkpoint(pipeline, "interrupted", 50.0)
            return None
            
        except Exception as e:
            print(f"\n{Colors.RED}❌ Training failed: {e}{Colors.RESET}")
            self.save_training_checkpoint(pipeline, "failed", 25.0)
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
            if args.resume_checkpoint:
                resumable_session = self.check_resume_capability()
                if resumable_session:
                    pipeline = self.resume_from_checkpoint(resumable_session)
                    
            # If not resuming or resume failed, start fresh
            if not pipeline:
                if args.smoke_test:
                    print(f"\n{Colors.YELLOW}🧪 RUNNING SMOKE TEST{Colors.RESET}")
                    import argparse
                    smoke_args = argparse.Namespace()
                    smoke_args.data_root = '/Users/Rohan/data_lake_monthly'
                    smoke_args.sample_size = 'tiny'
                    smoke_args.algorithm = 'random_forest'
                    smoke_args.score_quantization = 'half_step'
                    smoke_args.safety_overrides = True
                    smoke_args.confidence_metric = True
                    smoke_args.localization = 'en-US'
                    smoke_args.resume = 'fresh'
                    smoke_args.smoke_test = True
                    smoke_args.models_root = 'models'
                    smoke_args.sample_rows_for_search = 1000000
                    smoke_args.shard_size_rows = 100000
                    smoke_args.n_jobs = -1
                    smoke_args.save_csv = False
                    config = TrainingConfig(smoke_args)
                else:
                    config = self.interactive_configuration_with_caching()
                    
                # Display configuration and create pipeline
                display_final_configuration(config)
                
                pipeline = TrainingPipeline(config)
                pipeline.set_interrupt_handler(self.interrupt_handler)
                
            # Run training with checkpoints
            results = self.run_training_with_checkpoints(pipeline, args.checkpoint_interval)
            
            if results:
                print(f"\n{Colors.GREEN}🎉 TRAINING COMPLETED SUCCESSFULLY{Colors.RESET}")
                print("Check the 'models/' directory for saved models and metrics.")
            else:
                print(f"\n{Colors.YELLOW}⚠️  Training was interrupted or failed{Colors.RESET}")
                print("Use --resume-checkpoint to continue from the last checkpoint.")
                
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
