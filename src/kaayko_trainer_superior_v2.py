#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaayko Superior Trainer v2.0 - Main Orchestrator
================================================

🎯 MAIN ENTRY POINT:
• Command-line interface and argument parsing
• Interactive user interface coordination
• Training pipeline orchestration
• Results presentation and reporting
• Error handling and logging coordination

🚀 USAGE:
  python3 kaayko_trainer_superior_v2.py --sample-size small
  python3 kaayko_trainer_superior_v2.py --sample-size large --algorithm histgradient
  python3 kaayko_trainer_superior_v2.py --sample-size complete --algorithm ensemble
  python3 kaayko_trainer_superior_v2.py --smoke_test

📊 FEATURES:
• Professional modular architecture
• Comprehensive error handling and recovery
• Interactive and CLI modes
• Progress tracking and logging
• Model performance reporting

Author: Kaayko Intelligence Team
Version: 2.0
License: Proprietary
"""

import sys
import json
from datetime import datetime
from pathlib import Path

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

# ============================================================================
# MAIN ORCHESTRATOR CLASS
# ============================================================================

class KaaykoTrainerOrchestrator:
    """Main orchestrator for the Kaayko Superior Trainer v2.0."""
    
    def __init__(self):
        self.logger = None  # Will be initialized after config is available
        self.interrupt_handler = InterruptHandler()
        
    def display_startup_banner(self, config: TrainingConfig) -> None:
        """Display professional startup banner."""
        print_header("🚀 KAAYKO SUPERIOR TRAINER V2.0")
        
        # Get actual dataset info from data processor
        pipeline = TrainingPipeline(config)
        estimated_total, total_lakes = pipeline.data_processor.estimate_dataset_size()
        sample_multiplier = SAMPLE_CONFIGS[config.sample_size]['percentage'] / 100
        target_samples = int(estimated_total * sample_multiplier)
        time_estimate = SAMPLE_CONFIGS[config.sample_size]['time_estimate']
        
        print(f"📊 Target: {target_samples:,} samples from massive dataset")
        print(f"🌍 Data Source: {mask_path(str(config.data_root))}")
        print(f"💾 Output Dataset: kaayko_training_dataset.parquet (Parquet) + kaayko_training_dataset.csv (CSV)")
        
        if not config.smoke_test:
            print(f"🔍 Estimating total dataset size...")
            print(f"📊 Estimated dataset size: {estimated_total:,} samples across {total_lakes:,} lakes")
            print(f"📊 Sample size calculation:")
            print(f"  • Total estimated samples: {estimated_total:,}")
            print(f"  • Selected: {config.sample_size} ({sample_multiplier:.1%} of dataset)")
            print(f"  • Target samples: {target_samples:,}")
            print(f"  • Estimated training time: {time_estimate}")
        
        # Log configuration
        self.logger.info("=== Kaayko Superior Trainer v2.0 ===")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Sample size: {config.sample_size} ({sample_multiplier:.1%} = {target_samples:,} target samples)")
        self.logger.info(f"Resume mode: {config.resume}")
        self.logger.info(f"Sample rows for search: {config.sample_rows_for_search:,}")
        self.logger.info(f"Shard size: {config.shard_size_rows:,}")
        self.logger.info(f"Models root: {mask_path(str(config.models_root))}")
        self.logger.info(f"Data root: {mask_path(str(config.data_root))}")
        
        # Acknowledge non-wired flags
        unimplemented_flags = []
        if config.resume != 'fresh':
            unimplemented_flags.append(f"--resume={config.resume}")
        if config.confidence_metric:
            unimplemented_flags.append("--confidence_metric")
        if config.telemetry:
            unimplemented_flags.append("--telemetry")
        
        if unimplemented_flags:
            print(f"{Colors.YELLOW}⚠️  Accepted flags not yet active: {', '.join(unimplemented_flags)}{Colors.RESET}")
    
    def display_training_configuration(self, config: TrainingConfig) -> None:
        """Display training configuration summary."""
        print_header("🚀 KAAYKO SUPERIOR TRAINER V2.0")
        print(f"📊 Target: 50,000,000 samples from massive dataset")
        print(f"🌍 Data Source: {mask_path(str(config.data_root))}")
        print(f"💾 Output Dataset: kaayko_training_dataset.parquet (Parquet) + kaayko_training_dataset.csv (CSV)")
        print(f"🎯 Training Configuration:")
        print(f"  • Algorithm: {config.algorithm.upper()}")
        print(f"  • Score Quantization: {config.score_quantization.replace('_', ' ').title()}")
        print(f"  • Safety Overrides: {'Enabled' if config.safety_overrides else 'Disabled'}")
        print(f"  • Confidence Metrics: {'Enabled' if config.confidence_metric else 'Disabled'}")
        print(f"  • Localization: {config.localization}")
    
    def interactive_mode(self, args) -> TrainingConfig:
        """Run interactive configuration mode."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 KAAYKO INTERACTIVE CONFIGURATION MODE{Colors.RESET}")
        print("="*70)
        print("Configure your training session with guided prompts...")
        
        # Interactive selections - DATA PATH FIRST!
        data_path = interactive_data_path_selection()
        sample_size = interactive_sample_size_selection(data_path)
        algorithm = interactive_algorithm_selection()
        score_quantization = interactive_score_quantization()
        safety_overrides = interactive_safety_overrides()
        confidence_metric = interactive_confidence_metrics()
        localization = interactive_localization()
        
        # Create config object
        args.data_root = data_path  # Override with user-selected path
        args.sample_size = sample_size
        args.algorithm = algorithm
        args.score_quantization = score_quantization
        args.safety_overrides = safety_overrides
        args.confidence_metric = confidence_metric
        args.localization = localization
        
        config = TrainingConfig(args)
        
        # Initialize logging with proper base directory
        if self.logger is None:
            self.logger = setup_logging(base_dir=config.models_root)
        
        # Final confirmation
        if display_final_configuration(config):
            return config
        else:
            print(f"\n{Colors.YELLOW}🚫 Training cancelled by user{Colors.RESET}")
            sys.exit(0)
    
    def cli_mode(self, args) -> TrainingConfig:
        """Run CLI configuration mode."""
        # Check if data path exists, if not ask user
        data_path = Path(args.data_root)
        if not data_path.exists():
            print(f"\n{Colors.YELLOW}⚠️ Data path does not exist: {mask_path(str(data_path))}{Colors.RESET}")
            print(f"{Colors.CYAN}Please specify the correct data path:{Colors.RESET}")
            args.data_root = interactive_data_path_selection()
        
        config = TrainingConfig(args)
        
        # Initialize logging with proper base directory
        if self.logger is None:
            self.logger = setup_logging(base_dir=config.models_root)
        config.validate()
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 KAAYKO CLI CONFIGURATION MODE{Colors.RESET}")
        print("="*70)
        print("Using command-line arguments...")
        
        # Display configuration
        print(f"📂 Data Source: {mask_path(str(config.data_root))}")
        print(f"✨ Sample Size: {config.sample_size}")
        print(f"🤖 Algorithm: {config.algorithm}")
        print(f"📊 Quantization: {config.score_quantization}")
        print(f"🛡️ Safety Overrides: {'Enabled' if config.safety_overrides else 'Disabled'}")
        print(f"📈 Confidence Metrics: {'Enabled' if config.confidence_metric else 'Disabled'}")
        print(f"🌍 Localization: {config.localization}")
        
        return config
    
    def display_results(self, results: dict, config: TrainingConfig) -> None:
        """Display training results professionally."""
        if results['status'] == 'success':
            metrics = results['metrics']
            
            print_header("🎉 TRAINING COMPLETED SUCCESSFULLY")
            
            # Performance metrics
            print(f"\n{Colors.CYAN}📊 FINAL MODEL PERFORMANCE{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            print(f"{Colors.GREEN}R² Score: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}{Colors.RESET}")
            print(f"{Colors.GREEN}MAE Score: {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}{Colors.RESET}")
            
            # Algorithm performance comparison
            print(f"\n{Colors.CYAN}🏆 ALGORITHM PERFORMANCE{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            if config.algorithm == 'ensemble':
                print(f"{Colors.YELLOW}Ensemble Performance: {metrics['cv_r2_mean']:.2%} R²{Colors.RESET}")
                print(f"{Colors.GREEN}✨ Ensemble combines all algorithms for optimal performance{Colors.RESET}")
            else:
                algo_config = ALGORITHM_CONFIGS[config.algorithm]
                print(f"{Colors.YELLOW}{algo_config['emoji']} {config.algorithm.title()}: {metrics['cv_r2_mean']:.2%} R²{Colors.RESET}")
            
            # Configuration summary
            print(f"\n{Colors.CYAN}⚙️ TRAINING CONFIGURATION{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            print(f"Sample Size: {config.sample_size}")
            print(f"Algorithm: {config.algorithm}")
            print(f"Score Quantization: {config.score_quantization}")
            print(f"Safety Overrides: {'✅ Enabled' if config.safety_overrides else '❌ Disabled'}")
            print(f"Confidence Metrics: {'✅ Enabled' if config.confidence_metric else '❌ Disabled'}")
            print(f"Localization: {config.localization}")
            
            # Model information
            print(f"\n{Colors.CYAN}💾 MODEL ARTIFACTS{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            print(f"Model saved to: {mask_path(results['model_path'])}")
            print(f"Training dataset: kaayko_training_dataset.parquet")
            print(f"Training logs: kaayko_training.log")
            
            # Next steps
            print(f"\n{Colors.CYAN}🚀 NEXT STEPS{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            print("1. Test model with real-world predictions")
            print("2. Deploy model to production API")
            print("3. Monitor model performance metrics")
            print("4. Collect feedback for model improvements")
            
        elif results['status'] == 'interrupted':
            print_header("⚠️ TRAINING INTERRUPTED")
            print("🔄 Training was interrupted before completion.")
            print("💾 Partial dataset and progress may have been saved.")
            print("\n🚀 To resume training, run:")
            print("   python3 kaayko_trainer_superior_v2.py --resume append")
            
        else:
            print_header("❌ TRAINING FAILED")
            print(f"{Colors.RED}Error: {results.get('error', 'Unknown error')}{Colors.RESET}")
            print("\n🔍 Check the logs for detailed error information:")
            print("   tail -f kaayko_training.log")
            print("\n🚀 Try running with smaller sample size:")
            print("   python3 kaayko_trainer_superior_v2.py --sample-size small")
    
    def save_training_summary(self, results: dict, config: TrainingConfig) -> None:
        """Save training summary to file."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'config': config.to_dict(),
            'results': results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        summary_file = f"kaayko_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n📋 Training summary saved to: {summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save training summary: {str(e)}")
    
    def run(self) -> None:
        """Main execution method."""
        try:
            # Parse arguments
            parser = create_argument_parser()
            args = parser.parse_args()
            
            # Determine configuration mode
            has_explicit_args = any([
                args.algorithm != 'ensemble',
                args.sample_size != 'medium',
                args.score_quantization != 'half_step',
                args.safety_overrides,
                args.confidence_metric,
                args.localization != 'en-US',
                args.smoke_test
            ])
            
            if has_explicit_args:
                # CLI mode - use provided arguments
                config = self.cli_mode(args)
            else:
                # Interactive mode - prompt for configuration
                config = self.interactive_mode(args)
            
            # Display startup information
            self.display_startup_banner(config)
            self.display_training_configuration(config)
            
            # Create and run training pipeline
            pipeline = TrainingPipeline(config)
            pipeline.set_interrupt_handler(self.interrupt_handler)  # Proper interrupt connection
            results = pipeline.train_model()
            
            # Handle interruption
            if self.interrupt_handler.interrupted:
                results['status'] = 'interrupted'
            
            # Display results
            self.display_results(results, config)
            
            # Save summary
            self.save_training_summary(results, config)
            
            # Exit with appropriate code
            if results['status'] == 'success':
                sys.exit(0)
            elif results['status'] == 'interrupted':
                sys.exit(130)  # SIGINT exit code
            else:
                sys.exit(1)
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️ Training cancelled by user{Colors.RESET}")
            sys.exit(130)
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}", exc_info=True)
            print(f"\n{Colors.RED}❌ Fatal error: {str(e)}{Colors.RESET}")
            print(f"{Colors.RED}Check kaayko_training.log for detailed information{Colors.RESET}")
            sys.exit(1)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the application."""
    orchestrator = KaaykoTrainerOrchestrator()
    orchestrator.run()

if __name__ == '__main__':
    main()
