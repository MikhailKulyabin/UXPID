import os
import sys
import argparse
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import ForumDataProcessor
from bert_trainer import BERTTrainer
from config import *
from utils import create_timestamp_folder, get_timestamp_paths, create_run_summary


def setup_logging(log_dir: str = None, timestamp: str = None):
    """Set up logging configuration."""
    if log_dir is None:
        log_dir = OUTPUT_CONFIG["logs_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["log_level"]),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def save_config(output_dir: str):
    """Save configuration to file for reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    
    config_data = {
        "MODEL_CONFIG": MODEL_CONFIG,
        "TRAINING_CONFIG": TRAINING_CONFIG,
        "DATA_CONFIG": DATA_CONFIG,
        "PREDICTION_CONFIG": PREDICTION_CONFIG,
        "OUTPUT_CONFIG": OUTPUT_CONFIG,
        "LOGGING_CONFIG": LOGGING_CONFIG,
        "HARDWARE_CONFIG": HARDWARE_CONFIG,
        "METRICS_CONFIG": METRICS_CONFIG,
        "PREPROCESSING_CONFIG": PREPROCESSING_CONFIG,
        "timestamp": datetime.now().isoformat()
    }
    
    config_file = os.path.join(output_dir, "training_config.json")
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    
    return config_file


def main():
    """Main training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BERT Multi-label Classification Training")
    parser.add_argument("--dataset-path", default=DATA_CONFIG["dataset_path"], 
                       help="Path to dataset directory")
    parser.add_argument("--output-dir", default=OUTPUT_CONFIG["results_dir"],
                       help="Directory to save results")
    parser.add_argument("--model-name", default=MODEL_CONFIG["model_name"],
                       help="Pre-trained BERT model name")
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"],
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["num_epochs"],
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=TRAINING_CONFIG["learning_rate"],
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=MODEL_CONFIG["max_length"],
                       help="Maximum sequence length")
    parser.add_argument("--skip-data-processing", action="store_true",
                       help="Skip data processing if processed data already exists")
    parser.add_argument("--data-analysis-only", action="store_true",
                       help="Only perform data analysis without training")
    parser.add_argument("--use-timestamp-folder", action="store_true", 
                       default=OUTPUT_CONFIG["use_timestamp_folders"],
                       help="Use timestamp-based folder structure for outputs")
    parser.add_argument("--text-field", choices=["text", "insight_summary"], 
                       default=DATA_CONFIG["text_field"],
                       help="Field to use for training: 'text' or 'insight_summary'")
    parser.add_argument("--target-field", choices=["topics", "branch_status", "branch_type", "overall_thread_sentiment"], 
                       default=DATA_CONFIG["target_field"],
                       help="Target field for classification: 'topics', 'branch_status', 'branch_type', or 'overall_thread_sentiment'")
    parser.add_argument("--data-split", action="store_true", default=DATA_CONFIG["data_split"],
                       help="Create new data split (train/test). If False, use existing processed data split")
    parser.add_argument("--use-official-split", action="store_true", default=False,
                       help="Use the official Zenodo train/test split from splits/ instead of a random split (requires --data-split)")
    parser.add_argument("--splits-dir", type=str, default=DATA_CONFIG.get("splits_dir", "splits"),
                       help="Path to directory containing train_branches.txt / test_branches.txt")
    
    # Continue training arguments with config defaults
    continue_group = parser.add_mutually_exclusive_group()
    continue_group.add_argument("--continue-training", action="store_true",
                               help="Continue training from checkpoint")
    continue_group.add_argument("--no-continue-training", action="store_true",
                               help="Don't continue training (override config)")
    
    parser.add_argument("--resume-from-checkpoint", type=str,
                       default=TRAINING_CONFIG["resume_from_checkpoint"],
                       help="Path to checkpoint to resume from (overrides config)")
    
    # Class weighting arguments
    parser.add_argument("--use-class-weights", action="store_true",
                       default=TRAINING_CONFIG["use_class_weights"],
                       help="Use class weighting for imbalanced data")
    parser.add_argument("--no-class-weights", action="store_true",
                       help="Disable class weighting (override config)")
    parser.add_argument("--class-weight-method", choices=["balanced", "inverse_freq", "custom"],
                       default=TRAINING_CONFIG["class_weight_method"],
                       help="Method for calculating class weights")
    
    args = parser.parse_args()
    
    # Create timestamp-based output structure
    if args.use_timestamp_folder:
        timestamp, output_base_path = create_timestamp_folder(OUTPUT_CONFIG["outputs_base_dir"])
        paths = get_timestamp_paths(output_base_path)
        
        # Override args.output_dir to use timestamp folder
        args.output_dir = paths["results"]
        log_dir = paths["logs"]
        plots_dir = paths["plots"]
        models_dir = paths["models"]
        
        print(f"Created timestamp folder: {output_base_path}")
        print(f"All outputs will be saved under: {output_base_path}")
    else:
        # Use original directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_path = args.output_dir
        log_dir = OUTPUT_CONFIG["logs_dir"]
        plots_dir = os.path.join(args.output_dir, OUTPUT_CONFIG["plots_dir"])
        models_dir = args.output_dir
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging with timestamp
    logger = setup_logging(log_dir, timestamp)
    logger.info("Starting BERT Multi-label Classification Training")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output base path: {output_base_path}")
    logger.info(f"Using text field for training: {args.text_field}")
    
    # Save configuration
    config_file = save_config(args.output_dir)
    logger.info(f"Configuration saved to: {config_file}")
    
    # Create run summary if using timestamp folders
    if args.use_timestamp_folder:
        config_data = {
            "MODEL_CONFIG": MODEL_CONFIG,
            "TRAINING_CONFIG": TRAINING_CONFIG,
            "DATA_CONFIG": DATA_CONFIG,
            "PREDICTION_CONFIG": PREDICTION_CONFIG,
            "OUTPUT_CONFIG": OUTPUT_CONFIG,
            "LOGGING_CONFIG": LOGGING_CONFIG,
            "HARDWARE_CONFIG": HARDWARE_CONFIG,
            "METRICS_CONFIG": METRICS_CONFIG,
            "PREPROCESSING_CONFIG": PREPROCESSING_CONFIG,
            "args": vars(args)
        }
        summary_file = create_run_summary(output_base_path, config_data, timestamp)
        logger.info(f"Run summary saved to: {summary_file}")
    
    try:
        # Step 1: Data Processing
        logger.info("="*50)
        logger.info("STEP 1: DATA PROCESSING")
        logger.info("="*50)
        
        processed_data_dir = DATA_CONFIG["processed_data_dir"]
        
        # Initialize data processor
        processor = ForumDataProcessor(args.dataset_path, 
                                     topics_file=TOPICS_CONFIG["topics_file"],
                                     target_field=args.target_field)
        
        # Check if we should skip data processing entirely
        if args.skip_data_processing and not args.data_split and os.path.exists(processed_data_dir):
            logger.info("Loading existing processed data (skip data processing enabled)...")
            try:
                train_df, test_df = processor.load_processed_data(processed_data_dir)
                logger.info("Existing processed data loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load existing data: {e}. Creating new data...")
                train_df, test_df = processor.get_or_create_training_data(
                    args.data_split,
                    splits_dir=args.splits_dir if args.use_official_split else None
                )
        else:
            # Load and process data for analysis and/or training data creation
            data = processor.load_data()
            if not data:
                logger.error("No data loaded. Please check the dataset path.")
                return
            
            df = processor.create_dataframe()
            logger.info(f"Created DataFrame with {len(df)} entries")
            
            # Perform data analysis
            analysis = processor.analyze_data()
            logger.info("Data Analysis Results:")
            for key, value in analysis.items():
                logger.info(f"  {key}: {value}")
            
            # Save analysis results
            analysis_file = os.path.join(args.output_dir, "data_analysis.json")
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Data analysis saved to: {analysis_file}")
            
            # Create visualizations
            try:
                os.makedirs(plots_dir, exist_ok=True)
                viz_file = os.path.join(plots_dir, "data_visualization.png")
                processor.visualize_data(viz_file)
                logger.info(f"Data visualization saved to: {viz_file}")
            except Exception as e:
                logger.warning(f"Could not create visualizations: {e}")
            
            # Get or create training data based on data_split parameter
            train_df, test_df = processor.get_or_create_training_data(
                args.data_split,
                splits_dir=args.splits_dir if args.use_official_split else None
            )
        
        # Check if we got training data or if we're in data_split=False mode
        if train_df is None and test_df is None:
            logger.info("No train/test split created or loaded (data_split=False)")
            if args.data_analysis_only:
                logger.info("Data analysis complete. Exiting as requested.")
                return
            else:
                logger.info("Cannot proceed with training without train/test split.")
                logger.info("To create training data, use --data-split flag or set data_split=True in config.py")
                return
        
        # Check if we should only do data analysis
        if args.data_analysis_only:
            logger.info("Data analysis complete. Exiting as requested.")
            return
        
        # Step 2: Model Training
        logger.info("="*50)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*50)
        
        # Initialize trainer
        resume_checkpoint = None
        
        # Determine continue training setting
        if args.continue_training:
            continue_training = True
        elif args.no_continue_training:
            continue_training = False
        else:
            continue_training = TRAINING_CONFIG["continue_training"]
        
        if continue_training:
            # Command line argument takes priority over config
            resume_checkpoint = args.resume_from_checkpoint or TRAINING_CONFIG["resume_from_checkpoint"]
            if resume_checkpoint:
                logger.info(f"Will resume training from checkpoint: {resume_checkpoint}")
            else:
                logger.warning("Continue training requested but no checkpoint path provided")
                continue_training = False
        
        # Determine class weighting settings
        if args.no_class_weights:
            use_class_weights = False
        elif args.use_class_weights:
            use_class_weights = True
        else:
            use_class_weights = TRAINING_CONFIG["use_class_weights"]
        
        trainer = BERTTrainer(
            model_name=args.model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            warmup_steps=TRAINING_CONFIG["warmup_steps"],
            resume_from_checkpoint=resume_checkpoint if continue_training else None,
            models_output_dir=models_dir,  # Pass the models directory
            text_field=args.text_field,  # Pass the text field parameter
            target_field=args.target_field,  # Pass the target field parameter
            weight_decay=TRAINING_CONFIG["weight_decay"],
            max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
            dropout_rate=MODEL_CONFIG["dropout_rate"],
            use_class_weights=use_class_weights
        )
        
        # Load processed data
        train_df, test_df = trainer.load_data(processed_data_dir)
        
        # Calculate and set class weights if enabled
        if use_class_weights:
            logger.info("Calculating class weights for imbalanced data...")
            processor = ForumDataProcessor(args.dataset_path, 
                                         topics_file=TOPICS_CONFIG["topics_file"],
                                         target_field=args.target_field)
            # Set the mlb from trainer to processor for weight calculation
            processor.mlb = trainer.mlb
            class_weights = processor.calculate_class_weights(
                train_df, 
                method=args.class_weight_method
            )
            trainer.set_class_weights(class_weights)
            logger.info(f"Class weights applied using method: {args.class_weight_method}")
        else:
            logger.info("Class weighting disabled")
        
        # Prepare datasets
        train_loader, val_loader, test_loader = trainer.prepare_datasets(
            train_df, test_df, 
            validation_split=TRAINING_CONFIG["validation_split"]
        )
        
        # Initialize model
        num_labels = len(trainer.mlb.classes_)
        trainer.initialize_model(num_labels)
        
        logger.info(f"Model initialized with {num_labels} labels")
        logger.info(f"Model architecture: {args.model_name}")
        
        # Train model
        logger.info("Starting model training...")
        trainer.train(train_loader, val_loader)
        
        # Step 3: Final Evaluation
        logger.info("="*50)
        logger.info("STEP 3: FINAL EVALUATION")
        logger.info("="*50)
        
        # Evaluate on test set
        test_loss, test_metrics = trainer.evaluate(test_loader)
        
        logger.info("Final Test Results:")
        logger.info(f"Test Loss: {test_loss:.4f}")
        for metric, value in test_metrics.items():
            logger.info(f"Test {metric}: {value:.4f}")
        
        # Save test results
        test_results = {
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "model_config": {
                "model_name": args.model_name,
                "num_labels": num_labels,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "text_field": args.text_field
            }
        }
        
        results_file = os.path.join(args.output_dir, "final_results.json")
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)
        logger.info(f"Final results saved to: {results_file}")
        
        # Create training history plot
        try:
            os.makedirs(plots_dir, exist_ok=True)
            history_plot = os.path.join(plots_dir, "training_history.png")
            trainer.plot_training_history(history_plot)
            logger.info(f"Training history plot saved to: {history_plot}")
        except Exception as e:
            logger.warning(f"Could not create training history plot: {e}")
        
        # Save best model to models directory
        best_model_path = os.path.join(models_dir, "best_model.pth")
        if os.path.exists("best_model.pth"):
            import shutil
            shutil.move("best_model.pth", best_model_path)
            logger.info(f"Best model moved to: {best_model_path}")
        elif hasattr(trainer, 'models_output_dir') and os.path.exists(os.path.join(trainer.models_output_dir, "best_model.pth")):
            # Model was already saved in the correct directory
            logger.info(f"Best model saved to: {os.path.join(trainer.models_output_dir, 'best_model.pth')}")
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        
        # Print summary
        logger.info("SUMMARY:")
        logger.info(f"  - Dataset: {args.dataset_path}")
        logger.info(f"  - Model: {args.model_name}")
        logger.info(f"  - Data split: {'New split created' if args.data_split else 'Used existing split'}")
        logger.info(f"  - Training samples: {len(train_df) if 'train_df' in locals() else 'N/A'}")
        logger.info(f"  - Test samples: {len(test_df) if 'test_df' in locals() else 'N/A'}")
        logger.info(f"  - Number of labels: {num_labels}")
        logger.info(f"  - Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"  - Test Subset Accuracy: {test_metrics['subset_accuracy']:.4f}")
        logger.info(f"  - Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
