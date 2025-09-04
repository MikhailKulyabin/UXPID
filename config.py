# Data Configuration
DATA_CONFIG = {
    "dataset_path": "dataset",               # Path to dataset directory
    "processed_data_dir": "processed_data",  # Directory for processed data
    "min_text_length": 10,                   # Minimum text length to include
    "max_text_length": 5000,                 # Maximum text length to include
    "min_labels_per_sample": 1,              # Minimum number of labels per sample
    "clean_text": True,                      # Whether to apply text cleaning
    "text_field": "insight_summary",         # Input to use for training: "text" or "insight_summary"
    "target_field": "topics",                # Target field for classification: "topics", "topic_status", "topic_type", or "overall_thread_sentiment"
    "data_split": False,                     # Whether to create new data split (True) or use existing split (False)
}

# Model Configuration
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased", 
    "max_length": 512,                       
    "dropout_rate": 0.4,                     
    "num_hidden_layers": 6,                  
    "hidden_size": 768,                      
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 16,                  
    "learning_rate": 2e-5,             
    "num_epochs": 100,                 
    "warmup_steps": 0,                 # Warmup steps for learning rate scheduler
    "weight_decay": 0.01,              # Weight decay for regularization
    "max_grad_norm": 1.0,              # Maximum gradient norm for clipping
    "validation_split": 0.1,           
    "test_size": 0.2,                  
    "random_state": 42,                
    "continue_training": False,        # Whether to continue training from checkpoint
    "resume_from_checkpoint": "",      # Path to checkpoint to resume from 
    "use_class_weights": True,         # Whether to use class weighting for imbalanced data
    "class_weight_method": "balanced", # Method for calculating class weights: 'balanced', 'inverse_freq', or 'custom'
}

# Prediction Configuration
PREDICTION_CONFIG = {
    "threshold": 0.3,                  
    "top_k": 10,                       # Number of top predictions to show
    "batch_size": 32,                  # Batch size for inference
}

# Output Configuration
OUTPUT_CONFIG = {
    "model_save_path": "best_model.pth",           # Path to save best model (relative to models dir)
    "results_dir": "results",                      # Directory for results (when not using timestamps)
    "plots_dir": "plots",                          # Directory for plots (when not using timestamps)
    "logs_dir": "logs",                            # Directory for logs (when not using timestamps)
    "use_timestamp_folders": True,                 # Use timestamp-based folder organization
    "outputs_base_dir": "output",                 
    "save_intermediate": True,                     
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",               
    "log_file": "training.log",        
    "tensorboard_dir": "runs",         
    "wandb_project": "bert-multilabel", 
    "use_wandb": False,                
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "use_cuda": True,                  # Whether to use CUDA if available
    "mixed_precision": False,          # Whether to use mixed precision training
    "dataloader_num_workers": 4,       # Number of workers for data loading
    "pin_memory": True,                # Whether to pin memory for faster data transfer
}

# Topics Configuration
TOPICS_CONFIG = {
    "topics_file": "topics.json",      # Path to topics mapping file
}

# Evaluation Metrics
METRICS_CONFIG = {
    "primary_metric": "f1_macro",      # Primary metric for model selection
    "metrics_to_track": [
        "subset_accuracy",
        "hamming_loss", 
        "jaccard_score",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro", 
        "recall_micro",
        "f1_micro"
    ],
    "threshold_search": False,         # Whether to search for optimal threshold
    "threshold_range": (0.1, 0.9),     # Range for threshold search
    "threshold_steps": 9,              # Number of threshold values to test
}

# Text Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "lowercase": True,                 # Convert text to lowercase
    "remove_html": True,               # Remove HTML tags
    "remove_urls": True,               # Remove URLs
    "remove_mentions": False,          # Remove @mentions
    "remove_hashtags": False,          # Remove hashtags
    "remove_special_chars": False,     # Remove special characters
    "normalize_whitespace": True,      # Normalize whitespace
    "remove_short_words": False,       # Remove words shorter than n characters
    "min_word_length": 2,              # Minimum word length
}

# Model Checkpointing
CHECKPOINT_CONFIG = {
    "save_every_epoch": True,          # Save checkpoint after every epoch
    "keep_last_n": 3,                  # Keep last n checkpoints
    "save_best_only": True,            # Save only the best model
    "monitor_metric": "f1_macro",      # Metric to monitor for best model
    "mode": "max",                     # Mode for monitoring (max or min)
}

# Early Stopping Configuration
EARLY_STOPPING_CONFIG = {
    "enabled": True,                   # Enable early stopping
    "patience": 2,                     # Reduced patience for faster stopping
    "min_delta": 0.005,                # Increased minimum change requirement
    "monitor_metric": "f1_macro",      # Metric to monitor
    "mode": "max",                     # Mode for monitoring (max or min)
    "restore_best_weights": True,      # Restore best weights on early stop
}

