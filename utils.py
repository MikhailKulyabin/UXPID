import os
from datetime import datetime
from typing import Dict, Tuple
from config import OUTPUT_CONFIG


def create_timestamp_folder(base_dir: str = None) -> Tuple[str, str]:
    """
    Create a timestamp-based folder for organizing outputs.
    
    Args:
        base_dir: Base directory name for outputs
        
    Returns:
        Tuple of (timestamp_string, full_output_path)
    """
    if base_dir is None:
        base_dir = OUTPUT_CONFIG["outputs_base_dir"]
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_dir, timestamp)
    
    # Create the directory structure
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_path, OUTPUT_CONFIG["logs_dir"]), exist_ok=True)
    os.makedirs(os.path.join(output_path, OUTPUT_CONFIG["plots_dir"]), exist_ok=True)
    os.makedirs(os.path.join(output_path, OUTPUT_CONFIG["results_dir"]), exist_ok=True)
    os.makedirs(os.path.join(output_path, "evaluation_results"), exist_ok=True)
    
    return timestamp, output_path


def get_timestamp_paths(output_path: str) -> Dict[str, str]:
    """
    Get all timestamp-based output paths.
    
    Args:
        output_path: Base output path with timestamp
        
    Returns:
        Dictionary with all output paths
    """
    return {
        "base": output_path,
        "models": os.path.join(output_path, "models"),
        "logs": os.path.join(output_path, OUTPUT_CONFIG["logs_dir"]),
        "plots": os.path.join(output_path, OUTPUT_CONFIG["plots_dir"]),
        "results": os.path.join(output_path, OUTPUT_CONFIG["results_dir"]),
        "evaluation_results": os.path.join(output_path, "evaluation_results"),
    }


def create_run_summary(output_path: str, config_data: dict, timestamp: str) -> str:
    """
    Create a summary file for the training run.
    
    Args:
        output_path: Base output path
        config_data: Configuration data
        timestamp: Timestamp string
        
    Returns:
        Path to the summary file
    """
    import json
    
    summary_data = {
        "run_id": timestamp,
        "created_at": datetime.now().isoformat(),
        "output_path": output_path,
        "config": config_data
    }
    
    summary_file = os.path.join(output_path, "run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    return summary_file
