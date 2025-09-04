# BERT Multi-Label Text Classification for UXPID dataset

An implementation of BERT-based multi-label text classification for forum thread categorization.

## 📁 Project Structure

### Standard Structure
```
bert_classification/
├── dataset/                   # Raw data (JSON files)
├── processed_data/            # Processed training/test data
├── data_processor.py          # Data preprocessing module
├── bert_trainer.py            # BERT training module
├── predictor.py               # Inference module
├── evaluate.py                # Evaluation module
├── main.py                    # Main training script
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions for timestamp folders
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Timestamp-Based Output Structure (Default)
When using timestamp folders (enabled by default), all outputs are organized as:
```
outputs/
└── YYYYMMDD_HHMMSS/          
    ├── models/                
    │   └── best_model.pth
    ├── logs/                  
    │   └── training_YYYYMMDD_HHMMSS.log
    ├── plots/                 
    │   ├── training_history.png
    │   └── data_visualization.png
    ├── results/               
    │   ├── final_results.json
    │   ├── training_config.json
    │   └── data_analysis.json
    ├── evaluation_results/    
    │   ├── evaluation_report.json
    │   └── plots/
    └── run_summary.json    
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bert_classification
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Format

The dataset should contain JSON files in the `dataset/` directory. Each JSON file represents a forum thread with the following structure:

```json
{
  "metadata": {
    "branch_id": "1000153386",
    "thread_id": "9520646325", 
    "publication_year": 2021,
    "topic_status": "Unsolved",
    "topic_type": "question"
  },
  "content": [
    {
      "comment_id": 341703487007721,
      "user_name": "user123",
      "comment_position": 0,
      "is_reply": false,
      "comment_year": 2021,
      "comment_month": 9,
      "comment_body": "Forum post text content..."
    }
  ],
  "topics": {
    "21": "Bug Reports",
    "13": "Network Configuration Challenges"
  }
}
```

## 🚀 Quick Start

Run the complete training pipeline with automatic timestamp folders:

```bash
python main.py
```

This will:
- Create a timestamp folder (e.g., `outputs/20250118_143052/`)
- Process the raw forum data
- Create train/validation/test splits  
- Train the BERT model
- Save all outputs (models, logs, plots, results) in organized subfolders

### Making Predictions

```python
from predictor import BERTPredictor

# Initialize predictor
predictor = BERTPredictor("best_model.pth")

# Single prediction
text = "I'm having network connectivity issues with my device."
result = predictor.predict_single(text)
print(f"Predicted labels: {result['predicted_labels']}")

# Batch predictions
texts = ["Text 1", "Text 2", "Text 3"]
results = predictor.predict_batch(texts)

# Get top-k predictions
top_predictions = predictor.get_top_predictions(text, top_k=3)
```

### Model Evaluation

```bash
python evaluate.py --model-path best_model.pth --output-dir evaluation_results
```

This generates:
- Comprehensive evaluation metrics
- Confusion matrices
- Precision-Recall curves
- ROC curves
- Error analysis

## ⚙️ Configuration

Edit `config.py` to customize:

- **Model settings**: BERT model type, sequence length, etc.
- **Training parameters**: Learning rate, batch size, epochs
- **Data processing**: Text cleaning, filtering options
- **Evaluation metrics**: Thresholds, metrics to track

Key configuration sections:

```python
MODEL_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "dropout_rate": 0.3,
}

TRAINING_CONFIG = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
}
```

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ GPU memory
