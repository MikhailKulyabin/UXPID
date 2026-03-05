# BERT Multi-Label Text Classification for UXPID dataset

An implementation of BERT-based multi-label text classification for forum thread categorization.

**Dataset:** [UXPID on Zenodo](https://zenodo.org/records/17091284) | **Paper:** [arXiv:2509.11777](https://arxiv.org/abs/2509.11777)

## 📁 Project Structure

### Standard Structure
```
bert_classification/
├── dataset/                   # Raw data (JSON files)
├── splits/                    # Official Zenodo train/test split
│   ├── train_branches.txt
│   └── test_branches.txt
├── processed_data/            # Processed training/test data
├── data_processor.py          # Data preprocessing module
├── bert_trainer.py            # BERT training module
├── tfidf_baseline.py          # TF-IDF + Logistic Regression baseline
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

Download the dataset from [Zenodo](https://zenodo.org/records/17091284) and place the JSON files in the `dataset/` directory. Each JSON file represents a forum thread with the following structure:

```json
{
  "metadata": {
    "branch_id": "1000153386",
    "thread_id": "9520646325", 
    "publication_year": 2021,
    "branch_status": "Unsolved",
    "branch_type": "question"
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

Run the complete BERT training pipeline with automatic timestamp folders:

```bash
python main.py
```

Run the TF-IDF + Logistic Regression **baseline** as a separate pipeline:

```bash
python tfidf_baseline.py
```

Both scripts share the same data, config, and output folder structure. Key CLI options for the baseline:

| Flag | Default | Description |
|---|---|---|
| `--text-field` | from `config.py` | `text` or `insight_summary` |
| `--target-field` | from `config.py` | `topics`, `branch_status`, `branch_type`, `overall_thread_sentiment` |
| `--max-features` | `50000` | TF-IDF vocabulary size |
| `--ngram-max` | `2` | Upper n-gram bound (1 = unigrams, 2 = unigrams+bigrams) |
| `--C` | `1.0` | Logistic Regression regularisation strength |
| `--skip-data-processing` | off | Load existing `processed_data/` without re-processing |
| `--data-split` | off | Create a new train/test split |
| `--use-official-split` | off | Use the official Zenodo split (requires `--data-split`) |
| `--splits-dir` | `splits` | Path to `train_branches.txt` / `test_branches.txt` |

To use the **official Zenodo split** (recommended for reproducibility):

```bash
python tfidf_baseline.py --data-split --use-official-split
python main.py --data-split --use-official-split
```

To compare TF-IDF and BERT side by side, use `TFIDFBaseline.print_comparison()` / `plot_comparison()` programmatically after both runs.

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

## AI Use Disclosure

The **UXPID corpus** was created with the assistance of AI tools. For full details, see the paper: [arXiv:2509.11777](https://arxiv.org/abs/2509.11777). Specifically, the `analysis` fields present in each JSON record — including `insight_summary`, `user_expectations`, `severity_expectation_level`, `gain_keywords`, `pain_keywords`, `feature_keywords`, and `overall_thread_sentiment` — were generated using a large language model (LLM). These AI-generated annotations were produced by prompting the model to analyse the raw forum thread content and extract structured UX-relevant insights.

The raw forum content (`content` field) was artificially synthesized and anonymized from branches originally extracted from a public industrial automation forum. Thread metadata (`metadata` field) was derived from the same source.

## 📝 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ GPU memory

## 📄 Citation

If you use this dataset or code in your research, please cite the paper: [arXiv:2509.11777](https://arxiv.org/abs/2509.11777)

```bibtex
@article{kulyabin2025user,
  title={User eXperience Perception Insights Dataset (UXPID): Synthetic User Feedback from Public Industrial Forums},
  author={Kulyabin, Mikhail and Joosten, Jan and Pacheco, Nuno Miguel Martins and Ries, Fabian and Petridis, Filippos and Bosch, Jan and Olsson, Helena Holmstr{\"o}m and others},
  journal={arXiv preprint arXiv:2509.11777},
  year={2025}
}
```
