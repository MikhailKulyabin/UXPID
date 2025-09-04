import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from bert_trainer import BERTMultiLabelClassifier
from config import PREDICTION_CONFIG
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BERTPredictor:
    """Predictor class for BERT multi-label classification inference."""
    
    def __init__(self, model_path: str, data_dir: str = "processed_data", 
                 threshold: float = None, topics_file: str = "topics.json", target_field: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model
            data_dir: Directory containing processed data and label encoder
            threshold: Threshold for binary predictions (defaults to config value)
            topics_file: Path to topics.json for ID to name mapping
            target_field: Target field used during training (if None, tries to detect from config)
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.topics_file = topics_file
        
        # Determine target field
        if target_field is None:
            try:
                from config import DATA_CONFIG
                self.target_field = DATA_CONFIG.get("target_field", "topics")
            except ImportError:
                self.target_field = "topics"  # fallback
        else:
            self.target_field = target_field
            
        print(f"Predictor initialized for target field: {self.target_field}")
        self.threshold = threshold if threshold is not None else PREDICTION_CONFIG['threshold']
        
        print(f"Using prediction threshold: {self.threshold}")
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and dependencies
        self.model = None
        self.tokenizer = None
        self.mlb = None
        self.label_mapping = None
        self.topics_mapping = None
        
        self._load_model_and_dependencies()
        self._load_topics_mapping()
    
    def _load_model_and_dependencies(self):
        """Load the trained model and all dependencies."""
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['model_name'])
        
        # Initialize and load model
        self.model = BERTMultiLabelClassifier(
            checkpoint['model_name'], 
            checkpoint['num_labels']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder with target field suffix
        target_suffix = f"_{self.target_field}"
        encoder_file = os.path.join(self.data_dir, f"label_encoder{target_suffix}.pkl")
        
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Label encoder file not found for target '{self.target_field}': {encoder_file}")
            
        with open(encoder_file, "rb") as f:
            self.mlb = pickle.load(f)
        
        # Load class information with target field suffix
        class_info_file = os.path.join(self.data_dir, f"class_info{target_suffix}.json")
        if os.path.exists(class_info_file):
            with open(class_info_file, "r") as f:
                self.class_info = json.load(f)
                print(f"Loaded class info for '{self.target_field}': {self.class_info['num_classes']} classes")
        else:
            # Fallback for legacy files
            print(f"Class info file not found for target '{self.target_field}', using basic setup")
            self.class_info = {
                'classes': list(self.mlb.classes_),
                'num_classes': len(self.mlb.classes_),
                'target_field': self.target_field
            }
        
        self.max_length = checkpoint['max_length']
        
        print(f"Model loaded successfully!")
        print(f"Number of labels: {len(self.mlb.classes_)}")
        print(f"Model name: {checkpoint['model_name']}")
        print(f"Max sequence length: {self.max_length}")
    
    def _load_topics_mapping(self):
        """Load topics mapping from topics.json for enhanced readability."""
        try:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
                self.topics_mapping = topics_data.get('topics', {})
                print(f"Loaded topics mapping with {len(self.topics_mapping)} topics")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load topics file {self.topics_file}: {e}")
            self.topics_mapping = {}
    
    def get_topic_name(self, topic_id: str) -> str:
        """Get topic name from topic ID for display purposes."""
        return self.topics_mapping.get(topic_id, topic_id)
    
    def predict_single(self, text: str, include_topic_names: bool = False) -> Dict[str, any]:
        """
        Make prediction for a single text.
        
        Args:
            text: Input text string
            include_topic_names: Whether to include topic names alongside IDs
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Convert to binary predictions
        binary_predictions = (probabilities > self.threshold).astype(int)
        
        # Get predicted labels
        predicted_labels = []
        predicted_probs = {}
        predicted_labels_with_names = []  # For enhanced readability
        
        for i, (prob, pred) in enumerate(zip(probabilities, binary_predictions)):
            label = self.mlb.classes_[i]
            predicted_probs[label] = float(prob)
            if pred == 1:
                predicted_labels.append(label)
                if include_topic_names:
                    topic_name = self.get_topic_name(label)
                    predicted_labels_with_names.append(f"{label}: {topic_name}")
        
        result = {
            'text': text,
            'predicted_labels': predicted_labels,
            'all_probabilities': predicted_probs,
            'binary_predictions': binary_predictions.tolist(),
            'num_predicted_labels': len(predicted_labels)
        }
        
        if include_topic_names:
            result['predicted_labels_with_names'] = predicted_labels_with_names
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32, include_topic_names: bool = False) -> List[Dict[str, any]]:
        """
        Make predictions for a batch of texts.
        
        Args:
            texts: List of input text strings
            batch_size: Batch size for processing
            include_topic_names: Whether to include topic names alongside IDs
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Make predictions
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probabilities = torch.sigmoid(logits).cpu().numpy()
            
            # Process each prediction in the batch
            for j, (text, probs) in enumerate(zip(batch_texts, probabilities)):
                binary_predictions = (probs > self.threshold).astype(int)
                
                predicted_labels = []
                predicted_probs = {}
                predicted_labels_with_names = []  # For enhanced readability
                
                for k, (prob, pred) in enumerate(zip(probs, binary_predictions)):
                    label = self.mlb.classes_[k]
                    predicted_probs[label] = float(prob)
                    if pred == 1:
                        predicted_labels.append(label)
                        if include_topic_names:
                            topic_name = self.get_topic_name(label)
                            predicted_labels_with_names.append(f"{label}: {topic_name}")
                
                result = {
                    'text': text,
                    'predicted_labels': predicted_labels,
                    'all_probabilities': predicted_probs,
                    'binary_predictions': binary_predictions.tolist(),
                    'num_predicted_labels': len(predicted_labels)
                }
                
                if include_topic_names:
                    result['predicted_labels_with_names'] = predicted_labels_with_names
                
                results.append(result)
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str, 
                         text_column: str = 'text') -> pd.DataFrame:
        """
        Make predictions for texts from a CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save predictions
            text_column: Name of the text column
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(input_file)
        texts = df[text_column].tolist()
        
        print(f"Making predictions for {len(texts)} texts...")
        
        # Make predictions
        predictions = self.predict_batch(texts)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['predicted_labels'] = [pred['predicted_labels'] for pred in predictions]
        results_df['num_predicted_labels'] = [pred['num_predicted_labels'] for pred in predictions]
        
        # Add probability columns
        for label in self.mlb.classes_:
            results_df[f'prob_{label}'] = [pred['all_probabilities'][label] for pred in predictions]
        
        # Save results
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        return results_df
    
    def get_top_predictions(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k most likely labels for a text.
        
        Args:
            text: Input text string
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, probability) tuples sorted by probability
        """
        prediction = self.predict_single(text)
        
        # Sort probabilities in descending order
        sorted_probs = sorted(
            prediction['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_probs[:top_k]
    
    def analyze_predictions(self, texts: List[str]) -> Dict[str, any]:
        """
        Analyze predictions for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Analysis dictionary
        """
        predictions = self.predict_batch(texts)
        
        # Collect statistics
        all_predicted_labels = []
        num_labels_per_text = []
        
        for pred in predictions:
            all_predicted_labels.extend(pred['predicted_labels'])
            num_labels_per_text.append(pred['num_predicted_labels'])
        
        # Calculate statistics
        from collections import Counter
        
        label_counts = Counter(all_predicted_labels)
        
        analysis = {
            'total_texts': len(texts),
            'avg_labels_per_text': np.mean(num_labels_per_text),
            'max_labels_per_text': np.max(num_labels_per_text),
            'min_labels_per_text': np.min(num_labels_per_text),
            'most_common_labels': label_counts.most_common(10),
            'label_distribution': dict(label_counts),
            'texts_with_no_labels': sum(1 for n in num_labels_per_text if n == 0),
            'texts_with_multiple_labels': sum(1 for n in num_labels_per_text if n > 1)
        }
        
        return analysis
    
    def calculate_metrics(self, predictions: np.ndarray, labels: np.ndarray, 
                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate multi-label classification metrics.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: True labels
            threshold: Threshold for binary predictions
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, hamming_loss, jaccard_score, 
            precision_recall_fscore_support
        )
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Subset accuracy (exact match)
        metrics['subset_accuracy'] = accuracy_score(labels, binary_predictions)
        
        # Hamming loss
        metrics['hamming_loss'] = hamming_loss(labels, binary_predictions)
        
        # Jaccard score (IoU)
        metrics['jaccard_score'] = jaccard_score(labels, binary_predictions, average='samples')
        
        # Precision, Recall, F1 (macro and micro averages)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, binary_predictions, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, binary_predictions, average='micro', zero_division=0
        )
        
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro
        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        metrics['f1_micro'] = f1_micro
        
        return metrics

def main():
    """Main function for demonstration."""
    
    # Check if model exists
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    # Initialize predictor
    try:
        predictor = BERTPredictor(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example predictions
    example_texts = [
        "I'm having trouble with my device. It won't connect to the network properly.",
        "How can I configure the timer settings in the software?",
        "The web interface is not displaying data correctly. Need help troubleshooting.",
        "Looking for documentation on API integration features.",
        "Device compatibility issues with newer firmware versions."
    ]
    
    print("\n=== EXAMPLE PREDICTIONS ===")
    for i, text in enumerate(example_texts, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        
        # Single prediction with topic names
        prediction = predictor.predict_single(text, include_topic_names=True)
        print(f"Predicted Labels (IDs): {prediction['predicted_labels']}")
        if 'predicted_labels_with_names' in prediction:
            print(f"Predicted Labels (with names): {prediction['predicted_labels_with_names']}")
        
        # Top predictions
        top_predictions = predictor.get_top_predictions(text, top_k=3)
        print("Top 3 Predictions:")
        for label, prob in top_predictions:
            topic_name = predictor.get_topic_name(label)
            print(f"  {label} ({topic_name}): {prob:.3f}")
    
    # Batch analysis
    print("\n=== BATCH ANALYSIS ===")
    analysis = predictor.analyze_predictions(example_texts)
    for key, value in analysis.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
