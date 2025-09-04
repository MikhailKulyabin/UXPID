import os
import json
import pickle
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc,
    accuracy_score
)
from collections import defaultdict
from typing import Dict, List, Tuple
from config import PREDICTION_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
import warnings
warnings.filterwarnings('ignore')

try:
    from predictor import BERTPredictor
except ImportError:
    print("Warning: Could not import BERTPredictor. Some features may not work.")

try:
    from utils import create_timestamp_folder, get_timestamp_paths
except ImportError:
    print("Warning: Could not import utils. Timestamp folder features may not work.")


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, model_path: str, data_dir: str = "processed_data", 
                 text_field: str = "text", topics_file: str = "topics.json", target_field: str = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            data_dir: Directory containing processed data
            text_field: Field to use for evaluation: "text" or "insight_summary"
            topics_file: Path to topics.json for ID to name mapping
            target_field: Target field used during training (topics, topic_status, topic_type, overall_thread_sentiment)
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.text_field = text_field
        self.topics_file = topics_file
        self.target_field = target_field if target_field is not None else DATA_CONFIG.get("target_field", "topics")
        
        print(f"Evaluator initialized with target field: {self.target_field}")
        
        # Load topics mapping for better readability in reports (only for topics target field)
        if self.target_field == "topics":
            self.topics_mapping = self._load_topics_mapping()
        else:
            self.topics_mapping = {}
            print(f"Using target field '{self.target_field}' - skipping topics mapping")
        
        # Load data and model with target field suffix
        target_suffix = f"_{self.target_field}"
        test_data_file = os.path.join(data_dir, "test_data.csv")
        encoder_file = os.path.join(data_dir, f"label_encoder{target_suffix}.pkl")
        
        if not os.path.exists(test_data_file):
            raise FileNotFoundError(f"Test data file not found: {test_data_file}")
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Label encoder file not found for target '{self.target_field}': {encoder_file}")
        
        self.test_df = pd.read_csv(test_data_file)
        
        # Convert all label columns back to lists (they were saved as strings)
        import ast
        label_columns_to_convert = ['topics_labels', 'topic_status_labels', 'topic_type_labels', 'sentiment_labels']
        
        for col in label_columns_to_convert:
            if col in self.test_df.columns:
                self.test_df[col] = self.test_df[col].apply(ast.literal_eval)
        
        # Validate that the specified text field exists
        if self.text_field not in self.test_df.columns:
            raise ValueError(f"Text field '{self.text_field}' not found in test data. Available columns: {list(self.test_df.columns)}")
        
        with open(encoder_file, "rb") as f:
            self.mlb = pickle.load(f)
        
        print(f"Loaded test data for target '{self.target_field}': {len(self.test_df)} samples, {len(self.mlb.classes_)} classes")
        
        self.predictor = None
        if os.path.exists(model_path):
            try:
                # Initialize predictor with config threshold and target field
                self.predictor = BERTPredictor(model_path, data_dir, target_field=self.target_field)
                print(f"Predictor initialized with threshold from config: {PREDICTION_CONFIG['threshold']}")
            except Exception as e:
                print(f"Could not load predictor: {e}")
    
    def _load_topics_mapping(self) -> Dict[str, str]:
        """Load topics mapping from topics.json for enhanced readability (only for topics target field)."""
        if self.target_field != "topics":
            return {}
        
        try:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                topics_data = json.load(f)
                topics_mapping = topics_data.get('topics', {})
                print(f"Loaded topics mapping with {len(topics_mapping)} topics for evaluation reports")
                return topics_mapping
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load topics file {self.topics_file}: {e}")
            return {}
    
    def get_topic_name(self, label_id: str) -> str:
        """Get human-readable name for a label based on target field."""
        if self.target_field == "topics":
            return self.topics_mapping.get(label_id, label_id)
        else:
            # For non-topic fields, return the label as-is (it's already human-readable)
            return label_id
    
    def evaluate_test_set(self) -> Dict[str, any]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Dictionary containing evaluation results
        """
        if self.predictor is None:
            raise ValueError("Predictor not loaded. Cannot evaluate.")
        
        # Get test texts and true labels using the specified text field
        test_texts = self.test_df[self.text_field].tolist()
        
        # Determine if this is multilabel (topics) or binary classification (all others)
        self.is_multilabel = (self.target_field == "topics")
        
        # Get the appropriate label columns based on target field
        if self.is_multilabel:
            # For topics (multilabel), use one-hot encoded columns (topic_id_1, topic_id_2, etc.)
            label_columns = [col for col in self.test_df.columns if col.startswith('topic_id_')]
            if not label_columns:
                raise ValueError(f"No topic_id_ columns found for topics target field. Available columns: {list(self.test_df.columns)}")
            # Sort the columns to ensure proper order
            label_columns = sorted(label_columns, key=lambda x: int(x.split('_')[-1]))
            true_labels = self.test_df[label_columns].values.astype(int)
        else:
            # For binary classification targets, get the single label for each sample
            # Select the appropriate labels column based on target field
            if self.target_field == "topic_status":
                labels_column = 'topic_status_labels'
            elif self.target_field == "topic_type":
                labels_column = 'topic_type_labels'
            elif self.target_field == "overall_thread_sentiment":
                labels_column = 'sentiment_labels'
            else:
                raise ValueError(f"Unknown target field: {self.target_field}")
            
            if labels_column not in self.test_df.columns:
                raise ValueError(f"Labels column '{labels_column}' not found. Available columns: {list(self.test_df.columns)}")
            
            # For binary classification, each sample should have exactly one label
            # Convert single labels to class indices
            true_labels_list = []
            class_to_index = {cls: i for i, cls in enumerate(self.mlb.classes_)}
            
            skipped_samples = 0
            valid_indices = []
            
            for idx, sample_labels in enumerate(self.test_df[labels_column]):
                if len(sample_labels) == 0:
                    print(f"Warning: Sample {idx} has no labels, skipping...")
                    skipped_samples += 1
                    continue
                elif len(sample_labels) > 1:
                    print(f"Warning: Sample {idx} has multiple labels {sample_labels}, using first label only...")
                    label = sample_labels[0]
                else:
                    label = sample_labels[0]
                
                if label not in class_to_index:
                    print(f"Warning: Sample {idx} has unknown label '{label}', skipping...")
                    skipped_samples += 1
                    continue
                
                true_labels_list.append(class_to_index[label])
                valid_indices.append(idx)
            
            if skipped_samples > 0:
                print(f"Skipped {skipped_samples} samples with invalid labels")
                # Filter test_texts and test_df to only include valid samples
                test_texts = [test_texts[i] for i in valid_indices]
                self.test_df = self.test_df.iloc[valid_indices].reset_index(drop=True)
                print(f"Proceeding with {len(valid_indices)} valid samples")
            
            true_labels = np.array(true_labels_list)
        
        print(f"Evaluating {len(test_texts)} test samples...")
        
        # Get predictions
        predictions = self.predictor.predict_batch(test_texts)
        
        # Extract predicted labels and probabilities
        if self.is_multilabel:
            # For multilabel (topics), use the existing logic
            pred_labels = []
            pred_probs = []
            
            for pred in predictions:
                # Convert binary predictions
                binary_pred = np.zeros(len(self.mlb.classes_))
                for label in pred['predicted_labels']:
                    if label in self.mlb.classes_:
                        idx = list(self.mlb.classes_).index(label)
                        binary_pred[idx] = 1
                pred_labels.append(binary_pred)
                
                # Get probabilities in correct order
                prob_array = np.zeros(len(self.mlb.classes_))
                for i, label in enumerate(self.mlb.classes_):
                    prob_array[i] = pred['all_probabilities'][label]
                pred_probs.append(prob_array)
            
            pred_labels = np.array(pred_labels)
            pred_probs = np.array(pred_probs)
        else:
            # For binary classification, get the predicted class indices
            pred_labels = []
            pred_probs = []
            
            for pred in predictions:
                # For binary classification, take the class with highest probability
                probs = [pred['all_probabilities'][label] for label in self.mlb.classes_]
                predicted_class_idx = np.argmax(probs)
                
                pred_labels.append(predicted_class_idx)
                pred_probs.append(probs)
            
            pred_labels = np.array(pred_labels)
            pred_probs = np.array(pred_probs)
        
        # Calculate metrics based on classification type
        if self.is_multilabel:
            # For multilabel classification, use existing multilabel metrics
            metrics = self.predictor.calculate_metrics(pred_probs, true_labels)
        else:
            # For binary classification, calculate appropriate metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            # Calculate basic metrics for binary classification
            accuracy = accuracy_score(true_labels, pred_labels)
            
            # Get classification report for detailed metrics
            report_dict = classification_report(true_labels, pred_labels, 
                                              target_names=self.mlb.classes_, 
                                              output_dict=True, zero_division=0)
            
            # Extract macro and weighted averages
            metrics = {
                'accuracy': accuracy,
                'precision_macro': report_dict['macro avg']['precision'],
                'recall_macro': report_dict['macro avg']['recall'],
                'f1_macro': report_dict['macro avg']['f1-score'],
                'precision_weighted': report_dict['weighted avg']['precision'],
                'recall_weighted': report_dict['weighted avg']['recall'],
                'f1_weighted': report_dict['weighted avg']['f1-score'],
                'precision_micro': accuracy,  # For multiclass, micro avg precision = accuracy
                'recall_micro': accuracy,     # For multiclass, micro avg recall = accuracy
                'f1_micro': accuracy,         # For multiclass, micro avg f1 = accuracy
                'support': len(true_labels)
            }
        
        # Add additional metrics
        metrics['num_test_samples'] = len(test_texts)
        metrics['num_labels'] = len(self.mlb.classes_)
        
        # Per-label metrics
        per_label_metrics = {}
        
        if self.is_multilabel:
            # For multilabel, use the existing per-label logic
            for i, label in enumerate(self.mlb.classes_):
                label_true = true_labels[:, i]
                label_pred = pred_labels[:, i]
                label_prob = pred_probs[:, i]
                
                # Calculate confusion matrix components
                tp = np.sum((label_true == 1) & (label_pred == 1))
                fp = np.sum((label_true == 0) & (label_pred == 1))
                tn = np.sum((label_true == 0) & (label_pred == 0))
                fn = np.sum((label_true == 1) & (label_pred == 0))
                
                # Calculate basic metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                per_label_metrics[label] = {
                    'support': int(label_true.sum()),
                    'predicted_positive': int(label_pred.sum()),
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity,
                    'prevalence': label_true.sum() / len(label_true)
                }
                
                if label_true.sum() > 0:  # Only if label exists in test set
                    # Precision-Recall curve
                    precision_curve, recall_curve, _ = precision_recall_curve(label_true, label_prob)
                    ap_score = average_precision_score(label_true, label_prob)
                    
                    # ROC curve
                    fpr, tpr, _ = roc_curve(label_true, label_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    per_label_metrics[label].update({
                        'ap_score': ap_score,
                        'roc_auc': roc_auc,
                        'precision_curve': precision_curve.tolist(),
                        'recall_curve': recall_curve.tolist(),
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    })
                else:
                    per_label_metrics[label].update({
                        'ap_score': 0.0,
                        'roc_auc': 0.5,
                        'precision_curve': [],
                        'recall_curve': [],
                        'fpr': [],
                        'tpr': []
                    })
        else:
            # For binary classification, calculate per-class metrics
            from sklearn.metrics import confusion_matrix
            
            # Get confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            for i, label in enumerate(self.mlb.classes_):
                # For each class in binary classification
                # Create binary labels: current class vs all others
                label_true_binary = (true_labels == i).astype(int)
                label_pred_binary = (pred_labels == i).astype(int)
                label_prob = pred_probs[:, i]
                
                # Calculate confusion matrix components
                tp = np.sum((label_true_binary == 1) & (label_pred_binary == 1))
                fp = np.sum((label_true_binary == 0) & (label_pred_binary == 1))
                tn = np.sum((label_true_binary == 0) & (label_pred_binary == 0))
                fn = np.sum((label_true_binary == 1) & (label_pred_binary == 0))
                
                # Calculate basic metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                per_label_metrics[label] = {
                    'support': int(label_true_binary.sum()),
                    'predicted_positive': int(label_pred_binary.sum()),
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity,
                    'prevalence': label_true_binary.sum() / len(label_true_binary)
                }
                
                if label_true_binary.sum() > 0:  # Only if label exists in test set
                    # Precision-Recall curve
                    precision_curve, recall_curve, _ = precision_recall_curve(label_true_binary, label_prob)
                    ap_score = average_precision_score(label_true_binary, label_prob)
                    
                    # ROC curve
                    fpr, tpr, _ = roc_curve(label_true_binary, label_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    per_label_metrics[label].update({
                        'ap_score': ap_score,
                        'roc_auc': roc_auc,
                        'precision_curve': precision_curve.tolist(),
                        'recall_curve': recall_curve.tolist(),
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist()
                    })
                else:
                    per_label_metrics[label].update({
                        'ap_score': 0.0,
                        'roc_auc': 0.5,
                        'precision_curve': [],
                        'recall_curve': [],
                        'fpr': [],
                        'tpr': []
                    })
        
        # Calculate weighted averages for all classes
        all_supports = [metrics_data['support'] for metrics_data in per_label_metrics.values()]
        all_precisions = [metrics_data['precision'] for metrics_data in per_label_metrics.values()]
        all_recalls = [metrics_data['recall'] for metrics_data in per_label_metrics.values()]
        all_f1_scores = [metrics_data['f1_score'] for metrics_data in per_label_metrics.values()]
        all_specificities = [metrics_data['specificity'] for metrics_data in per_label_metrics.values()]
        all_prevalences = [metrics_data['prevalence'] for metrics_data in per_label_metrics.values()]
        all_ap_scores = [metrics_data['ap_score'] for metrics_data in per_label_metrics.values() if 'ap_score' in metrics_data and metrics_data['ap_score'] is not None]
        all_roc_aucs = [metrics_data['roc_auc'] for metrics_data in per_label_metrics.values() if 'roc_auc' in metrics_data and metrics_data['roc_auc'] is not None]
        
        # Filter out classes with ANY zero values in precision, recall, or f1
        valid_classes_data = [(label, metrics_data) for label, metrics_data in per_label_metrics.items() 
                             if metrics_data['precision'] > 0 and metrics_data['recall'] > 0 and metrics_data['f1_score'] > 0]
        
        if valid_classes_data:
            valid_precisions = [metrics_data['precision'] for label, metrics_data in valid_classes_data]
            valid_recalls = [metrics_data['recall'] for label, metrics_data in valid_classes_data]
            valid_f1_scores = [metrics_data['f1_score'] for label, metrics_data in valid_classes_data]
            valid_specificities = [metrics_data['specificity'] for label, metrics_data in valid_classes_data]
            valid_supports = [metrics_data['support'] for label, metrics_data in valid_classes_data]
            valid_ap_scores = [metrics_data['ap_score'] for label, metrics_data in valid_classes_data 
                              if 'ap_score' in metrics_data and metrics_data['ap_score'] is not None and metrics_data['ap_score'] > 0]
            valid_roc_aucs = [metrics_data['roc_auc'] for label, metrics_data in valid_classes_data 
                             if 'roc_auc' in metrics_data and metrics_data['roc_auc'] is not None and metrics_data['roc_auc'] > 0]
            valid_ap_supports = [metrics_data['support'] for label, metrics_data in valid_classes_data 
                                if 'ap_score' in metrics_data and metrics_data['ap_score'] is not None and metrics_data['ap_score'] > 0]
            valid_roc_supports = [metrics_data['support'] for label, metrics_data in valid_classes_data 
                                 if 'roc_auc' in metrics_data and metrics_data['roc_auc'] is not None and metrics_data['roc_auc'] > 0]
        else:
            valid_precisions = valid_recalls = valid_f1_scores = valid_specificities = valid_supports = []
            valid_ap_scores = valid_roc_aucs = valid_ap_supports = valid_roc_supports = []
        
        total_all_support = sum(all_supports)
        total_valid_support = sum(valid_supports) if valid_supports else 0
        
        # Calculate weighted averages for valid classes only
        weighted_all_precision = (sum(p * s for p, s in zip(valid_precisions, valid_supports)) / 
                                 total_valid_support) if total_valid_support > 0 else 0.0
        weighted_all_recall = (sum(r * s for r, s in zip(valid_recalls, valid_supports)) / 
                              total_valid_support) if total_valid_support > 0 else 0.0
        weighted_all_f1 = (sum(f * s for f, s in zip(valid_f1_scores, valid_supports)) / 
                           total_valid_support) if total_valid_support > 0 else 0.0
        weighted_all_specificity = (sum(sp * s for sp, s in zip(valid_specificities, valid_supports)) / 
                                   total_valid_support) if total_valid_support > 0 else 0.0
        weighted_all_prevalence = sum(pr * s for pr, s in zip(all_prevalences, all_supports)) / total_all_support if total_all_support > 0 else 0.0
        
        # For AP and AUC, only include valid classes that have these metrics and are > 0
        weighted_all_ap = (sum(a * s for a, s in zip(valid_ap_scores, valid_ap_supports)) / 
                          sum(valid_ap_supports)) if valid_ap_supports else 0.0
        weighted_all_roc_auc = (sum(a * s for a, s in zip(valid_roc_aucs, valid_roc_supports)) / 
                               sum(valid_roc_supports)) if valid_roc_supports else 0.0
        
        avg_metrics_all_classes = {
            'num_classes': len(per_label_metrics),
            'num_valid_classes': len(valid_classes_data),
            'total_support': total_all_support,
            # Unweighted averages (valid classes only)
            'avg_precision': np.mean(valid_precisions) if valid_precisions else 0.0,
            'avg_recall': np.mean(valid_recalls) if valid_recalls else 0.0,
            'avg_f1_score': np.mean(valid_f1_scores) if valid_f1_scores else 0.0,
            'avg_specificity': np.mean(valid_specificities) if valid_specificities else 0.0,
            'avg_prevalence': np.mean(all_prevalences),  # Keep all for prevalence as it can be 0
            'avg_ap_score': np.mean(valid_ap_scores) if valid_ap_scores else 0.0,
            'avg_roc_auc': np.mean(valid_roc_aucs) if valid_roc_aucs else 0.0,
            # Weighted averages (by support, valid classes only)
            'weighted_precision': weighted_all_precision,
            'weighted_recall': weighted_all_recall,
            'weighted_f1_score': weighted_all_f1,
            'weighted_specificity': weighted_all_specificity,
            'weighted_prevalence': weighted_all_prevalence,
            'weighted_ap_score': weighted_all_ap,
            'weighted_roc_auc': weighted_all_roc_auc,
            'class_names': list(per_label_metrics.keys())
        }
        
        # Calculate average metrics for classes with more than 50 supports
        high_support_classes = {label: metrics_data for label, metrics_data in per_label_metrics.items() 
                               if metrics_data['support'] > 50}
        
        if high_support_classes:
            # Filter high support classes with ANY zero values in precision, recall, or f1
            valid_hs_classes_data = [(label, metrics_data) for label, metrics_data in high_support_classes.items() 
                                    if metrics_data['precision'] > 0 and metrics_data['recall'] > 0 and metrics_data['f1_score'] > 0]
            
            total_hs_support = sum(metrics_data['support'] for metrics_data in high_support_classes.values())
            
            if valid_hs_classes_data:
                valid_hs_precisions = [metrics_data['precision'] for label, metrics_data in valid_hs_classes_data]
                valid_hs_recalls = [metrics_data['recall'] for label, metrics_data in valid_hs_classes_data]
                valid_hs_f1_scores = [metrics_data['f1_score'] for label, metrics_data in valid_hs_classes_data]
                valid_hs_specificities = [metrics_data['specificity'] for label, metrics_data in valid_hs_classes_data]
                valid_hs_supports = [metrics_data['support'] for label, metrics_data in valid_hs_classes_data]
                valid_hs_prevalences = [metrics_data['prevalence'] for label, metrics_data in valid_hs_classes_data]
                valid_hs_ap_scores = [metrics_data['ap_score'] for label, metrics_data in valid_hs_classes_data 
                                     if 'ap_score' in metrics_data and metrics_data['ap_score'] is not None and metrics_data['ap_score'] > 0]
                valid_hs_roc_aucs = [metrics_data['roc_auc'] for label, metrics_data in valid_hs_classes_data 
                                    if 'roc_auc' in metrics_data and metrics_data['roc_auc'] is not None and metrics_data['roc_auc'] > 0]
                valid_hs_ap_supports = [metrics_data['support'] for label, metrics_data in valid_hs_classes_data 
                                       if 'ap_score' in metrics_data and metrics_data['ap_score'] is not None and metrics_data['ap_score'] > 0]
                valid_hs_roc_supports = [metrics_data['support'] for label, metrics_data in valid_hs_classes_data 
                                        if 'roc_auc' in metrics_data and metrics_data['roc_auc'] is not None and metrics_data['roc_auc'] > 0]
            else:
                valid_hs_precisions = valid_hs_recalls = valid_hs_f1_scores = valid_hs_specificities = valid_hs_supports = []
                valid_hs_prevalences = valid_hs_ap_scores = valid_hs_roc_aucs = valid_hs_ap_supports = valid_hs_roc_supports = []
            
            total_valid_hs_support = sum(valid_hs_supports) if valid_hs_supports else 0
            all_hs_prevalences = [metrics_data['prevalence'] for metrics_data in high_support_classes.values()]
            all_hs_supports = [metrics_data['support'] for metrics_data in high_support_classes.values()]
            
            # Calculate weighted averages (by support, valid classes only)
            weighted_precision = (sum(p * s for p, s in zip(valid_hs_precisions, valid_hs_supports)) / 
                                 total_valid_hs_support) if total_valid_hs_support > 0 else 0.0
            weighted_recall = (sum(r * s for r, s in zip(valid_hs_recalls, valid_hs_supports)) / 
                              total_valid_hs_support) if total_valid_hs_support > 0 else 0.0
            weighted_f1 = (sum(f * s for f, s in zip(valid_hs_f1_scores, valid_hs_supports)) / 
                          total_valid_hs_support) if total_valid_hs_support > 0 else 0.0
            weighted_specificity = (sum(sp * s for sp, s in zip(valid_hs_specificities, valid_hs_supports)) / 
                                   total_valid_hs_support) if total_valid_hs_support > 0 else 0.0
            weighted_prevalence = sum(pr * s for pr, s in zip(all_hs_prevalences, all_hs_supports)) / total_hs_support
            
            # For AP and AUC, only include valid classes that have these metrics and are > 0
            weighted_ap = (sum(a * s for a, s in zip(valid_hs_ap_scores, valid_hs_ap_supports)) / 
                          sum(valid_hs_ap_supports)) if valid_hs_ap_supports else 0.0
            weighted_roc_auc = (sum(a * s for a, s in zip(valid_hs_roc_aucs, valid_hs_roc_supports)) / 
                               sum(valid_hs_roc_supports)) if valid_hs_roc_supports else 0.0
            
            avg_metrics_high_support = {
                'num_classes': len(high_support_classes),
                'num_valid_classes': len(valid_hs_classes_data),
                'total_support': total_hs_support,
                # Unweighted averages (valid classes only)
                'avg_precision': np.mean(valid_hs_precisions) if valid_hs_precisions else 0.0,
                'avg_recall': np.mean(valid_hs_recalls) if valid_hs_recalls else 0.0,
                'avg_f1_score': np.mean(valid_hs_f1_scores) if valid_hs_f1_scores else 0.0,
                'avg_specificity': np.mean(valid_hs_specificities) if valid_hs_specificities else 0.0,
                'avg_prevalence': np.mean(all_hs_prevalences),  # Keep all for prevalence
                'avg_ap_score': np.mean(valid_hs_ap_scores) if valid_hs_ap_scores else 0.0,
                'avg_roc_auc': np.mean(valid_hs_roc_aucs) if valid_hs_roc_aucs else 0.0,
                # Weighted averages (by support, valid classes only)
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1_score': weighted_f1,
                'weighted_specificity': weighted_specificity,
                'weighted_prevalence': weighted_prevalence,
                'weighted_ap_score': weighted_ap,
                'weighted_roc_auc': weighted_roc_auc,
                'class_names': list(high_support_classes.keys())
            }
        else:
            avg_metrics_high_support = {
                'num_classes': 0,
                'num_valid_classes': 0,
                'total_support': 0,
                # Unweighted averages
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_f1_score': 0.0,
                'avg_specificity': 0.0,
                'avg_prevalence': 0.0,
                'avg_ap_score': 0.0,
                'avg_roc_auc': 0.0,
                # Weighted averages (by support)
                'weighted_precision': 0.0,
                'weighted_recall': 0.0,
                'weighted_f1_score': 0.0,
                'weighted_specificity': 0.0,
                'weighted_prevalence': 0.0,
                'weighted_ap_score': 0.0,
                'weighted_roc_auc': 0.0,
                'class_names': []
            }
        
        return {
            'overall_metrics': metrics,
            'per_label_metrics': per_label_metrics,
            'avg_metrics_all_classes': avg_metrics_all_classes,
            'avg_metrics_high_support_classes': avg_metrics_high_support,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'pred_probs': pred_probs,
            'label_names': list(self.mlb.classes_),
            'label_names_with_display': [f"{label_id}: {self.get_topic_name(label_id)}" if self.target_field == "topics" 
                                       else str(label_id) for label_id in self.mlb.classes_]
        }
    
    def create_confusion_matrices(self, results: Dict, save_dir: str = "evaluation_plots"):
        """
        Create confusion matrices for each label.
        
        Args:
            results: Evaluation results from evaluate_test_set
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        true_labels = results['true_labels']
        pred_labels = results['pred_labels']
        label_names = results['label_names']
        
        if self.is_multilabel:
            # Create multilabel confusion matrices
            cm_matrices = multilabel_confusion_matrix(true_labels, pred_labels)
            
            # Plot confusion matrices for top labels
            top_labels_indices = []
            for i, label in enumerate(label_names):
                if true_labels[:, i].sum() >= 5:  # Only labels with at least 5 samples
                    top_labels_indices.append(i)
        else:
            # For binary classification, create a single multiclass confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            
            # Create a single confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_names, yticklabels=label_names)
            plt.title(f'Confusion Matrix: {self.target_field}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            # Save the plot
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Confusion matrix saved to: {os.path.join(save_dir, 'confusion_matrix.png')}")
            return
        
        if not top_labels_indices:
            print("No labels with sufficient samples for confusion matrix.")
            return
        
        # Limit to top 12 labels for visualization
        top_labels_indices = top_labels_indices[:12]
        
        n_rows = (len(top_labels_indices) + 3) // 4
        n_cols = min(4, len(top_labels_indices))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, label_idx in enumerate(top_labels_indices):
            row = idx // n_cols
            col = idx % n_cols
            
            cm = cm_matrices[label_idx]
            label_name = label_names[label_idx]
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[row, col],
                xticklabels=['Not ' + label_name, label_name],
                yticklabels=['Not ' + label_name, label_name]
            )
            axes[row, col].set_title(f'Confusion Matrix: {label_name}')
            axes[row, col].set_ylabel('True Label')
            axes[row, col].set_xlabel('Predicted Label')
        
        # Hide empty subplots
        for idx in range(len(top_labels_indices), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_precision_recall_curves(self, results: Dict, save_dir: str = "evaluation_plots"):
        """
        Create precision-recall curves for top labels.
        
        Args:
            results: Evaluation results from evaluate_test_set
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        per_label_metrics = results['per_label_metrics']
        
        # Sort labels by support (number of positive examples)
        sorted_labels = sorted(
            per_label_metrics.items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )
        
        # Plot top 10 labels
        top_labels = sorted_labels[:10]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (label, metrics) in enumerate(top_labels):
            if idx >= 10:
                break
                
            precision = np.array(metrics['precision_curve'])
            recall = np.array(metrics['recall_curve'])
            ap_score = metrics['ap_score']
            
            axes[idx].plot(recall, precision, linewidth=2, 
                          label=f'AP = {ap_score:.3f}')
            axes[idx].set_xlabel('Recall')
            axes[idx].set_ylabel('Precision')
            axes[idx].set_title(f'PR Curve: {label}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(top_labels), 10):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, "precision_recall_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_roc_curves(self, results: Dict, save_dir: str = "evaluation_plots"):
        """
        Create ROC curves for top labels.
        
        Args:
            results: Evaluation results from evaluate_test_set
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        per_label_metrics = results['per_label_metrics']
        
        # Sort labels by support
        sorted_labels = sorted(
            per_label_metrics.items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )
        
        # Plot top 10 labels
        top_labels = sorted_labels[:10]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (label, metrics) in enumerate(top_labels):
            if idx >= 10:
                break
                
            fpr = np.array(metrics['fpr'])
            tpr = np.array(metrics['tpr'])
            roc_auc = metrics['roc_auc']
            
            axes[idx].plot(fpr, tpr, linewidth=2, 
                          label=f'AUC = {roc_auc:.3f}')
            axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.6)
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].set_title(f'ROC Curve: {label}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(top_labels), 10):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Ensure plots directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_predictions(self, results: Dict) -> Dict[str, any]:
        """
        Analyze prediction patterns and errors.
        
        Args:
            results: Evaluation results from evaluate_test_set
            
        Returns:
            Analysis dictionary
        """
        true_labels = results['true_labels']
        pred_labels = results['pred_labels']
        pred_probs = results['pred_probs']
        label_names = results['label_names']
        
        analysis = {}
        
        # Overall statistics
        analysis['total_samples'] = len(true_labels)
        analysis['total_labels'] = len(label_names)
        
        if self.is_multilabel:
            # Multilabel analysis
            true_label_counts = true_labels.sum(axis=0)
            pred_label_counts = pred_labels.sum(axis=0)
            
            analysis['true_label_distribution'] = {
                label_names[i]: int(count) for i, count in enumerate(true_label_counts)
            }
            analysis['pred_label_distribution'] = {
                label_names[i]: int(count) for i, count in enumerate(pred_label_counts)
            }
            
            # Multi-label statistics
            true_labels_per_sample = true_labels.sum(axis=1)
            pred_labels_per_sample = pred_labels.sum(axis=1)
            
            analysis['avg_true_labels_per_sample'] = float(np.mean(true_labels_per_sample))
            analysis['avg_pred_labels_per_sample'] = float(np.mean(pred_labels_per_sample))
            
            # Error analysis for multilabel
            errors = []
            
            for i in range(len(true_labels)):
                true_set = set(np.where(true_labels[i])[0])
                pred_set = set(np.where(pred_labels[i])[0])
                
                false_positives = pred_set - true_set
                false_negatives = true_set - pred_set
                
                if false_positives or false_negatives:
                    errors.append({
                        'sample_idx': i,
                        'true_labels': [label_names[j] for j in true_set],
                        'pred_labels': [label_names[j] for j in pred_set],
                        'false_positives': [label_names[j] for j in false_positives],
                        'false_negatives': [label_names[j] for j in false_negatives],
                        'text': self.test_df.iloc[i][self.text_field][:200] + "..." if len(self.test_df.iloc[i][self.text_field]) > 200 else self.test_df.iloc[i][self.text_field]
                    })
            
            # Most confused label pairs
            confusion_pairs = defaultdict(int)
            for error in errors:
                for fp in error['false_positives']:
                    for fn in error['false_negatives']:
                        confusion_pairs[(fn, fp)] += 1
                        
        else:
            # Binary classification analysis
            from collections import Counter
            
            true_label_counts = Counter(true_labels)
            pred_label_counts = Counter(pred_labels)
            
            analysis['true_label_distribution'] = {
                label_names[i]: int(true_label_counts.get(i, 0)) for i in range(len(label_names))
            }
            analysis['pred_label_distribution'] = {
                label_names[i]: int(pred_label_counts.get(i, 0)) for i in range(len(label_names))
            }
            
            # For binary classification, each sample has exactly one label
            analysis['avg_true_labels_per_sample'] = 1.0
            analysis['avg_pred_labels_per_sample'] = 1.0
            
            # Error analysis for binary classification
            errors = []
            
            for i in range(len(true_labels)):
                if true_labels[i] != pred_labels[i]:
                    errors.append({
                        'sample_idx': i,
                        'true_label': label_names[true_labels[i]],
                        'pred_label': label_names[pred_labels[i]],
                        'confidence': float(pred_probs[i][pred_labels[i]]),
                        'text': self.test_df.iloc[i][self.text_field][:200] + "..." if len(self.test_df.iloc[i][self.text_field]) > 200 else self.test_df.iloc[i][self.text_field]
                    })
            
            # Most confused label pairs for binary classification
            confusion_pairs = defaultdict(int)
            for error in errors:
                confusion_pairs[(error['true_label'], error['pred_label'])] += 1
        
        analysis['error_count'] = len(errors)
        analysis['error_rate'] = len(errors) / len(true_labels)
        analysis['sample_errors'] = errors[:10]  # First 10 errors for inspection
        
        # Convert confusion pairs to serializable format
        most_confused_pairs = {}
        for (true_label, pred_label), count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
            key = f"{true_label} -> {pred_label}"
            most_confused_pairs[key] = count
        
        analysis['most_confused_pairs'] = most_confused_pairs
        
        return analysis
    
    def create_per_label_statistics_table(self, results: Dict, save_dir: str = "evaluation_results"):
        """
        Create a detailed per-label statistics table and save as CSV.
        
        Args:
            results: Evaluation results from evaluate_test_set
            save_dir: Directory to save the table
        """
        os.makedirs(save_dir, exist_ok=True)
        
        per_label_metrics = results['per_label_metrics']
        
        # Create comprehensive per-label table
        table_data = []
        for label, metrics in per_label_metrics.items():
            # Create display label - for topics, include both ID and name; for others, use as-is
            if self.target_field == "topics":
                label_name = self.get_topic_name(label)
                display_label = f"{label}: {label_name}"
            else:
                display_label = label  # For sentiment, status, type - use the label directly
                
            row = {
                'Label': display_label,
                'Label_ID': label,  # Keep original ID for reference
                'Support': metrics['support'],
                'Predicted_Positive': metrics['predicted_positive'],
                'True_Positives': metrics['true_positives'],
                'False_Positives': metrics['false_positives'],
                'True_Negatives': metrics['true_negatives'],
                'False_Negatives': metrics['false_negatives'],
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Prevalence': f"{metrics['prevalence']:.4f}",
                'AP_Score': f"{metrics['ap_score']:.4f}",
                'ROC_AUC': f"{metrics['roc_auc']:.4f}"
            }
            table_data.append(row)
        
        # Convert to DataFrame and sort by support (descending)
        df = pd.DataFrame(table_data)
        df['Support'] = df['Support'].astype(int)
        df = df.sort_values('Support', ascending=False)
        
        # Save as CSV
        csv_path = os.path.join(save_dir, "per_label_statistics.csv")
        df.to_csv(csv_path, index=False)
        
        # Create a summary table for all labels
        all_labels = df  # Show all labels instead of just top 20
        
        # Print summary to console
        print("\n" + "="*120)
        print("PER-LABEL STATISTICS SUMMARY (All Labels)")
        print("="*120)
        print(f"{'Label':<50} {'Support':<8} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AP':<6} {'AUC':<6}")
        print("-"*120)
        
        for _, row in all_labels.iterrows():
            print(f"{row['Label']:<50} {row['Support']:<8} {row['Precision']:<6} {row['Recall']:<6} "
                  f"{row['F1_Score']:<6} {row['AP_Score']:<6} {row['ROC_AUC']:<6}")
        
        print(f"\nDetailed per-label statistics saved to: {csv_path}")
        
        return df
    
    def create_performance_distribution_plots(self, results: Dict, save_dir: str = "evaluation_plots"):
        """
        Create distribution plots for various performance metrics.
        
        Args:
            results: Evaluation results from evaluate_test_set
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        per_label_metrics = results['per_label_metrics']
        
        # Extract metrics for plotting
        labels = []
        precisions = []
        recalls = []
        f1_scores = []
        supports = []
        ap_scores = []
        roc_aucs = []
        
        for label, metrics in per_label_metrics.items():
            if metrics['support'] > 0:  # Only include labels with support
                labels.append(label)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1_score'])
                supports.append(metrics['support'])
                ap_scores.append(metrics['ap_score'])
                roc_aucs.append(metrics['roc_auc'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Precision distribution
        axes[0, 0].hist(precisions, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_ylabel('Number of Labels')
        axes[0, 0].set_title('Distribution of Precision Scores')
        axes[0, 0].axvline(np.mean(precisions), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(precisions):.3f}')
        axes[0, 0].legend()
        
        # Recall distribution
        axes[0, 1].hist(recalls, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Number of Labels')
        axes[0, 1].set_title('Distribution of Recall Scores')
        axes[0, 1].axvline(np.mean(recalls), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(recalls):.3f}')
        axes[0, 1].legend()
        
        # F1 Score distribution
        axes[0, 2].hist(f1_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('F1 Score')
        axes[0, 2].set_ylabel('Number of Labels')
        axes[0, 2].set_title('Distribution of F1 Scores')
        axes[0, 2].axvline(np.mean(f1_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[0, 2].legend()
        
        # Support distribution (log scale)
        axes[1, 0].hist(supports, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Support (Log Scale)')
        axes[1, 0].set_ylabel('Number of Labels')
        axes[1, 0].set_title('Distribution of Label Support')
        axes[1, 0].set_yscale('log')
        axes[1, 0].axvline(np.mean(supports), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(supports):.1f}')
        axes[1, 0].legend()
        
        # AP Score distribution
        axes[1, 1].hist(ap_scores, bins=20, alpha=0.7, color='brown', edgecolor='black')
        axes[1, 1].set_xlabel('Average Precision Score')
        axes[1, 1].set_ylabel('Number of Labels')
        axes[1, 1].set_title('Distribution of AP Scores')
        axes[1, 1].axvline(np.mean(ap_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(ap_scores):.3f}')
        axes[1, 1].legend()
        
        # ROC AUC distribution
        axes[1, 2].hist(roc_aucs, bins=20, alpha=0.7, color='pink', edgecolor='black')
        axes[1, 2].set_xlabel('ROC AUC Score')
        axes[1, 2].set_ylabel('Number of Labels')
        axes[1, 2].set_title('Distribution of ROC AUC Scores')
        axes[1, 2].axvline(np.mean(roc_aucs), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(roc_aucs):.3f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "performance_distributions.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance distribution plots saved to: {os.path.join(save_dir, 'performance_distributions.png')}")

    def generate_report(self, results: Dict, analysis: Dict, save_path: str = "evaluation_report.json"):
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            analysis: Prediction analysis
            save_path: Path to save the report
        """
        overall_metrics = results['overall_metrics']
        per_label_metrics = results['per_label_metrics']
        all_classes_metrics = results['avg_metrics_all_classes']
        high_support_metrics = results['avg_metrics_high_support_classes']
        
        # Create comprehensive report
        report = {
            "model_info": {
                "model_path": self.model_path,
                "num_test_samples": overall_metrics['num_test_samples'],
                "num_labels": overall_metrics['num_labels'],
                "is_multilabel": self.is_multilabel,
                "target_field": self.target_field
            }
        }
        
        # Add overall performance metrics based on classification type
        if self.is_multilabel:
            # For multilabel classification
            report["overall_performance"] = {
                "subset_accuracy": overall_metrics.get('subset_accuracy', 0.0),
                "hamming_loss": overall_metrics.get('hamming_loss', 0.0),
                "jaccard_score": overall_metrics.get('jaccard_score', 0.0),
                "f1_macro": all_classes_metrics['avg_f1_score'],
                "f1_micro": overall_metrics.get('f1_micro', 0.0),
                "precision_macro": all_classes_metrics['avg_precision'],
                "precision_micro": overall_metrics.get('precision_micro', 0.0),
                "recall_macro": all_classes_metrics['avg_recall'],
                "recall_micro": overall_metrics.get('recall_micro', 0.0)
            }
        else:
            # For binary classification
            report["overall_performance"] = {
                "accuracy": overall_metrics.get('accuracy', 0.0),
                "f1_macro": overall_metrics.get('f1_macro', 0.0),
                "f1_weighted": overall_metrics.get('f1_weighted', 0.0),
                "f1_micro": overall_metrics.get('f1_micro', 0.0),
                "precision_macro": overall_metrics.get('precision_macro', 0.0),
                "precision_weighted": overall_metrics.get('precision_weighted', 0.0),
                "precision_micro": overall_metrics.get('precision_micro', 0.0),
                "recall_macro": overall_metrics.get('recall_macro', 0.0),
                "recall_weighted": overall_metrics.get('recall_weighted', 0.0),
                "recall_micro": overall_metrics.get('recall_micro', 0.0)
            }
        
        # Add common performance sections
        report.update({
            "all_classes_performance": {
                "num_classes": all_classes_metrics['num_classes'],
                "total_support": all_classes_metrics['total_support'],
                # Unweighted averages
                "avg_precision": all_classes_metrics['avg_precision'],
                "avg_recall": all_classes_metrics['avg_recall'],
                "avg_f1_score": all_classes_metrics['avg_f1_score'],
                "avg_specificity": all_classes_metrics['avg_specificity'],
                "avg_prevalence": all_classes_metrics['avg_prevalence'],
                "avg_ap_score": all_classes_metrics['avg_ap_score'],
                "avg_roc_auc": all_classes_metrics['avg_roc_auc'],
                # Weighted averages (by support)
                "weighted_precision": all_classes_metrics['weighted_precision'],
                "weighted_recall": all_classes_metrics['weighted_recall'],
                "weighted_f1_score": all_classes_metrics['weighted_f1_score'],
                "weighted_specificity": all_classes_metrics['weighted_specificity'],
                "weighted_prevalence": all_classes_metrics['weighted_prevalence'],
                "weighted_ap_score": all_classes_metrics['weighted_ap_score'],
                "weighted_roc_auc": all_classes_metrics['weighted_roc_auc'],
                "class_names": all_classes_metrics['class_names']
            },
            "high_support_classes_performance": {
                "num_classes": high_support_metrics['num_classes'],
                "total_support": high_support_metrics['total_support'],
                # Unweighted averages
                "avg_precision": high_support_metrics['avg_precision'],
                "avg_recall": high_support_metrics['avg_recall'],
                "avg_f1_score": high_support_metrics['avg_f1_score'],
                "avg_specificity": high_support_metrics['avg_specificity'],
                "avg_prevalence": high_support_metrics['avg_prevalence'],
                "avg_ap_score": high_support_metrics['avg_ap_score'],
                "avg_roc_auc": high_support_metrics['avg_roc_auc'],
                # Weighted averages (by support)
                "weighted_precision": high_support_metrics['weighted_precision'],
                "weighted_recall": high_support_metrics['weighted_recall'],
                "weighted_f1_score": high_support_metrics['weighted_f1_score'],
                "weighted_specificity": high_support_metrics['weighted_specificity'],
                "weighted_prevalence": high_support_metrics['weighted_prevalence'],
                "weighted_ap_score": high_support_metrics['weighted_ap_score'],
                "weighted_roc_auc": high_support_metrics['weighted_roc_auc'],
                "class_names": high_support_metrics['class_names']
            },
            "label_performance": {},
            "prediction_analysis": analysis
        })
        
        # Add per-label performance
        for label, metrics in per_label_metrics.items():
            report["label_performance"][label] = {
                "support": metrics['support'],
                "predicted_positive": metrics['predicted_positive'],
                "true_positives": metrics['true_positives'],
                "false_positives": metrics['false_positives'],
                "true_negatives": metrics['true_negatives'],
                "false_negatives": metrics['false_negatives'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1_score": metrics['f1_score'],
                "specificity": metrics['specificity'],
                "prevalence": metrics['prevalence'],
                "average_precision": metrics['ap_score'],
                "roc_auc": metrics['roc_auc']
            }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to: {save_path}")
        
        return report


def main():
    """Main evaluation function using configuration from config.py."""
    
    # Use configuration instead of argument parsing
    model_path = "models/bert_sum_to_topic_180825.pth"  # Default model path
    data_dir = DATA_CONFIG["processed_data_dir"]
    text_field = DATA_CONFIG["text_field"]
    target_field = DATA_CONFIG["target_field"]
    use_timestamp_folder = OUTPUT_CONFIG["use_timestamp_folders"]
    output_dir = "evaluation_results"  # Default output directory
    skip_plots = False  # Default to generating plots
    
    print("="*60)
    print("BERT MODEL EVALUATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model path: {model_path}")
    print(f"  Data directory: {data_dir}")
    print(f"  Text field: {text_field}")
    print(f"  Target field: {target_field}")
    print(f"  Use timestamp folder: {use_timestamp_folder}")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    # Create timestamp-based output structure if requested
    if use_timestamp_folder:
        from datetime import datetime
        timestamp, output_base_path = create_timestamp_folder("evaluation_outputs")
        paths = get_timestamp_paths(output_base_path)
        
        # Use evaluation_results subdirectory within timestamp folder
        output_dir = paths["evaluation_results"]
        plot_dir = os.path.join(paths["evaluation_results"], "plots")
        
        print(f"Created timestamp folder: {output_base_path}")
        print(f"Evaluation results will be saved under: {output_dir}")
    else:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, "plots")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, data_dir, text_field, target_field=target_field)
    
    print("Starting model evaluation...")
    print(f"Using text field for evaluation: {text_field}")
    print(f"Using target field: {evaluator.target_field}")
    
    # Evaluate test set
    results = evaluator.evaluate_test_set()
    
    # Analyze predictions
    analysis = evaluator.analyze_predictions(results)
    
    # Create per-label statistics table
    per_label_df = evaluator.create_per_label_statistics_table(results, output_dir)
    
    # Generate report
    report_path = os.path.join(output_dir, "evaluation_report.json")
    report = evaluator.generate_report(results, analysis, report_path)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)

    metrics = results['overall_metrics']
    all_classes_metrics = results['avg_metrics_all_classes']
    print(f"Test Samples: {metrics['num_test_samples']}")
    print(f"Number of Labels: {metrics['num_labels']}")
    print(f"Classification Type: {'Multilabel' if evaluator.is_multilabel else 'Binary'}")
    
    if evaluator.is_multilabel:
        # Multilabel metrics
        print(f"Subset Accuracy: {metrics.get('subset_accuracy', 0.0):.4f}")
        print(f"Hamming Loss: {metrics.get('hamming_loss', 0.0):.4f}")
        print(f"Jaccard Score: {metrics.get('jaccard_score', 0.0):.4f}")
        print(f"F1 Score (Macro): {all_classes_metrics['avg_f1_score']:.4f}")
        print(f"F1 Score (Micro): {metrics.get('f1_micro', 0.0):.4f}")
    else:
        # Binary classification metrics
        print(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        print(f"F1 Score (Macro): {metrics.get('f1_macro', 0.0):.4f}")
        print(f"F1 Score (Weighted): {metrics.get('f1_weighted', 0.0):.4f}")
        print(f"F1 Score (Micro): {metrics.get('f1_micro', 0.0):.4f}")
    
    print(f"\nError Rate: {analysis['error_rate']:.2%}")
    print(f"Avg True Labels per Sample: {analysis['avg_true_labels_per_sample']:.2f}")
    print(f"Avg Predicted Labels per Sample: {analysis['avg_pred_labels_per_sample']:.2f}")
    
    # Print all classes metrics
    all_classes_metrics = results['avg_metrics_all_classes']
    print("\n" + "="*50)
    print("ALL CLASSES METRICS")
    print("="*50)
    print(f"Total classes: {all_classes_metrics['num_classes']}")
    print(f"Valid classes: {all_classes_metrics['num_valid_classes']}")
    print(f"Total support: {all_classes_metrics['total_support']}")
    
    print("\nUNWEIGHTED (Simple Average):")
    print(f"  Precision: {all_classes_metrics['avg_precision']:.4f}")
    print(f"  Recall: {all_classes_metrics['avg_recall']:.4f}")
    print(f"  F1-Score: {all_classes_metrics['avg_f1_score']:.4f}")
    print(f"  Specificity: {all_classes_metrics['avg_specificity']:.4f}")
    print(f"  AP Score: {all_classes_metrics['avg_ap_score']:.4f}")
    print(f"  ROC AUC: {all_classes_metrics['avg_roc_auc']:.4f}")
    print(f"  Prevalence: {all_classes_metrics['avg_prevalence']:.4f}")
    
    print("\nWEIGHTED (By Support):")
    print(f"  Precision: {all_classes_metrics['weighted_precision']:.4f}")
    print(f"  Recall: {all_classes_metrics['weighted_recall']:.4f}")
    print(f"  F1-Score: {all_classes_metrics['weighted_f1_score']:.4f}")
    print(f"  Specificity: {all_classes_metrics['weighted_specificity']:.4f}")
    print(f"  AP Score: {all_classes_metrics['weighted_ap_score']:.4f}")
    print(f"  ROC AUC: {all_classes_metrics['weighted_roc_auc']:.4f}")
    print(f"  Prevalence: {all_classes_metrics['weighted_prevalence']:.4f}")
    
    # Print high-support classes metrics
    high_support_metrics = results['avg_metrics_high_support_classes']
    print("\n" + "="*50)
    print("HIGH-SUPPORT CLASSES METRICS (>50 samples)")
    print("="*50)
    print(f"High-support classes: {high_support_metrics['num_classes']}")
    print(f"Valid classes: {high_support_metrics['num_valid_classes']}")
    print(f"Total support: {high_support_metrics['total_support']}")
    
    if high_support_metrics['num_classes'] > 0:
        print("\nUNWEIGHTED (Simple Average):")
        print(f"  Precision: {high_support_metrics['avg_precision']:.4f}")
        print(f"  Recall: {high_support_metrics['avg_recall']:.4f}")
        print(f"  F1-Score: {high_support_metrics['avg_f1_score']:.4f}")
        print(f"  Specificity: {high_support_metrics['avg_specificity']:.4f}")
        print(f"  AP Score: {high_support_metrics['avg_ap_score']:.4f}")
        print(f"  ROC AUC: {high_support_metrics['avg_roc_auc']:.4f}")
        print(f"  Prevalence: {high_support_metrics['avg_prevalence']:.4f}")
        
        print("\nWEIGHTED (By Support):")
        print(f"  Precision: {high_support_metrics['weighted_precision']:.4f}")
        print(f"  Recall: {high_support_metrics['weighted_recall']:.4f}")
        print(f"  F1-Score: {high_support_metrics['weighted_f1_score']:.4f}")
        print(f"  Specificity: {high_support_metrics['weighted_specificity']:.4f}")
        print(f"  AP Score: {high_support_metrics['weighted_ap_score']:.4f}")
        print(f"  ROC AUC: {high_support_metrics['weighted_roc_auc']:.4f}")
        print(f"  Prevalence: {high_support_metrics['weighted_prevalence']:.4f}")
        
        print(f"\nHigh-support classes ({len(high_support_metrics['class_names'])}):")
        for class_name in high_support_metrics['class_names'][:10]:  # Show first 10
            class_metrics = results['per_label_metrics'][class_name]
            print(f"  {class_name}: {evaluator.get_topic_name(class_name)} (support: {class_metrics['support']})")
        if len(high_support_metrics['class_names']) > 10:
            print(f"  ... and {len(high_support_metrics['class_names']) - 10} more classes")
    else:
        print("No classes found with more than 50 support samples.")
    
    # Generate plots
    if not skip_plots:
        print("\nGenerating evaluation plots...")
        
        try:
            evaluator.create_confusion_matrices(results, plot_dir)
            evaluator.create_precision_recall_curves(results, plot_dir)
            evaluator.create_roc_curves(results, plot_dir)
            evaluator.create_performance_distribution_plots(results, plot_dir)
            print(f"Plots saved to: {plot_dir}")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    print(f"\nEvaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
